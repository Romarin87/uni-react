"""Step-based joint task trainer."""
from __future__ import annotations

import csv
import dataclasses
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ...configs.joint import JointConfig
from ...training.batch import move_batch_to_device
from ...training.checkpoint import validate_restart_config
from ...training.distributed import is_main_process
from .data_plan import DataPlan


class JointTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: JointConfig,
        data_plan: DataPlan,
        distributed: bool,
        rank: int,
        world_size: int,
        device: torch.device,
        logger=None,
    ) -> None:
        self.raw_model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.data_plan = data_plan
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.logger = logger
        self.model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        ) if distributed else model
        self.out_dir = Path(cfg.run_value("out_dir", "runs/joint"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.best_val = float("inf")
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(cfg.run_value("seed", 42)) + rank)

        self.active_tasks = list(cfg.active_train_tasks)
        probs = torch.tensor(
            [float(cfg.schedule["sample_prob"].get(name, 0.0)) for name in self.active_tasks],
            dtype=torch.float64,
        )
        self.sample_probs = (probs / probs.sum()).float()
        self._task_to_idx = {name: i for i, name in enumerate(self.active_tasks)}

        self.train_loaders = self._build_train_loaders()
        self.val_loaders = self._build_val_loaders()
        self._train_sampler_epochs = {name: 0 for name in self.train_loaders}
        for name, loader in self.train_loaders.items():
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._train_sampler_epochs[name])
        self._train_iters = {name: iter(loader) for name, loader in self.train_loaders.items()}
        self._task_steps = {name: 0 for name in self.active_tasks}
        self._sample_counts = {name: 0 for name in self.active_tasks}

        self.max_steps = self._resolve_max_steps()
        self.estimated_joint_epoch_steps = self._estimate_joint_epoch_steps()
        self.metrics_file = self.out_dir / str(
            cfg.advanced_value("diagnostics", "metrics_file", default="metrics.csv")
        )
        self._csv_columns: Optional[List[str]] = None

    def _build_train_loaders(self) -> Dict[str, DataLoader]:
        loaders: Dict[str, DataLoader] = {}
        num_workers = int(self.cfg.run_value("num_workers", 0))
        pin_memory = self.device.type == "cuda"
        for name in self.active_tasks:
            task_data = self.data_plan.task_data[name]
            if task_data.train is None:
                continue
            sampler = DistributedSampler(task_data.train.dataset, shuffle=True, drop_last=True) if self.distributed else None
            loaders[name] = DataLoader(
                task_data.train.dataset,
                batch_size=task_data.adapter.batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
                drop_last=True,
                collate_fn=task_data.adapter.collate_fn,
            )
        return loaders

    def _build_val_loaders(self) -> Dict[str, DataLoader]:
        loaders: Dict[str, DataLoader] = {}
        num_workers = int(self.cfg.run_value("num_workers", 0))
        pin_memory = self.device.type == "cuda"
        max_batches = int(self.cfg.evaluation_value("max_val_batches_per_task", 0) or 0)
        del max_batches  # Applied in eval loop, not loader construction.
        for name, task_data in self.data_plan.task_data.items():
            if task_data.val is None:
                continue
            sampler = DistributedSampler(task_data.val.dataset, shuffle=False, drop_last=False) if self.distributed else None
            loaders[name] = DataLoader(
                task_data.val.dataset,
                batch_size=task_data.adapter.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
                drop_last=False,
                collate_fn=task_data.adapter.collate_fn,
            )
        return loaders

    def _resolve_max_steps(self) -> int:
        opt = self.cfg.optimization
        if opt.get("train_unit", "steps") == "steps":
            max_steps = int(opt["max_steps"])
            limit = int(self.cfg.advanced_value("limits", "train_batch_limit", default=0) or 0)
            return min(max_steps, limit) if limit > 0 else max_steps
        ref = opt["epoch_reference_task"]
        steps_per_epoch = max(1, len(self.train_loaders[ref]))
        if is_main_process(self.rank):
            print(
                "[joint:warning] train_unit=epochs uses epoch_reference_task="
                f"{ref}; steps_per_epoch={steps_per_epoch}. steps mode is recommended."
            )
        max_steps = int(opt["epochs"]) * steps_per_epoch
        limit = int(self.cfg.advanced_value("limits", "train_batch_limit", default=0) or 0)
        return min(max_steps, limit) if limit > 0 else max_steps

    def _estimate_joint_epoch_steps(self) -> Optional[float]:
        estimates = []
        for name in self.active_tasks:
            loader = self.train_loaders.get(name)
            if loader is None:
                continue
            prob = float(self.sample_probs[self._task_to_idx[name]].item())
            if prob <= 0:
                continue
            estimates.append(float(len(loader)) / prob)
        if not estimates:
            return None
        return max(estimates)

    def _progress(self) -> float:
        return min(float(self.global_step) / max(float(self.max_steps), 1.0), 1.0)

    def _est_epoch(self) -> float:
        if not self.estimated_joint_epoch_steps:
            return 0.0
        return float(self.global_step) / self.estimated_joint_epoch_steps

    def _estimated_total_epochs(self) -> float:
        if not self.estimated_joint_epoch_steps:
            return 0.0
        return float(self.max_steps) / self.estimated_joint_epoch_steps

    def _lr_factor(self) -> float:
        kind = self.cfg.optimization.get("lr_scheduler", "none")
        warmup = int(self.cfg.optimization.get("warmup_steps", 0) or 0)
        step = max(self.global_step, 0)
        if kind == "none":
            return 1.0
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if kind == "linear":
            remain = max(self.max_steps - warmup, 1)
            return max(0.0, 1.0 - float(step - warmup) / float(remain))
        if kind == "cosine":
            remain = max(self.max_steps - warmup, 1)
            progress = min(max(float(step - warmup) / float(remain), 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported lr_scheduler: {kind}")

    def _set_active_lrs(self, task_name: str) -> None:
        factor = self._lr_factor()
        desc_lr = float(self.cfg.learning_rates["descriptor"][task_name]) * factor
        head_lr = float(self.cfg.learning_rates["head"][task_name]) * factor
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if name == "descriptor":
                group["lr"] = desc_lr
            elif name == f"task.{task_name}":
                group["lr"] = head_lr
            else:
                group["lr"] = 0.0

    def _task_weight(self, task_name: str) -> float:
        initial = float(self.cfg.loss_weights.get("initial", {}).get(task_name, 1.0))
        final = float(self.cfg.loss_weights.get("final", {}).get(task_name, initial))
        progress = self._progress()
        return initial + (final - initial) * progress

    def _sample_task(self) -> str:
        if self.distributed and dist.is_available() and dist.is_initialized():
            if self.rank == 0:
                sampled = int(torch.multinomial(self.sample_probs, 1, generator=self._rng).item())
            else:
                sampled = 0
            device = self.device if self.device.type == "cuda" else torch.device("cpu")
            idx_tensor = torch.tensor([sampled], dtype=torch.long, device=device)
            dist.broadcast(idx_tensor, src=0)
            idx = int(idx_tensor.item())
        else:
            idx = int(torch.multinomial(self.sample_probs, 1, generator=self._rng).item())
        return self.active_tasks[idx]

    def _next_batch(self, task_name: str):
        try:
            return next(self._train_iters[task_name])
        except StopIteration:
            loader = self.train_loaders[task_name]
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                self._train_sampler_epochs[task_name] += 1
                sampler.set_epoch(self._train_sampler_epochs[task_name])
            self._train_iters[task_name] = iter(self.train_loaders[task_name])
            return next(self._train_iters[task_name])

    def fit(self) -> None:
        if is_main_process(self.rank):
            self._print_init()
        eval_every = int(self.cfg.evaluation_value("eval_every_steps", 0) or 0)
        save_every = int(self.cfg.checkpoint_value("save_every_steps", 0) or 0)
        while self.global_step < self.max_steps:
            self.train_step()
            if eval_every > 0 and self.global_step % eval_every == 0:
                metrics = self.eval_all()
                self._log_eval(metrics)
                self._write_metrics_csv(metrics)
                val = float(metrics.get("weighted_val_loss", float("inf")))
                if is_main_process(self.rank) and val < self.best_val:
                    self.best_val = val
                    self.save_checkpoint("best")
            if save_every > 0 and self.global_step % save_every == 0:
                self.save_checkpoint(f"step_{self.global_step:08d}")
                self.save_checkpoint("latest")
        if is_main_process(self.rank):
            self.save_checkpoint("latest")
        if self.logger is not None:
            self.logger.finish()

    def train_step(self) -> None:
        self.model.train()
        task_name = self._sample_task()
        task_data = self.data_plan.task_data[task_name]
        batch = move_batch_to_device(self._next_batch(task_name), self.device)
        self._set_active_lrs(task_name)
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model.module.forward_task(task_name, batch) if self.distributed else self.model.forward_task(task_name, batch)
        metrics = task_data.adapter.compute_metrics(outputs, batch)
        loss = metrics["loss"] * self._task_weight(task_name)
        loss.backward()
        grad_clip = float(self.cfg.optimization.get("grad_clip", 0.0) or 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()
        self.global_step += 1
        self._task_steps[task_name] += 1
        self._sample_counts[task_name] += 1

    @torch.no_grad()
    def eval_all(self) -> Dict[str, float]:
        self.model.eval()
        out: Dict[str, float] = {
            "step": float(self.global_step),
            "est_epoch": float(self._est_epoch()),
        }
        task_losses: List[tuple[str, float]] = []
        max_batches = int(self.cfg.evaluation_value("max_val_batches_per_task", 0) or 0)
        val_limit = int(self.cfg.advanced_value("limits", "val_batch_limit", default=0) or 0)
        if max_batches <= 0 and val_limit > 0:
            max_batches = val_limit
        for task_name, loader in self.val_loaders.items():
            task_data = self.data_plan.task_data[task_name]
            sums: Dict[str, float] = {}
            weight = 0
            for batch_i, batch in enumerate(loader, start=1):
                if max_batches > 0 and batch_i > max_batches:
                    break
                batch = move_batch_to_device(batch, self.device)
                outputs = self.model.module.forward_task(task_name, batch) if self.distributed else self.model.forward_task(task_name, batch)
                metrics = task_data.adapter.compute_metrics(outputs, batch)
                bs = self._batch_size(batch)
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().item()) * bs
                weight += bs
            if self.distributed and dist.is_available() and dist.is_initialized():
                metric_names = list(task_data.adapter.metric_names())
                device = self.device if self.device.type == "cuda" else torch.device("cpu")
                totals = torch.tensor(
                    [sums.get(key, 0.0) for key in metric_names] + [float(weight)],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                sums = {key: float(totals[i].item()) for i, key in enumerate(metric_names)}
                weight = int(totals[-1].item())
            if weight <= 0:
                continue
            for key, value in sums.items():
                out[f"{task_name}_{key}"] = value / weight
            if f"{task_name}_loss" in out:
                task_losses.append((task_name, out[f"{task_name}_loss"]))

        denom = 0.0
        weighted = 0.0
        for task_name, loss in task_losses:
            weight = self._task_weight(task_name)
            weighted += weight * loss
            denom += weight
        out["weighted_val_loss"] = float("inf") if denom <= 0 else weighted / denom
        return out

    @staticmethod
    def _batch_size(batch: Dict[str, torch.Tensor]) -> int:
        for key in ("atomic_numbers", "input_atomic_numbers"):
            if key in batch:
                return int(batch[key].shape[0])
        return 1

    def _log_eval(self, metrics: Dict[str, float]) -> None:
        if not is_main_process(self.rank):
            return
        step = int(metrics.get("step", self.global_step))
        weighted = metrics.get("weighted_val_loss", float("nan"))
        line1 = (
            f"[joint:eval] step={step}/{self.max_steps} "
            f"est_epoch={self._est_epoch():.2f}/{self._estimated_total_epochs():.2f} "
            f"weighted={weighted:.6f}"
        )
        parts = []
        aliases = {"electron_density": "density", "coord_denoise": "coord"}
        for task_name in self.val_loaders:
            prefix = aliases.get(task_name, task_name)
            vals = []
            for metric in ("loss", "acc", "mae", "rmse"):
                key = f"{task_name}_{metric}"
                if key in metrics:
                    vals.append(f"{metric}={metrics[key]:.6f}")
            if vals:
                parts.append(f"{prefix} " + " ".join(vals))
        text = line1 + "\n" + " | ".join(parts)
        print(text)
        log_file = self.out_dir / str(self.cfg.advanced_value("diagnostics", "log_file", default="train.log"))
        with log_file.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _write_metrics_csv(self, metrics: Dict[str, float]) -> None:
        if not is_main_process(self.rank):
            return
        columns = ["step", "est_epoch", "weighted_val_loss"]
        for task_name in self.val_loaders:
            for metric in ("loss", "acc", "mae", "rmse"):
                key = f"{task_name}_{metric}"
                if key in metrics:
                    columns.append(key)
        if self._csv_columns is None:
            self._csv_columns = columns
            with self.metrics_file.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                writer.writeheader()
        row = {key: metrics.get(key, "") for key in self._csv_columns}
        row["step"] = int(metrics.get("step", self.global_step))
        with self.metrics_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_columns)
            writer.writerow(row)

    def _print_init(self) -> None:
        print(
            "[joint:init] "
            f"max_steps={self.max_steps} active_tasks={','.join(self.active_tasks)} "
            f"estimated_joint_epoch_steps={self.estimated_joint_epoch_steps or 0:.2f} "
            f"estimated_total_epochs={self._estimated_total_epochs():.2f}"
        )
        if bool(self.cfg.advanced_value("diagnostics", "print_data_plan", default=True)):
            print(self.data_plan.format())

    def save_checkpoint(self, tag: str) -> None:
        if not is_main_process(self.rank):
            return
        include_optimizer = bool(self.cfg.checkpoint_value("save_optimizer", True))
        state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        payload = {
            "model": state,
            "optimizer": self.optimizer.state_dict() if include_optimizer else None,
            "config": dataclasses.asdict(self.cfg),
            "global_step": self.global_step,
            "best_val": self.best_val,
            "world_size": self.world_size,
            "time": time.time(),
        }
        torch.save(payload, self.out_dir / f"{tag}.pt")

    def load_checkpoint(self, path: str, *, strict: bool, ignore_config_mismatch: bool) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if "config" in ckpt:
            validate_restart_config(
                ckpt_args=ckpt["config"],
                cur_args=dataclasses.asdict(self.cfg),
                ignore_config_mismatch=ignore_config_mismatch,
                rank=self.rank,
                logger=self.logger,
            )
        self.raw_model.load_state_dict(ckpt["model"], strict=strict)
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val = float(ckpt.get("best_val", self.best_val))
