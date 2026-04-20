"""GotenNet-style QM9 trainer."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from ....training.batch import move_batch_to_device
from ....training.distributed import is_main_process
from ....training import MetricBag
from ..common.trainer import FinetuneQM9Trainer


class _WarmupPlateauController:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, factor: float, patience: int, min_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0
        self.base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        if self.warmup_steps > 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(factor), patience=int(patience), min_lr=float(min_lr)
        )

    def step_batch(self) -> None:
        self.step_count += 1
        if self.warmup_steps <= 0 or self.step_count > self.warmup_steps:
            return
        scale = self.step_count / max(self.warmup_steps, 1)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def step_epoch(self, metric: float) -> None:
        if self.step_count < self.warmup_steps:
            return
        self.plateau.step(float(metric))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "base_lrs": list(self.base_lrs),
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.step_count = int(state_dict.get("step_count", 0))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        plateau_state = state_dict.get("plateau")
        if plateau_state is not None:
            self.plateau.load_state_dict(plateau_state)


class GotenNetQM9Trainer(FinetuneQM9Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.monitor_key = "mae"
        self.best_monitor = float("inf")
        self.no_improve_epochs = 0
        self.early_stopping_patience = int(self.cfg.early_stopping_patience)
        self.gotennet_scheduler = _WarmupPlateauController(
            optimizer=self.optimizer,
            warmup_steps=self.cfg.warmup_steps,
            factor=self.cfg.lr_factor,
            patience=self.cfg.lr_patience,
            min_lr=self.cfg.lr_min,
        )
        self.scheduler = self.gotennet_scheduler

    def load_checkpoint(self, path: str, strict: bool = True, ignore_config_mismatch: bool = False) -> int:
        next_epoch = super().load_checkpoint(path, strict=strict, ignore_config_mismatch=ignore_config_mismatch)
        self.best_monitor = self.best_val
        return next_epoch

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self.freeze_backbone_epochs > 0:
            self._set_backbone_grad(epoch > self.freeze_backbone_epochs)
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        self.model.train()
        bag = MetricBag(["loss", "mae"])
        iterator = self._train_loader
        if is_main_process(self.rank):
            from tqdm import tqdm
            iterator = tqdm(self._train_loader, desc=f"train e{epoch}", leave=False)

        for batch in iterator:
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad()
            outputs = self.model(
                atomic_numbers=batch["atomic_numbers"],
                coords=batch["coords"],
                atom_padding=batch["atom_padding"],
            )
            losses = self.loss_fn(outputs, batch, target_mean=self.target_mean, target_std=self.target_std)
            losses["loss"].backward()
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.gotennet_scheduler.step_batch()

            bs = batch["atomic_numbers"].shape[0]
            bag.update_dict({k: float(v.item()) for k, v in losses.items()}, weight=bs)
            self.global_step += 1

            if self.logger is not None and self.cfg.log_interval > 0:
                if self.global_step == 1 or self.global_step % self.cfg.log_interval == 0:
                    self.logger.log({k: float(v.item()) for k, v in losses.items()}, step=self.global_step, phase="train_batch")

        return self._reduce_bag(bag)

    def fit(self, start_epoch: int = 1, end_epoch: int | None = None) -> None:
        end = end_epoch or self.epochs
        for epoch in range(start_epoch, end + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.eval_epoch(epoch)
            elapsed = time.time() - t0
            monitor_value = float(val_metrics.get(self.monitor_key, val_metrics.get("loss", float("inf"))))
            self.gotennet_scheduler.step_epoch(monitor_value)

            if self.logger is not None:
                self.logger.log(train_metrics, step=epoch, phase="train")
                if val_metrics:
                    self.logger.log(val_metrics, step=epoch, phase="val")

            should_stop = False
            if is_main_process(self.rank):
                is_best = monitor_value < self.best_monitor
                if is_best:
                    self.best_monitor = monitor_value
                    self.best_val = monitor_value
                    self.no_improve_epochs = 0
                    self.save_checkpoint(epoch, tag="best")
                else:
                    self.no_improve_epochs += 1
                if epoch % self.save_every == 0:
                    self.save_checkpoint(epoch, tag=f"epoch_{epoch:04d}")
                self.save_checkpoint(epoch, tag="latest")
                summary = {
                    "epoch": epoch,
                    "epochs": end,
                    "train_loss": float(train_metrics.get("loss", float("nan"))),
                    "val_loss": float(val_metrics.get("loss", float("nan"))),
                    "val_monitor": float(monitor_value),
                    "time_sec": float(elapsed),
                    "is_best": bool(is_best),
                }
                self.epoch_history.append(
                    {
                        "epoch": epoch,
                        "train": {k: float(v) for k, v in train_metrics.items()},
                        "val": {k: float(v) for k, v in val_metrics.items()},
                        "val_monitor": float(monitor_value),
                        "time_sec": float(elapsed),
                        "is_best": bool(is_best),
                    }
                )
                if self.logger is not None:
                    self.logger.log(summary, step=epoch, phase="epoch")
                if self.early_stopping_patience > 0 and self.no_improve_epochs >= self.early_stopping_patience:
                    should_stop = True
                    if self.logger is not None:
                        self.logger.log(
                            {
                                "epoch": epoch,
                                "reason": "early_stopping",
                                "no_improve_epochs": self.no_improve_epochs,
                                "monitor_key": self.monitor_key,
                                "best_monitor": self.best_monitor,
                            },
                            step=epoch,
                            phase="early_stop",
                        )
            if self.distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
                stop_tensor = torch.tensor([1 if should_stop else 0], device=self.device, dtype=torch.int64)
                torch.distributed.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
            if should_stop:
                break

        if self.logger is not None:
            self.logger.finish()
