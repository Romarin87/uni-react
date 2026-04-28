"""Shared trainer for geometric and CDFT pretraining."""

import dataclasses
import json
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ....configs.pretrain import PretrainConfig
from ....logger import LoggerProtocol
from ....training.batch import move_batch_to_device
from ....training.distributed import is_main_process
from ....training import BaseTrainer, MetricBag
from .dataset import collate_fn_pretrain
from .dataset_helpers import build_pretrain_dataset
from .samplers import EpochRandomSampler, OffsetSampler


class PretrainTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Any,
        optimizer: torch.optim.Optimizer,
        cfg: PretrainConfig,
        scheduler=None,
        logger: Optional[LoggerProtocol] = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = None,
        find_unused_parameters: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            out_dir=cfg.out_dir,
            epochs=cfg.epochs,
            save_every=cfg.save_every,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            device=device,
            find_unused_parameters=find_unused_parameters,
            checkpoint_config=dataclasses.asdict(cfg),
            save_optimizer=cfg.save_optimizer,
        )
        self.loss_fn = loss_fn
        self.cfg = cfg
        self._metric_keys = list(loss_fn.metric_keys())
        self._active_task = "electronic_structure" if cfg.train_mode in {"cdft", "electronic_structure"} else cfg.train_mode

        max_masked = None if cfg.max_masked <= 0 else cfg.max_masked
        dataset_kwargs = dict(
            mask_ratio=cfg.mask_ratio,
            mask_token_id=cfg.mask_token_id,
            atom_vocab_size=cfg.atom_vocab_size,
            min_masked=cfg.min_masked,
            max_masked=max_masked,
            noise_std=cfg.noise_std,
            center_coords=not cfg.no_center_coords,
            recenter_noisy=not cfg.no_recenter_noisy,
            deterministic=False,
            seed=cfg.seed,
            require_reactivity=(self._active_task == "electronic_structure"),
            reactivity_global_keys=cfg.vip_vea_keys,
            reactivity_atom_keys=cfg.fukui_keys,
        )
        train_dataset = build_pretrain_dataset(
            cfg.train_h5,
            file_limit=cfg.smoke_h5_file_limit,
            **dataset_kwargs,
        )

        if distributed:
            base_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        else:
            base_sampler = EpochRandomSampler(train_dataset, seed=cfg.seed)
        self._train_sampler = OffsetSampler(base_sampler)

        self._train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=self._train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=(device is not None and device.type == "cuda"),
            persistent_workers=(cfg.num_workers > 0),
            drop_last=True,
            collate_fn=collate_fn_pretrain,
            prefetch_factor=4 if cfg.num_workers > 0 else None,
        )

        self._val_loader: Optional[DataLoader] = None
        if cfg.val_h5:
            val_kwargs = dict(dataset_kwargs)
            val_kwargs["deterministic"] = True
            val_dataset = build_pretrain_dataset(
                cfg.val_h5,
                file_limit=cfg.smoke_h5_file_limit,
                **val_kwargs,
            )
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if distributed else None
            self._val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=cfg.num_workers,
                pin_memory=(device is not None and device.type == "cuda"),
                persistent_workers=(cfg.num_workers > 0),
                drop_last=False,
                collate_fn=collate_fn_pretrain,
                prefetch_factor=4 if cfg.num_workers > 0 else None,
            )

        if self.scheduler is not None and hasattr(self.scheduler, "set_total_steps"):
            steps_per_epoch = len(self._train_loader)
            if cfg.smoke_train_batch_limit > 0:
                steps_per_epoch = min(steps_per_epoch, cfg.smoke_train_batch_limit)
            self.scheduler.set_total_steps(max(1, cfg.epochs * steps_per_epoch))

    @staticmethod
    def _finite_summary(tensor: torch.Tensor) -> Dict[str, Any]:
        data = tensor.detach()
        summary: Dict[str, Any] = {
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "device": str(data.device),
        }
        if data.numel() == 0:
            summary.update({"numel": 0, "finite": True})
            return summary

        if torch.is_floating_point(data) or torch.is_complex(data):
            finite = torch.isfinite(data)
            summary.update(
                {
                    "numel": int(data.numel()),
                    "finite_count": int(finite.sum().item()),
                    "nan_count": int(torch.isnan(data).sum().item()),
                    "posinf_count": int(torch.isposinf(data).sum().item()),
                    "neginf_count": int(torch.isneginf(data).sum().item()),
                    "finite": bool(finite.all().item()),
                }
            )
            finite_data = data[finite]
            if finite_data.numel() > 0:
                finite_float = finite_data.float()
                summary.update(
                    {
                        "min": float(finite_float.min().item()),
                        "max": float(finite_float.max().item()),
                        "mean": float(finite_float.mean().item()),
                    }
                )
            return summary

        summary.update(
            {
                "numel": int(data.numel()),
                "finite": True,
                "min": int(data.min().item()),
                "max": int(data.max().item()),
            }
        )
        return summary

    @staticmethod
    def _batch_summary(batch: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        atom_padding = batch.get("atom_padding")
        if isinstance(atom_padding, torch.Tensor):
            atoms_per_mol = (~atom_padding).sum(dim=1)
            summary["atoms_per_mol"] = {
                "min": int(atoms_per_mol.min().item()),
                "max": int(atoms_per_mol.max().item()),
                "mean": float(atoms_per_mol.float().mean().item()),
            }

        mask_positions = batch.get("mask_positions", batch.get("mask"))
        if isinstance(mask_positions, torch.Tensor) and mask_positions.ndim >= 2:
            masked_per_mol = mask_positions.sum(dim=1)
            summary["masked_per_mol"] = {
                "min": int(masked_per_mol.min().item()),
                "max": int(masked_per_mol.max().item()),
                "mean": float(masked_per_mol.float().mean().item()),
            }

        for name in ("atomic_numbers", "input_atomic_numbers", "coords", "coords_noisy", "charges", "charge_valid"):
            value = batch.get(name)
            if isinstance(value, torch.Tensor):
                summary[name] = PretrainTrainer._finite_summary(value)

        return summary

    def _log_nonfinite(
        self,
        *,
        where: str,
        epoch: int,
        step_in_epoch: int,
        batch: Dict[str, Any],
        tensors: Dict[str, Any],
    ) -> None:
        payload: Dict[str, Any] = {
            "where": where,
            "epoch": epoch,
            "global_step": self.global_step + 1,
            "step_in_epoch": step_in_epoch,
            "rank": self.rank,
            "world_size": self.world_size,
            "batch": self._batch_summary(batch),
            "tensors": {},
        }
        for name, value in tensors.items():
            if isinstance(value, torch.Tensor):
                payload["tensors"][name] = self._finite_summary(value)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        nonfinite_path = self.out_dir / f"nonfinite_rank{self.rank}.jsonl"
        with nonfinite_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        print("[nonfinite] " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)
        if self.logger is not None:
            self.logger.log(payload, step=self.global_step + 1, phase="nonfinite")

    def _check_finite(
        self,
        *,
        where: str,
        epoch: int,
        step_in_epoch: int,
        batch: Dict[str, Any],
        tensors: Dict[str, Any],
    ) -> None:
        bad = {
            name: value
            for name, value in tensors.items()
            if isinstance(value, torch.Tensor)
            and (torch.is_floating_point(value) or torch.is_complex(value))
            and value.numel() > 0
            and not torch.isfinite(value.detach()).all().item()
        }
        if not bad:
            return
        self._log_nonfinite(
            where=where,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
            batch=batch,
            tensors=bad,
        )
        names = ", ".join(sorted(bad))
        raise RuntimeError(
            f"Non-finite tensor(s) detected at {where}: {names}; "
            f"epoch={epoch} global_step={self.global_step + 1} "
            f"step_in_epoch={step_in_epoch} rank={self.rank}"
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self._train_sampler.set_epoch(epoch)
        self._train_sampler.set_skip(self.resume_step_in_epoch * self.cfg.batch_size if self.resume_step_in_epoch > 0 else 0)
        bag = MetricBag(self._metric_keys)
        iterator = self._train_loader
        if is_main_process(self.rank):
            iterator = tqdm(self._train_loader, desc=f"train e{epoch}", leave=False)

        steps_seen = 0
        for step_in_epoch, batch in enumerate(iterator, start=self.resume_step_in_epoch):
            if self.cfg.smoke_train_batch_limit > 0 and steps_seen >= self.cfg.smoke_train_batch_limit:
                break
            steps_seen += 1

            batch = move_batch_to_device(batch, self.device)
            self._check_finite(
                where="batch_inputs",
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                batch=batch,
                tensors={
                    "coords": batch.get("coords"),
                    "coords_noisy": batch.get("coords_noisy"),
                    "charges": batch.get("charges"),
                },
            )
            self.optimizer.zero_grad()
            outputs = self.model(
                input_atomic_numbers=batch["input_atomic_numbers"],
                coords_noisy=batch["coords_noisy"],
                atom_padding=batch["atom_padding"],
                active_pipeline_tasks=(self._active_task,),
            )
            self._check_finite(
                where="forward_outputs",
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                batch=batch,
                tensors=outputs,
            )
            losses = self.loss_fn(outputs, batch)
            self._check_finite(
                where="loss",
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                batch=batch,
                tensors=losses,
            )
            losses["loss"].backward()
            if self.cfg.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self._check_finite(
                    where="grad_norm",
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    batch=batch,
                    tensors={"grad_norm": grad_norm if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm)},
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            bs = batch["input_atomic_numbers"].shape[0]
            bag.update_dict({k: float(v.item()) for k, v in losses.items() if k in self._metric_keys}, weight=bs)
            self.global_step += 1

            if self.logger is not None and self.cfg.log_interval > 0:
                if self.global_step == 1 or self.global_step % self.cfg.log_interval == 0:
                    self.logger.log(
                        {k: float(v.item()) for k, v in losses.items() if k in self._metric_keys},
                        step=self.global_step,
                        phase="train_batch",
                    )

            if self.cfg.save_every_steps > 0 and self.global_step % self.cfg.save_every_steps == 0 and is_main_process(self.rank):
                self.save_checkpoint(epoch, tag=f"step_{self.global_step:08d}", step_in_epoch=step_in_epoch + 1)

        self._train_sampler.set_skip(0)
        self.resume_step_in_epoch = 0
        return self._reduce_bag(bag)

    @torch.no_grad()
    def eval_epoch(self, epoch: int) -> Dict[str, float]:
        if self._val_loader is None:
            return {}
        self.model.eval()
        bag = MetricBag(self._metric_keys)
        iterator = self._val_loader
        if is_main_process(self.rank):
            iterator = tqdm(self._val_loader, desc="val", leave=False)

        for step_idx, batch in enumerate(iterator):
            if self.cfg.smoke_val_batch_limit > 0 and step_idx >= self.cfg.smoke_val_batch_limit:
                break

            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(
                input_atomic_numbers=batch["input_atomic_numbers"],
                coords_noisy=batch["coords_noisy"],
                atom_padding=batch["atom_padding"],
                active_pipeline_tasks=(self._active_task,),
            )
            losses = self.loss_fn(outputs, batch)
            bs = batch["input_atomic_numbers"].shape[0]
            bag.update_dict({k: float(v.item()) for k, v in losses.items() if k in self._metric_keys}, weight=bs)

        return self._reduce_bag(bag)
