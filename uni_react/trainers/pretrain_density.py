"""DensityPretrainTrainer – electron density pretraining."""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..core.logger import LoggerProtocol
from ..training.distributed import is_main_process
from .base import BaseTrainer


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
    if "sample_ids" in batch:
        out["sample_ids"] = batch["sample_ids"]  # type: ignore[assignment]
    return out


def _reduce_sums(sums: Dict[str, float], count: int, device: torch.device, distributed: bool) -> Dict[str, float]:
    if not distributed:
        if count <= 0:
            return {k: 0.0 for k in sums}
        return {k: v / count for k, v in sums.items()}

    keys = list(sums.keys())
    tensor = torch.tensor([sums[k] for k in keys] + [float(count)], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = int(tensor[-1].item())
    if total_count <= 0:
        return {k: 0.0 for k in keys}
    return {k: float(tensor[i].item() / total_count) for i, k in enumerate(keys)}


class DensityPretrainTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg,
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
        self.cfg = cfg
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._train_sampler = (
            train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
        )
        if self.scheduler is not None and hasattr(self.scheduler, "set_total_steps"):
            self.scheduler.set_total_steps(max(1, cfg.epochs * len(self._train_loader)))

    def _run_one_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        desc: str,
        train: bool,
    ) -> Dict[str, float]:
        self.model.train(train)
        sums = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "rmse": 0.0}
        count = 0

        iterator = loader
        if is_main_process(self.rank):
            iterator = tqdm(loader, desc=f"{desc} e{epoch}" if train else desc, leave=False)

        for step, batch in enumerate(iterator, start=1):
            batch = _move_batch_to_device(batch, self.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(True):
                    out = self.model(
                        atomic_numbers=batch["atomic_numbers"],
                        coords=batch["coords"],
                        atom_padding=batch["atom_padding"],
                        query_points=batch["query_points"],
                        total_charge=batch["total_charge"],
                        spin_multiplicity=batch["spin_multiplicity"],
                    )
                    pred = out["density_pred"]
                    tgt = batch["density_target"]
                    mse = F.mse_loss(pred, tgt)
                    mae = F.l1_loss(pred, tgt)
                    loss = mse
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                with torch.no_grad():
                    out = self.model(
                        atomic_numbers=batch["atomic_numbers"],
                        coords=batch["coords"],
                        atom_padding=batch["atom_padding"],
                        query_points=batch["query_points"],
                        total_charge=batch["total_charge"],
                        spin_multiplicity=batch["spin_multiplicity"],
                    )
                    pred = out["density_pred"]
                    tgt = batch["density_target"]
                    mse = F.mse_loss(pred, tgt)
                    mae = F.l1_loss(pred, tgt)
                    loss = mse

            batch_metrics = {
                "loss": float(loss.item()),
                "mse": float(mse.item()),
                "mae": float(mae.item()),
                "rmse": float(torch.sqrt(torch.clamp(mse, min=0.0)).item()),
            }
            if self.cfg.log_interval > 0 and (self.global_step == 0 or self.global_step % self.cfg.log_interval == 0):
                if is_main_process(self.rank) and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({k: f"{v:.4f}" for k, v in batch_metrics.items() if k != "mse"})
                if self.logger is not None:
                    step_id = self.global_step if train else ((epoch - 1) * max(1, len(loader)) + step)
                    self.logger.log(batch_metrics, step=step_id, phase=f"{desc}_batch")

            bs = int(batch["atomic_numbers"].shape[0])
            for k, v in batch_metrics.items():
                sums[k] += v * bs
            count += bs
            if train:
                self.global_step += 1

        return _reduce_sums(sums=sums, count=count, device=self.device, distributed=self.distributed)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        return self._run_one_epoch(self._train_loader, epoch=epoch, desc="train", train=True)

    @torch.no_grad()
    def eval_epoch(self, epoch: int) -> Dict[str, float]:
        if self._val_loader is None:
            return {}
        return self._run_one_epoch(self._val_loader, epoch=epoch, desc="val", train=False)
