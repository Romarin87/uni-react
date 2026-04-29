"""Common QM9 trainer."""

import dataclasses
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ....configs.qm9 import QM9Config
from ....training.logger import LoggerProtocol
from ....training.batch import move_batch_to_device
from ....training.distributed import is_main_process
from ....training import BaseTrainer, MetricBag
from ..dataset import build_qm9_pyg_splits, collate_fn_qm9
from .loss import QM9RegressionLoss


def _compute_target_stats(dataset, targets: List[str], rank: int):
    if hasattr(dataset, "base_dataset") and hasattr(dataset, "indices") and hasattr(dataset, "target_indices"):
        try:
            base = dataset.base_dataset
            indices = dataset.indices
            t_idx = dataset.target_indices
            all_y = np.stack(
                [base[int(i)].y.reshape(-1).numpy()[[int(j) for j in t_idx]] for i in indices],
                axis=0,
            ).astype(np.float64)
            return all_y.mean(axis=0), all_y.std(axis=0) + 1e-8
        except Exception:
            pass

    all_y = []
    for i in range(len(dataset)):
        if hasattr(dataset, "get_targets"):
            y = dataset.get_targets(i)
        else:
            sample = dataset[i]
            y = sample.get("y", None) if isinstance(sample, dict) else None
            if y is None:
                continue
            y = y.numpy() if hasattr(y, "numpy") else np.array(y)
        all_y.append(np.asarray(y, dtype=np.float64).reshape(-1))
    arr = np.stack(all_y, axis=0)
    return arr.mean(axis=0), arr.std(axis=0) + 1e-8


class FinetuneQM9Trainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: QM9Config,
        optimizer: torch.optim.Optimizer,
        targets: List[str],
        scheduler=None,
        logger: Optional[LoggerProtocol] = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = None,
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
            find_unused_parameters=bool(distributed and cfg.freeze_backbone_epochs > 0),
            checkpoint_config=dataclasses.asdict(cfg),
            save_optimizer=cfg.save_optimizer,
        )
        self.cfg = cfg
        self.targets = targets
        self.loss_fn = QM9RegressionLoss(
            regression_loss_name=cfg.regression_loss,
            huber_delta=cfg.huber_delta,
            charbonnier_eps=cfg.charbonnier_eps,
        )
        self.freeze_backbone_epochs = cfg.freeze_backbone_epochs

        splits = build_qm9_pyg_splits(
            root=cfg.data_root,
            split_mode=cfg.split,
            targets=targets,
            target_index_variant=cfg.qm9_target_variant,
            center_coords=not cfg.no_center_coords,
            force_reload=cfg.force_reload,
        )
        train_ds = splits["train"]
        val_ds = splits["valid"]
        test_ds = splits["test"]

        if is_main_process(rank):
            mean_np, std_np = _compute_target_stats(train_ds, targets, rank)
        else:
            mean_np = np.zeros(len(targets), dtype=np.float64)
            std_np = np.ones(len(targets), dtype=np.float64)

        stats = torch.tensor(np.stack([mean_np, std_np], axis=0), dtype=torch.float64, device=device or torch.device("cpu"))
        if distributed:
            dist.broadcast(stats, src=0)
        self.target_mean = stats[0].float()
        self.target_std = stats[1].float()
        if hasattr(self.raw_model, "set_target_stats"):
            self.raw_model.set_target_stats(mean=self.target_mean.detach().clone(), std=self.target_std.detach().clone())

        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if distributed else None
        _pin = device is not None and device.type == "cuda"
        _pw = cfg.num_workers > 0
        self._train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=_pin,
            persistent_workers=_pw,
            drop_last=True,
            collate_fn=collate_fn_qm9,
        )
        self._train_sampler = train_sampler
        train_eval_sampler = DistributedSampler(train_ds, shuffle=False, drop_last=False) if distributed else None
        self._train_eval_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=train_eval_sampler,
            num_workers=cfg.num_workers,
            pin_memory=_pin,
            persistent_workers=_pw,
            drop_last=False,
            collate_fn=collate_fn_qm9,
        )
        self._val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=_pin,
            persistent_workers=_pw,
            drop_last=False,
            collate_fn=collate_fn_qm9,
        )
        self._test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=cfg.num_workers,
            pin_memory=_pin,
            persistent_workers=_pw,
            drop_last=False,
            collate_fn=collate_fn_qm9,
        )
        if self.scheduler is not None and hasattr(self.scheduler, "set_total_steps"):
            self.scheduler.set_total_steps(max(1, cfg.epochs * len(self._train_loader)))

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        raw = self.raw_model
        if hasattr(raw, "descriptor"):
            for p in raw.descriptor.parameters():
                p.requires_grad_(requires_grad)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self.freeze_backbone_epochs > 0:
            self._set_backbone_grad(epoch > self.freeze_backbone_epochs)
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        self.model.train()
        bag = MetricBag(["loss", "mae"])
        iterator = self._train_loader
        if is_main_process(self.rank):
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
            if self.scheduler is not None:
                self.scheduler.step()

            bs = batch["atomic_numbers"].shape[0]
            bag.update_dict({k: float(v.item()) for k, v in losses.items()}, weight=bs)
            self.global_step += 1

            if self.logger is not None and self.cfg.log_interval > 0:
                if self.global_step == 1 or self.global_step % self.cfg.log_interval == 0:
                    self.logger.log({k: float(v.item()) for k, v in losses.items()}, step=self.global_step, phase="train_batch")

        return self._reduce_bag(bag)

    def _eval_loader_metrics(self, loader) -> Dict[str, float]:
        self.model.eval()
        bag = MetricBag(["loss", "mae"])
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, self.device)
                outputs = self.model(
                    atomic_numbers=batch["atomic_numbers"],
                    coords=batch["coords"],
                    atom_padding=batch["atom_padding"],
                )
                losses = self.loss_fn(
                    outputs,
                    batch,
                    target_mean=self.target_mean,
                    target_std=self.target_std,
                )
                bs = batch["atomic_numbers"].shape[0]
                bag.update_dict({k: float(v.item()) for k, v in losses.items()}, weight=bs)
        return self._reduce_bag(bag)

    def eval_epoch(self, epoch: int) -> Dict[str, float]:
        del epoch
        return self.eval_val()

    def eval_train(self) -> Dict[str, float]:
        return self._eval_loader_metrics(self._train_eval_loader)

    def eval_val(self) -> Dict[str, float]:
        return self._eval_loader_metrics(self._val_loader)

    def eval_test(self) -> Dict[str, float]:
        return self._eval_loader_metrics(self._test_loader)
