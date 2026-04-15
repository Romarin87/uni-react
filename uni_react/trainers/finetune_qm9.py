"""FinetuneQM9Trainer – QM9 property regression fine-tuning."""
import dataclasses
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..configs.finetune_qm9 import FinetuneQM9Config
from ..core.logger import LoggerProtocol
from ..losses.qm9 import QM9RegressionLoss
from ..metrics import MetricBag
from ..training.batch import move_batch_to_device
from ..training.distributed import is_main_process
from ..utils.qm9_dataset import build_qm9_pyg_splits, collate_fn_qm9
from .base import BaseTrainer


def _compute_target_stats(
    dataset, targets: List[str], rank: int
):
    """Compute per-target mean and std on the training set (rank-0 only).

    Uses vectorised bulk loading when the dataset exposes a ``base_dataset``
    with raw ``y`` tensors (QM9PyGDataset), otherwise falls back to iterating
    ``get_targets()`` per sample.
    """
    import numpy as np

    # Fast path: QM9PyGDataset stores indices into a PyG InMemoryDataset.
    # We can bulk-load all y-values without iterating Python-side.
    if hasattr(dataset, "base_dataset") and hasattr(dataset, "indices") and hasattr(dataset, "target_indices"):
        try:
            base = dataset.base_dataset
            indices = dataset.indices  # numpy int64 array
            t_idx = dataset.target_indices  # torch LongTensor
            # Collect y rows for all training indices at once.
            all_y = np.stack(
                [base[int(i)].y.reshape(-1).numpy()[[int(j) for j in t_idx]] for i in indices],
                axis=0,
            ).astype(np.float64)
            return all_y.mean(axis=0), all_y.std(axis=0) + 1e-8
        except Exception:
            pass  # fall through to slow path

    # Slow path: generic iteration.
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
    """Trainer for QM9 property regression fine-tuning.

    Normalises targets using training-set statistics and reports both
    normalised MSE (loss) and unnormalised MAE.

    Args:
        model: The QM9 fine-tuning model (must have ``forward(atomic_numbers,
               coords, atom_padding)`` returning ``{"pred": ...}``).
        cfg: Fully-populated :class:`~uni_react.configs.FinetuneQM9Config`.
        optimizer: Configured optimizer.
        scheduler: LR scheduler (optional).
        logger: Logging backend (optional).
        distributed / rank / world_size / device: DDP settings.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: FinetuneQM9Config,
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
        self.loss_fn = QM9RegressionLoss()
        self.freeze_backbone_epochs = cfg.freeze_backbone_epochs

        # Build splits
        splits = build_qm9_pyg_splits(
            root=cfg.data_root,
            split_mode=cfg.split,
            targets=targets,
            center_coords=not cfg.no_center_coords,
            force_reload=cfg.force_reload,
        )
        train_ds = splits["train"]
        val_ds   = splits["valid"]
        test_ds  = splits["test"]

        # Target normalisation statistics (rank-0 computes, broadcasts)
        if is_main_process(rank):
            mean_np, std_np = _compute_target_stats(train_ds, targets, rank)
        else:
            mean_np = np.zeros(len(targets), dtype=np.float64)
            std_np = np.ones(len(targets), dtype=np.float64)

        stats = torch.tensor(
            np.stack([mean_np, std_np], axis=0), dtype=torch.float64,
            device=device or torch.device("cpu"),
        )
        if distributed:
            dist.broadcast(stats, src=0)
        self.target_mean = stats[0].float()
        self.target_std = stats[1].float()

        # Data loaders
        train_sampler = (
            DistributedSampler(train_ds, shuffle=True, drop_last=True)
            if distributed else None
        )
        val_sampler = (
            DistributedSampler(val_ds, shuffle=False, drop_last=False)
            if distributed else None
        )
        _pin = (device is not None and device.type == "cuda")
        _pw = cfg.num_workers > 0
        self._train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=cfg.num_workers,
            pin_memory=_pin, persistent_workers=_pw,
            drop_last=True, collate_fn=collate_fn_qm9,
        )
        self._train_sampler = train_sampler
        train_eval_sampler = (
            DistributedSampler(train_ds, shuffle=False, drop_last=False)
            if distributed else None
        )
        self._train_eval_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=False,
            sampler=train_eval_sampler, num_workers=cfg.num_workers,
            pin_memory=_pin, persistent_workers=_pw,
            drop_last=False, collate_fn=collate_fn_qm9,
        )
        self._val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            sampler=val_sampler, num_workers=cfg.num_workers,
            pin_memory=_pin, persistent_workers=_pw,
            drop_last=False, collate_fn=collate_fn_qm9,
        )
        self._test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            sampler=None, num_workers=cfg.num_workers,
            pin_memory=_pin, persistent_workers=_pw,
            drop_last=False, collate_fn=collate_fn_qm9,
        )
        if self.scheduler is not None and hasattr(self.scheduler, "set_total_steps"):
            self.scheduler.set_total_steps(max(1, cfg.epochs * len(self._train_loader)))

    # ------------------------------------------------------------------
    # Backbone freeze helper
    # ------------------------------------------------------------------

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        raw = self.raw_model
        if hasattr(raw, "descriptor"):
            for p in raw.descriptor.parameters():
                p.requires_grad_(requires_grad)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self.freeze_backbone_epochs > 0:
            frozen = epoch <= self.freeze_backbone_epochs
            self._set_backbone_grad(not frozen)
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
            losses = self.loss_fn(
                outputs, batch,
                target_mean=self.target_mean,
                target_std=self.target_std,
            )
            losses["loss"].backward()
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            bs = batch["atomic_numbers"].shape[0]
            bag.update_dict({k: float(v.item()) for k, v in losses.items()}, weight=bs)
            self.global_step += 1

            if self.logger is not None and self.cfg.log_interval > 0:
                if self.global_step == 1 or self.global_step % self.cfg.log_interval == 0:
                    self.logger.log(
                        {k: float(v.item()) for k, v in losses.items()},
                        step=self.global_step,
                        phase="train_batch",
                    )

        return self._reduce_bag(bag)

    # ------------------------------------------------------------------
    # Validation / test epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_epoch(self, epoch: int) -> Dict[str, float]:
        return self._evaluate(self._val_loader)

    @torch.no_grad()
    def eval_train(self) -> Dict[str, float]:
        return self._evaluate(self._train_eval_loader)

    @torch.no_grad()
    def eval_val(self) -> Dict[str, float]:
        return self._evaluate(self._val_loader)

    @torch.no_grad()
    def eval_test(self) -> Dict[str, float]:
        return self._evaluate(self._test_loader)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        bag = MetricBag(["loss", "mae"])
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(
                atomic_numbers=batch["atomic_numbers"],
                coords=batch["coords"],
                atom_padding=batch["atom_padding"],
            )
            losses = self.loss_fn(
                outputs, batch,
                target_mean=self.target_mean,
                target_std=self.target_std,
            )
            bs = batch["atomic_numbers"].shape[0]
            bag.update_dict({k: float(v.item()) for k, v in losses.items()}, weight=bs)
        return self._reduce_bag(bag)
