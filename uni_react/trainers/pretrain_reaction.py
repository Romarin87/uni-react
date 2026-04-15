"""ReactionPretrainTrainer – reaction triplet consistency pretraining."""
import dataclasses
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ..configs.pretrain_reaction import ReactionPretrainConfig
from ..core.logger import LoggerProtocol
from ..metrics import MetricBag
from ..metrics.classification import binary_accuracy
from ..training.batch import move_batch_to_device
from ..training.distributed import is_main_process
from ..utils.reaction_dataset import (
    ReactionTripletH5Dataset,
    collate_reaction_triplet,
    split_dataset,
)
from .base import BaseTrainer


class ReactionPretrainTrainer(BaseTrainer):
    """Trainer for reaction triplet consistency + completion pretraining.

    Uses an EMA teacher-student (:class:`~uni_react.encoders.ReactionPretrainNet`)
    trained on reaction triplet HDF5 data.

    Args:
        model: An unwrapped :class:`~uni_react.encoders.ReactionPretrainNet` instance.
        cfg: A fully-populated :class:`~uni_react.configs.ReactionPretrainConfig`.
        optimizer: Pre-built optimizer.
        scheduler: Optional LR scheduler.
        logger: Optional logging backend.
        distributed / rank / world_size / device: DDP settings.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: ReactionPretrainConfig,
        optimizer: torch.optim.Optimizer,
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
            checkpoint_config=dataclasses.asdict(cfg),
            save_optimizer=cfg.save_optimizer,
        )
        self.cfg = cfg

        # Build datasets
        train_base = ReactionTripletH5Dataset(
            h5_path=cfg.train_h5,
            neg_ratio=cfg.neg_ratio,
            hard_negative=cfg.hard_negative,
            seed=cfg.seed,
        )
        if cfg.val_h5:
            train_ds = train_base
            val_ds = ReactionTripletH5Dataset(
                h5_path=cfg.val_h5,
                neg_ratio=cfg.neg_ratio,
                hard_negative=cfg.hard_negative,
                seed=cfg.seed + 1234,
            )
        else:
            train_ds, val_ds = split_dataset(train_base, val_ratio=cfg.val_ratio, seed=cfg.seed)

        train_sampler = DistributedSampler(train_ds, shuffle=True,  drop_last=False) if distributed else None
        val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False) if (distributed and val_ds is not None) else None

        _pin = device is not None and device.type == "cuda"
        _pw  = cfg.num_workers > 0
        self._train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=_pin,
            persistent_workers=_pw,
            drop_last=False,
            collate_fn=collate_reaction_triplet,
        )
        self._val_loader: Optional[DataLoader] = None
        if val_ds is not None:
            self._val_loader = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=cfg.num_workers,
                pin_memory=_pin,
                persistent_workers=_pw,
                drop_last=False,
                collate_fn=collate_reaction_triplet,
            )
        self._train_sampler = train_sampler
        if self.scheduler is not None and hasattr(self.scheduler, "set_total_steps"):
            self.scheduler.set_total_steps(max(1, cfg.epochs * len(self._train_loader)))

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        module = self.model.module if self.distributed else self.model
        module.teacher_descriptor.eval()

        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

        _KEYS = ["loss", "consistency_loss", "completion_loss", "consistency_acc"]
        bag = MetricBag(_KEYS)

        iterator = self._train_loader
        if is_main_process(self.rank):
            iterator = tqdm(self._train_loader, desc=f"train e{epoch}", leave=False)

        for batch in iterator:
            batch   = move_batch_to_device(batch, self.device)
            outputs = self.model(batch)

            cons_loss  = F.binary_cross_entropy_with_logits(outputs["cons_logits"], batch["cons_label"])
            comp_loss  = self.cfg.completion_weight  * outputs["comp_loss"]
            total_loss = self.cfg.consistency_weight * cons_loss + comp_loss

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            module.update_teacher()
            if self.scheduler is not None:
                self.scheduler.step()

            with torch.no_grad():
                acc = binary_accuracy(outputs["cons_logits"], batch["cons_label"])
                bs  = int(batch["cons_label"].shape[0])
                batch_metrics = {
                    "loss":             float(total_loss.item()),
                    "consistency_loss": float(cons_loss.item()),
                    "completion_loss":  float(comp_loss.item()),
                    "consistency_acc":  float(acc.item()),
                }
                bag.update_dict(batch_metrics, weight=bs)
            self.global_step += 1

            if self.logger is not None and self.cfg.log_interval > 0:
                if self.global_step == 1 or self.global_step % self.cfg.log_interval == 0:
                    self.logger.log(batch_metrics, step=self.global_step, phase="train_batch")

        return self._reduce_bag(bag)

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_epoch(self, epoch: int) -> Dict[str, float]:
        if self._val_loader is None:
            return {}
        _KEYS = ["loss", "consistency_loss", "completion_loss", "consistency_acc"]
        self.model.eval()
        bag = MetricBag(_KEYS)

        for batch in self._val_loader:
            batch   = move_batch_to_device(batch, self.device)
            outputs = self.model(batch)
            cons_loss  = F.binary_cross_entropy_with_logits(outputs["cons_logits"], batch["cons_label"])
            comp_loss  = self.cfg.completion_weight  * outputs["comp_loss"]
            total_loss = self.cfg.consistency_weight * cons_loss + comp_loss
            acc = binary_accuracy(outputs["cons_logits"], batch["cons_label"])
            bs  = int(batch["cons_label"].shape[0])
            bag.update_dict({
                "loss":             float(total_loss.item()),
                "consistency_loss": float(cons_loss.item()),
                "completion_loss":  float(comp_loss.item()),
                "consistency_acc":  float(acc.item()),
            }, weight=bs)

        return self._reduce_bag(bag)
