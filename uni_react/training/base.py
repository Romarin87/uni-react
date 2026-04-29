"""Base trainer with shared checkpointing, logging, and distributed helpers."""
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .logger import LoggerProtocol
from .accumulator import MetricBag
from .checkpoint import validate_restart_config
from .distributed import is_main_process


class BaseTrainer:
    """Base class providing checkpoint, logging, and distributed utilities."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        logger: Optional[LoggerProtocol] = None,
        out_dir: str = "runs/exp",
        epochs: int = 20,
        save_every: int = 1,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = None,
        find_unused_parameters: bool = False,
        checkpoint_config: Optional[Mapping[str, Any]] = None,
        save_optimizer: bool = True,
    ) -> None:
        self.raw_model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.out_dir = Path(out_dir)
        self.epochs = epochs
        self.save_every = save_every
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device("cpu")
        self.checkpoint_config = dict(checkpoint_config) if checkpoint_config is not None else None
        self.save_optimizer = bool(save_optimizer)
        self.resume_step_in_epoch: int = 0

        if distributed:
            self.model: torch.nn.Module = DDP(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            self.model = model

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_val: float = float("inf")
        self.global_step: int = 0
        self.epoch_history: list[Dict[str, Any]] = []

        if hasattr(logger, "set_rank"):
            logger.set_rank(rank)  # type: ignore[union-attr]

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        raise NotImplementedError

    def eval_epoch(self, epoch: int) -> Dict[str, float]:
        raise NotImplementedError

    def fit(self, start_epoch: int = 1, end_epoch: Optional[int] = None) -> None:
        end = end_epoch or self.epochs
        for epoch in range(start_epoch, end + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.eval_epoch(epoch)
            elapsed = time.time() - t0

            if self.logger is not None:
                self.logger.log(train_metrics, step=epoch, phase="train")
                if val_metrics:
                    self.logger.log(val_metrics, step=epoch, phase="val")

            if is_main_process(self.rank):
                val_loss = val_metrics.get("loss", float("inf"))
                is_best = val_loss < self.best_val
                if is_best:
                    self.best_val = val_loss
                    self.save_checkpoint(epoch, tag="best")

                if epoch % self.save_every == 0:
                    self.save_checkpoint(epoch, tag=f"epoch_{epoch:04d}")
                self.save_checkpoint(epoch, tag="latest")

                summary = {
                    "epoch": epoch,
                    "epochs": end,
                    "train_loss": float(train_metrics.get("loss", float("nan"))),
                    "val_loss": float(val_metrics.get("loss", float("nan"))),
                    "time_sec": float(elapsed),
                    "is_best": bool(is_best),
                }
                self.epoch_history.append(
                    {
                        "epoch": epoch,
                        "train": {k: float(v) for k, v in train_metrics.items()},
                        "val": {k: float(v) for k, v in val_metrics.items()},
                        "time_sec": float(elapsed),
                        "is_best": bool(is_best),
                    }
                )
                if self.logger is not None:
                    self.logger.log(summary, step=epoch, phase="epoch")
                else:
                    print(
                        f"[epoch {epoch}/{end}] "
                        f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
                        f"val_loss={val_metrics.get('loss', float('nan')):.4f} "
                        f"time={elapsed:.1f}s"
                        + (" [BEST]" if is_best else "")
                    )

        if self.logger is not None:
            self.logger.finish()

    def save_checkpoint(
        self,
        epoch: int,
        tag: str = "latest",
        include_optimizer: Optional[bool] = None,
        step_in_epoch: Optional[int] = None,
    ) -> None:
        if not is_main_process(self.rank):
            return
        if include_optimizer is None:
            include_optimizer = self.save_optimizer
        state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        payload: Dict[str, Any] = {
            "epoch": epoch,
            "model": state,
            "best_val": self.best_val,
            "global_step": self.global_step,
            "world_size": self.world_size,
            "time": time.time(),
        }
        if self.checkpoint_config is not None:
            payload["config"] = dict(self.checkpoint_config)
        if step_in_epoch is not None:
            payload["step_in_epoch"] = int(step_in_epoch)
        if include_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            payload["scheduler"] = self.scheduler.state_dict()
        torch.save(payload, self.out_dir / f"{tag}.pt")

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        ignore_config_mismatch: bool = False,
    ) -> int:
        ckpt = torch.load(path, map_location=self.device)
        ckpt_config = ckpt.get("config", ckpt.get("args"))
        if ckpt_config is not None and self.checkpoint_config is not None:
            validate_restart_config(
                ckpt_args=ckpt_config,
                cur_args=self.checkpoint_config,
                ignore_config_mismatch=ignore_config_mismatch,
                rank=self.rank,
                current_world_size=self.world_size,
                ckpt_world_size=ckpt.get("world_size"),
                logger=self.logger,
            )
        self.raw_model.load_state_dict(ckpt["model"], strict=strict)
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if (
            self.scheduler is not None
            and "scheduler" in ckpt
            and ckpt["scheduler"] is not None
            and hasattr(self.scheduler, "load_state_dict")
        ):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_val = float(ckpt.get("best_val", self.best_val))
        self.global_step = int(ckpt.get("global_step", self.global_step))
        self.resume_step_in_epoch = int(ckpt.get("step_in_epoch", 0))
        if self.resume_step_in_epoch > 0:
            return int(ckpt["epoch"])
        return int(ckpt["epoch"]) + 1

    def barrier(self) -> None:
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def reduce_metrics(self, meters: Dict[str, float], steps: int) -> Dict[str, float]:
        if not meters:
            return {}
        if steps <= 0:
            return dict(meters)
        return {key: float(value) / float(steps) for key, value in meters.items()}

    def _reduce_bag(self, bag: MetricBag) -> Dict[str, float]:
        if not self.distributed:
            return bag.compute()
        device = self.device if self.device.type == "cuda" else torch.device("cpu")
        reduced: Dict[str, float] = {}
        for key in bag.keys():
            tensor = torch.tensor(
                [bag.get_sums()[key], bag.get_weight()],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            denom = float(tensor[1].item())
            reduced[key] = 0.0 if denom == 0.0 else float(tensor[0].item() / denom)
        return reduced
