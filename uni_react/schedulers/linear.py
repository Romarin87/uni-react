"""Warmup + linear decay LR scheduler."""
from typing import List

import torch

from ..registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("linear")
class WarmupLinearScheduler:
    """Linear warmup followed by linear decay to ``min_lr_ratio * base_lr``.

    Config example::

        scheduler:
          type: linear
          warmup_steps: 1000
          total_steps: 100000
          min_lr_ratio: 0.0
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
    ) -> None:
        self._optimizer = optimizer
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps
        self._min_lr_ratio = float(min_lr_ratio)
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step_count = 0
        self._last_lrs: List[float] = list(self._base_lrs)

    def step(self) -> None:
        self._step_count += 1
        lrs = [self._compute_lr(base) for base in self._base_lrs]
        for pg, lr in zip(self._optimizer.param_groups, lrs):
            pg["lr"] = lr
        self._last_lrs = lrs

    def get_last_lr(self) -> List[float]:
        return list(self._last_lrs)

    def state_dict(self) -> dict:
        return {
            "step_count": self._step_count,
            "last_lrs": list(self._last_lrs),
            "base_lrs": list(self._base_lrs),
            "warmup_steps": self._warmup_steps,
            "total_steps": self._total_steps,
            "min_lr_ratio": self._min_lr_ratio,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._step_count = int(state_dict.get("step_count", 0))
        self._last_lrs = list(state_dict.get("last_lrs", self._last_lrs))
        self._base_lrs = list(state_dict.get("base_lrs", self._base_lrs))
        self._warmup_steps = int(state_dict.get("warmup_steps", self._warmup_steps))
        self._total_steps = int(state_dict.get("total_steps", self._total_steps))
        self._min_lr_ratio = float(state_dict.get("min_lr_ratio", self._min_lr_ratio))
        for pg, lr in zip(self._optimizer.param_groups, self._last_lrs):
            pg["lr"] = lr

    def set_total_steps(self, total_steps: int) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self._total_steps = int(total_steps)

    def _compute_lr(self, base_lr: float) -> float:
        t = self._step_count
        w = self._warmup_steps
        T = self._total_steps
        min_lr = base_lr * self._min_lr_ratio
        if t < w:
            return min_lr + (base_lr - min_lr) * t / max(w, 1)
        progress = (t - w) / max(T - w, 1)
        progress = min(progress, 1.0)
        return base_lr + (min_lr - base_lr) * progress
