"""Constant (no-op) LR scheduler."""
from typing import List

import torch

from ..registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("none")
class ConstantScheduler:
    """No-op scheduler – keeps the LR constant throughout training.

    Config example::

        scheduler:
          type: none
    """

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer
        self._last_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self) -> None:
        pass  # nothing to do

    def get_last_lr(self) -> List[float]:
        return list(self._last_lrs)

    def state_dict(self) -> dict:
        return {"last_lrs": list(self._last_lrs)}

    def load_state_dict(self, state_dict: dict) -> None:
        self._last_lrs = list(state_dict.get("last_lrs", self._last_lrs))
        for pg, lr in zip(self._optimizer.param_groups, self._last_lrs):
            pg["lr"] = lr
