"""VIP scalar task for joint training."""
from __future__ import annotations

from typing import Dict

import torch

from ..common import TaskSpec
from ..components.electronic_scalar import ElectronicScalarAdapter, build_scalar_head, forward_scalar


class VipAdapter(ElectronicScalarAdapter):
    target_key = "vip"


def forward(model, head: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return forward_scalar("vip", model, head, batch)


TASK_SPEC = TaskSpec(
    name="vip",
    adapter_cls=VipAdapter,
    build_head=build_scalar_head,
    forward=forward,
)
