"""Joint task model with one shared descriptor and independent task heads."""
from __future__ import annotations

from typing import Dict, Iterable

import torch

from ..registry import get_task_spec


class JointTaskModel(torch.nn.Module):
    def __init__(
        self,
        descriptor: torch.nn.Module,
        emb_dim: int,
        atom_vocab_size: int,
        task_configs: Dict[str, Dict],
    ) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.emb_dim = int(emb_dim)
        self.tasks = torch.nn.ModuleDict()
        self.task_specs = {}
        for task_name, task_cfg in task_configs.items():
            if not bool(task_cfg.get("enabled", True)):
                continue
            spec = get_task_spec(task_name)
            params = dict(task_cfg.get("params", {}) or {})
            self.task_specs[task_name] = spec
            self.tasks[task_name] = spec.build_head(
                emb_dim=emb_dim,
                atom_vocab_size=atom_vocab_size,
                params=params,
            )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extract_descriptors(
        self,
        atomic_numbers: torch.Tensor,
        coords: torch.Tensor,
        atom_padding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.descriptor(
            input_atomic_numbers=atomic_numbers,
            coords_noisy=coords,
            atom_padding=atom_padding,
        )

    def task_parameters(self, task_name: str) -> Iterable[torch.nn.Parameter]:
        return self.tasks[task_name].parameters()

    def forward_task(self, task_name: str, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if task_name not in self.tasks:
            raise KeyError(f"Task {task_name!r} is not enabled in the model")
        spec = self.task_specs[task_name]
        return spec.forward(self, self.tasks[task_name], batch)
