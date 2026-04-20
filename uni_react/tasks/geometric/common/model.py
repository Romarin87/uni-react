"""Shared geometric/CDFT pretraining assembled model."""
from typing import Dict, Iterable, Optional

import torch

from ....tasks.cdft.common.electronic_pipeline import ElectronicStructureTask
from .geometric_pipeline import GeometricStructureTask


class PretrainTaskModel(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        atom_vocab_size: int = 128,
        enable_electronic_structure_task: bool = False,
        electronic_structure_vip_vea_dim: int = 2,
        electronic_structure_fukui_dim: int = 3,
        descriptor: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        if descriptor is None:
            raise ValueError("PretrainTaskModel requires an explicit backbone descriptor.")
        self.emb_dim = int(emb_dim)
        self.descriptor = descriptor
        self.tasks = torch.nn.ModuleDict({
            "geometric_structure": GeometricStructureTask(
                emb_dim=emb_dim,
                atom_vocab_size=atom_vocab_size,
            ),
        })
        if enable_electronic_structure_task:
            self.tasks["electronic_structure"] = ElectronicStructureTask(
                emb_dim=emb_dim,
                vip_vea_dim=electronic_structure_vip_vea_dim,
                fukui_dim=electronic_structure_fukui_dim,
            )
        self.default_pipeline_tasks = ("geometric_structure",)
        self.task_atom_heads = torch.nn.ModuleDict()
        self.task_graph_heads = torch.nn.ModuleDict()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def atom_head(self):
        return self.tasks["geometric_structure"].atom_mask.head

    @property
    def coord_head(self):
        return self.tasks["geometric_structure"].coord_denoise.head

    @property
    def charge_head(self):
        return self.tasks["geometric_structure"].charge.head

    def extract_descriptors(
        self,
        input_atomic_numbers: torch.Tensor,
        coords_noisy: torch.Tensor,
        atom_padding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.descriptor(
            input_atomic_numbers=input_atomic_numbers,
            coords_noisy=coords_noisy,
            atom_padding=atom_padding,
        )

    def forward(
        self,
        input_atomic_numbers: torch.Tensor,
        coords_noisy: torch.Tensor,
        atom_padding: Optional[torch.Tensor] = None,
        active_pipeline_tasks: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        descriptors = self.extract_descriptors(
            input_atomic_numbers=input_atomic_numbers,
            coords_noisy=coords_noisy,
            atom_padding=atom_padding,
        )
        pipeline_tasks = tuple(active_pipeline_tasks) if active_pipeline_tasks is not None else self.default_pipeline_tasks
        out = dict(descriptors)
        for task_name in pipeline_tasks:
            if task_name not in self.tasks:
                raise KeyError(f"Unknown pipeline task: {task_name!r}")
            out.update(self.tasks[task_name](descriptors))
        for name, head in self.task_atom_heads.items():
            out[name] = head(descriptors["node_feats"])
        for name, head in self.task_graph_heads.items():
            out[name] = head(descriptors["graph_feats"])
        return out

    def compute_geometric_structure_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        atom_weight: float = 1.0,
        coord_weight: float = 1.0,
        charge_weight: float = 1.0,
        active_subtasks: Optional[Iterable[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.tasks["geometric_structure"].compute_loss_dict(
            outputs=outputs,
            batch=batch,
            atom_weight=atom_weight,
            coord_weight=coord_weight,
            charge_weight=charge_weight,
            active_subtasks=active_subtasks,
        )

    def compute_electronic_structure_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        vip_vea_weight: float = 1.0,
        fukui_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if "electronic_structure" not in self.tasks:
            raise RuntimeError("Enable electronic_structure task first.")
        return self.tasks["electronic_structure"].compute_loss_dict(
            outputs=outputs,
            batch=batch,
            vip_vea_weight=vip_vea_weight,
            fukui_weight=fukui_weight,
        )
