"""Shared pieces for scalar electronic-structure joint tasks."""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from ...training.losses import RegressionLoss
from ..common import DatasetBuildResult, MoleculeTaskAdapter, extract_descriptors, regression_metrics
from .molecule_dataset import H5SingleMolPretrainDataset


class ScalarGraphHead(torch.nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, 1),
        )

    def forward(self, graph_feats: torch.Tensor) -> torch.Tensor:
        return self.head(graph_feats).squeeze(-1)


class ElectronicScalarAdapter(MoleculeTaskAdapter):
    target_key: str
    required_keys = (
        "frames/offsets|mol_offsets",
        "atoms/Z|atom_numbers",
        "atoms/R|coords",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not getattr(self, "target_key", None):
            raise ValueError(f"{type(self).__name__} must define target_key")
        self.required_keys = tuple(self.required_keys) + (f"frames/{self.target_key}|{self.target_key}",)
        self.loss_fn = RegressionLoss(
            self.loss_cfg.get("regression_loss", "mse"),
            huber_delta=float(self.loss_cfg.get("huber_delta", 1.0)),
            charbonnier_eps=float(self.loss_cfg.get("charbonnier_eps", 1e-3)),
        )

    def build_dataset(self, files: Sequence[str], split: str) -> DatasetBuildResult:
        kwargs = self._dataset_kwargs(split=split, require_reactivity=True)
        kwargs["reactivity_global_keys"] = (self.target_key,)
        kwargs["reactivity_atom_keys"] = ()
        dataset = H5SingleMolPretrainDataset(files, **kwargs)
        return DatasetBuildResult(dataset, self.schema, self.required_keys, files)

    def compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = outputs[f"{self.name}_pred"]
        target = batch["reactivity_global"][:, 0]
        return regression_metrics(pred, target, self.loss_fn)

    def metric_names(self) -> Sequence[str]:
        return ("loss", "mae", "rmse")


def build_scalar_head(*, emb_dim: int, atom_vocab_size: int, params: Dict) -> torch.nn.Module:
    del atom_vocab_size, params
    return ScalarGraphHead(emb_dim=emb_dim)


def forward_scalar(task_name: str, model, head: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    desc = extract_descriptors(
        model,
        batch["atomic_numbers"],
        batch["coords"],
        batch["atom_padding"],
    )
    out = dict(desc)
    out[f"{task_name}_pred"] = head(desc["graph_feats"])
    return out
