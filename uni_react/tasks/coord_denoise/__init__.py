"""Coordinate denoising task for joint training."""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from ...training.losses import RegressionLoss
from ..common import DatasetBuildResult, MoleculeTaskAdapter, TaskSpec, extract_descriptors, regression_metrics
from ..components.molecule_dataset import H5SingleMolPretrainDataset
from .head import CoordDenoiseHead


class CoordDenoiseAdapter(MoleculeTaskAdapter):
    required_keys = ("frames/offsets|mol_offsets", "atoms/Z|atom_numbers", "atoms/R|coords")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = RegressionLoss(
            self.loss_cfg.get("regression_loss", "mse"),
            huber_delta=float(self.loss_cfg.get("huber_delta", 1.0)),
            charbonnier_eps=float(self.loss_cfg.get("charbonnier_eps", 1e-3)),
        )

    def build_dataset(self, files: Sequence[str], split: str) -> DatasetBuildResult:
        kwargs = self._dataset_kwargs(split=split, require_reactivity=False)
        kwargs["noise_std"] = float(self.params.get("noise_std", 0.1))
        dataset = H5SingleMolPretrainDataset(files, **kwargs)
        return DatasetBuildResult(dataset, self.schema, self.required_keys, files)

    def compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valid = ~batch["atom_padding"]
        return regression_metrics(outputs["coords_denoised"], batch["coords"], self.loss_fn, valid)

    def metric_names(self) -> Sequence[str]:
        return ("loss", "mae", "rmse")


def build_head(*, emb_dim: int, atom_vocab_size: int, params: Dict) -> torch.nn.Module:
    del atom_vocab_size, params
    return CoordDenoiseHead(emb_dim=emb_dim)


def forward(model, head: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    desc = extract_descriptors(
        model,
        batch["atomic_numbers"],
        batch["coords_noisy"],
        batch["atom_padding"],
    )
    out = dict(desc)
    out.update(head(desc))
    return out


TASK_SPEC = TaskSpec(
    name="coord_denoise",
    adapter_cls=CoordDenoiseAdapter,
    build_head=build_head,
    forward=forward,
)
