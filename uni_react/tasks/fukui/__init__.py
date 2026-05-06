"""Fukui atom-level reactivity task for joint training."""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from ...training.losses import RegressionLoss
from ..common import (
    DatasetBuildResult,
    MoleculeTaskAdapter,
    TaskSpec,
    extract_descriptors,
    regression_metrics,
    zero_like,
)
from ..components.molecule_dataset import H5SingleMolPretrainDataset
from .head import FukuiHead


class FukuiAdapter(MoleculeTaskAdapter):
    required_keys = (
        "frames/offsets|mol_offsets",
        "atoms/Z|atom_numbers",
        "atoms/R|coords",
        "atoms/f_plus|f_plus",
        "atoms/f_minus|f_minus",
        "atoms/f_zero|f_zero",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = RegressionLoss(
            self.loss_cfg.get("regression_loss", "mse"),
            huber_delta=float(self.loss_cfg.get("huber_delta", 1.0)),
            charbonnier_eps=float(self.loss_cfg.get("charbonnier_eps", 1e-3)),
        )

    def build_dataset(self, files: Sequence[str], split: str) -> DatasetBuildResult:
        kwargs = self._dataset_kwargs(split=split, require_reactivity=True)
        kwargs["reactivity_global_keys"] = ()
        kwargs["reactivity_atom_keys"] = tuple(self.params.get("targets", ["f_plus", "f_minus", "f_zero"]))
        dataset = H5SingleMolPretrainDataset(files, **kwargs)
        return DatasetBuildResult(dataset, self.schema, self.required_keys, files)

    def compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valid = (~batch["atom_padding"]) & batch.get("reactivity_atom_valid", ~batch["atom_padding"])
        if not valid.any():
            zero = zero_like(outputs)
            return {"loss": zero, "mae": zero, "rmse": zero}
        return regression_metrics(outputs["fukui_pred"], batch["reactivity_atom"], self.loss_fn, valid)

    def metric_names(self) -> Sequence[str]:
        return ("loss", "mae", "rmse")


def build_head(*, emb_dim: int, atom_vocab_size: int, params: Dict) -> torch.nn.Module:
    del atom_vocab_size
    targets = params.get("targets", ["f_plus", "f_minus", "f_zero"])
    return FukuiHead(emb_dim=emb_dim, out_dim=len(targets))


def forward(model, head: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    desc = extract_descriptors(
        model,
        batch["atomic_numbers"],
        batch["coords"],
        batch["atom_padding"],
    )
    out = dict(desc)
    out.update(head(desc))
    return out


TASK_SPEC = TaskSpec(
    name="fukui",
    adapter_cls=FukuiAdapter,
    build_head=build_head,
    forward=forward,
)
