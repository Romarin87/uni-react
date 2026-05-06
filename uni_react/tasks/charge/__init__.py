"""Atomic charge prediction task for joint training."""
from __future__ import annotations

from typing import Dict, Sequence

import h5py
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
from .head import ChargeHead


class ChargeAdapter(MoleculeTaskAdapter):
    required_keys = (
        "frames/offsets|mol_offsets",
        "atoms/Z|atom_numbers",
        "atoms/R|coords",
        "atoms/q|atoms/q_mulliken|atoms/q_hirshfeld|charges",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = RegressionLoss(
            self.loss_cfg.get("regression_loss", "mse"),
            huber_delta=float(self.loss_cfg.get("huber_delta", 1.0)),
            charbonnier_eps=float(self.loss_cfg.get("charbonnier_eps", 1e-3)),
        )

    def build_dataset(self, files: Sequence[str], split: str) -> DatasetBuildResult:
        self._validate_charge_labels(files)
        dataset = H5SingleMolPretrainDataset(files, **self._dataset_kwargs(split=split, require_reactivity=False))
        return DatasetBuildResult(dataset, self.schema, self.required_keys, files)

    @staticmethod
    def _validate_charge_labels(files: Sequence[str]) -> None:
        missing = []
        for path in files:
            with h5py.File(path, "r") as h5:
                has_stable = "atoms" in h5 and (
                    "q" in h5["atoms"]
                    or "q_mulliken" in h5["atoms"]
                    or "q_hirshfeld" in h5["atoms"]
                )
                has_extxyz = "charges" in h5
                if not (has_stable or has_extxyz):
                    missing.append(path)
        if missing:
            preview = ", ".join(missing[:3])
            suffix = "" if len(missing) <= 3 else f", ... (+{len(missing) - 3} more)"
            raise ValueError(f"Charge task requires charge labels. Missing in: {preview}{suffix}")

    def compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        valid = (~batch["atom_padding"]) & batch.get("charge_valid", ~batch["atom_padding"])
        if not valid.any():
            zero = zero_like(outputs)
            return {"loss": zero, "mae": zero, "rmse": zero}
        return regression_metrics(outputs["charge_pred"], batch["charges"], self.loss_fn, valid)

    def metric_names(self) -> Sequence[str]:
        return ("loss", "mae", "rmse")


def build_head(*, emb_dim: int, atom_vocab_size: int, params: Dict) -> torch.nn.Module:
    del atom_vocab_size, params
    return ChargeHead(emb_dim=emb_dim)


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
    name="charge",
    adapter_cls=ChargeAdapter,
    build_head=build_head,
    forward=forward,
)
