"""Shared primitives for independently trainable task implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from ..training.losses import RegressionLoss
from .components.molecule_dataset import collate_fn_pretrain


def zero_like(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    for value in outputs.values():
        if isinstance(value, torch.Tensor):
            return value.sum() * 0.0
    return torch.tensor(0.0)


def regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_fn: RegressionLoss,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    elem_loss = loss_fn(pred, target, reduction="none")
    abs_err = torch.abs(pred - target)
    sq_err = (pred - target).square()
    if mask is not None:
        mask_f = mask.float()
        while mask_f.ndim < elem_loss.ndim:
            mask_f = mask_f.unsqueeze(-1)
        denom = mask_f.expand_as(elem_loss).sum().clamp_min(1.0)
        loss = (elem_loss * mask_f).sum() / denom
        mae = (abs_err * mask_f).sum() / denom
        rmse = torch.sqrt((sq_err * mask_f).sum() / denom)
    else:
        loss = elem_loss.mean()
        mae = abs_err.mean()
        rmse = torch.sqrt(sq_err.mean().clamp_min(0.0))
    return {"loss": loss, "mae": mae, "rmse": rmse}


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset: torch.utils.data.Dataset
    schema: str
    required_keys: Sequence[str]
    files: Sequence[str]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    adapter_cls: type["TaskAdapter"]
    build_head: object
    forward: object


class TaskAdapter:
    name: str
    schema: str
    required_keys: Sequence[str]

    def __init__(self, name: str, task_cfg: Dict, run_cfg: Dict, model_cfg: Dict, advanced_cfg: Dict) -> None:
        self.name = name
        self.task_cfg = task_cfg
        self.run_cfg = run_cfg
        self.model_cfg = model_cfg
        self.advanced_cfg = advanced_cfg
        self.params = dict(task_cfg.get("params", {}) or {})
        self.loss_cfg = dict(task_cfg.get("loss", {}) or {})
        self.batch_size = int(task_cfg.get("batch_size", 1))
        self.seed = int(run_cfg.get("seed", 42))

    def build_dataset(self, files: Sequence[str], split: str) -> DatasetBuildResult:
        raise NotImplementedError

    @property
    def collate_fn(self):
        raise NotImplementedError

    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def metric_names(self) -> Sequence[str]:
        raise NotImplementedError


class MoleculeTaskAdapter(TaskAdapter):
    schema = "molecule_h5"

    def _dataset_kwargs(self, *, split: str, require_reactivity: bool = False) -> Dict:
        params = self.params
        return {
            "mask_ratio": float(params.get("mask_ratio", 0.0)),
            "mask_token_id": int(params.get("mask_token_id", 94)),
            "atom_vocab_size": int(self.model_cfg.get("atom_vocab_size", 128)),
            "min_masked": int(params.get("min_masked", 0)),
            "max_masked": None if int(params.get("max_masked", 0) or 0) <= 0 else int(params["max_masked"]),
            "noise_std": float(params.get("noise_std", 0.0)),
            "center_coords": not bool(params.get("no_center_coords", False)),
            "recenter_noisy": not bool(params.get("no_recenter_noisy", False)),
            "deterministic": split != "train",
            "seed": self.seed,
            "require_reactivity": require_reactivity,
            "reactivity_global_keys": ("vip", "vea"),
            "reactivity_atom_keys": tuple(params.get("targets", ["f_plus", "f_minus", "f_zero"])),
        }

    @property
    def collate_fn(self):
        return collate_fn_pretrain


def extract_descriptors(model, atomic_numbers: torch.Tensor, coords: torch.Tensor, atom_padding: torch.Tensor):
    return model.descriptor(
        input_atomic_numbers=atomic_numbers,
        coords_noisy=coords,
        atom_padding=atom_padding,
    )
