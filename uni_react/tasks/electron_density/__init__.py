"""Electron density prediction task for joint training."""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from ...training.losses import RegressionLoss
from ..common import DatasetBuildResult, TaskAdapter, TaskSpec, extract_descriptors, regression_metrics
from .dataset import H5ElectronDensityDataset, collate_fn_density
from .head import QueryPointDensityHead


class ElectronDensityAdapter(TaskAdapter):
    schema = "density_grid_h5"
    required_keys = (
        "frames/atom_offsets",
        "frames/n_atoms",
        "frames/density_offsets",
        "frames/n_voxels",
        "frames/grid_shape",
        "frames/grid_origin",
        "frames/grid_vectors",
        "frames/total_charge",
        "frames/spin_multiplicity",
        "atoms/Z",
        "atoms/R",
        "density/target",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = RegressionLoss(
            self.loss_cfg.get("regression_loss", "mse"),
            huber_delta=float(self.loss_cfg.get("huber_delta", 1.0)),
            charbonnier_eps=float(self.loss_cfg.get("charbonnier_eps", 1e-3)),
        )

    @property
    def collate_fn(self):
        return collate_fn_density

    def build_dataset(self, files: Sequence[str], split: str) -> DatasetBuildResult:
        dataset = H5ElectronDensityDataset(
            files,
            num_query_points=int(self.params.get("num_query_points", 2048)),
            center_coords=bool(self.params.get("center_coords", True)) and not bool(self.params.get("no_center_coords", False)),
            deterministic=split != "train",
            seed=self.seed,
            return_ids=False,
        )
        return DatasetBuildResult(dataset, self.schema, self.required_keys, files)

    def compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return regression_metrics(outputs["density_pred"], batch["density_target"], self.loss_fn)

    def metric_names(self) -> Sequence[str]:
        return ("loss", "mae", "rmse")


def build_head(*, emb_dim: int, atom_vocab_size: int, params: Dict) -> torch.nn.Module:
    del atom_vocab_size
    return QueryPointDensityHead(
        emb_dim=emb_dim,
        point_hidden_dim=int(params.get("point_hidden_dim", 128)),
        cond_hidden_dim=int(params.get("cond_hidden_dim", 64)),
        head_hidden_dim=int(params.get("head_hidden_dim", 512)),
        radial_sigma=float(params.get("radial_sigma", 1.5)),
    )


def forward(model, head: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    desc = extract_descriptors(
        model,
        batch["atomic_numbers"],
        batch["coords"],
        batch["atom_padding"],
    )
    pred = head(
        node_feats=desc["node_feats"],
        graph_feats=desc["graph_feats"],
        coords=batch["coords"],
        atom_padding=batch["atom_padding"],
        query_points=batch["query_points"],
        total_charge=batch["total_charge"],
        spin_multiplicity=batch["spin_multiplicity"],
    )
    out = dict(desc)
    out["density_pred"] = pred
    return out


TASK_SPEC = TaskSpec(
    name="electron_density",
    adapter_cls=ElectronDensityAdapter,
    build_head=build_head,
    forward=forward,
)
