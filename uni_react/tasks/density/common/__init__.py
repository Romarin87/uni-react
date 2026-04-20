"""Common density builders."""

from .dataset import H5DensityPretrainDataset, collate_fn_density
from .model import DensityPretrainNet, QueryPointDensityHead
from .trainer import DensityPretrainTrainer


def build_density_model(cfg, model_spec, task_spec=None):
    del task_spec
    return DensityPretrainNet(
        descriptor=model_spec.build_backbone(cfg),
        emb_dim=cfg.emb_dim,
        point_hidden_dim=cfg.point_hidden_dim,
        cond_hidden_dim=cfg.cond_hidden_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        radial_sigma=cfg.radial_sigma,
    )


__all__ = [
    "H5DensityPretrainDataset",
    "collate_fn_density",
    "DensityPretrainNet",
    "QueryPointDensityHead",
    "DensityPretrainTrainer",
    "build_density_model",
]
