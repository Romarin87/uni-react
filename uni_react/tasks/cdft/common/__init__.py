"""Common CDFT pretraining builders."""

from .electronic_pipeline import ElectronicStructureTask
from .fukui import FukuiHead
from .loss import ElectronicStructureLoss
from .vip_vea import VipVeaHead
from ...geometric.common import PretrainTaskModel


def build_cdft_model(cfg, model_spec, task_spec=None):
    del task_spec
    return PretrainTaskModel(
        emb_dim=cfg.emb_dim,
        atom_vocab_size=cfg.atom_vocab_size,
        descriptor=model_spec.build_backbone(cfg),
        enable_electronic_structure_task=True,
    )


__all__ = [
    "ElectronicStructureTask",
    "ElectronicStructureLoss",
    "FukuiHead",
    "PretrainTaskModel",
    "VipVeaHead",
    "build_cdft_model",
]
