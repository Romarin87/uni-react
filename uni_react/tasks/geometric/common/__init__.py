"""Common geometric pretraining builders."""

from .atom_mask import AtomMaskHead
from .charge import ChargeHead
from .coord_denoise import CoordDenoiseHead
from .dataset import H5SingleMolPretrainDataset, collate_fn_pretrain
from .dataset_helpers import build_pretrain_dataset, expand_h5_files, split_h5_files
from .geometric_pipeline import GeometricStructureTask
from .loss import GeometricStructureLoss
from .model import PretrainTaskModel
from .samplers import EpochRandomSampler, OffsetSampler
from .trainer import PretrainTrainer
from .transforms import AddGaussianNoise, CenterCoords, Compose, MaskAtoms


def build_geometric_model(cfg, model_spec, task_spec=None):
    del task_spec
    return PretrainTaskModel(
        emb_dim=cfg.emb_dim,
        atom_vocab_size=cfg.atom_vocab_size,
        descriptor=model_spec.build_backbone(cfg),
        enable_electronic_structure_task=False,
    )


__all__ = [
    "AtomMaskHead",
    "ChargeHead",
    "CoordDenoiseHead",
    "H5SingleMolPretrainDataset",
    "collate_fn_pretrain",
    "build_pretrain_dataset",
    "expand_h5_files",
    "split_h5_files",
    "Compose",
    "CenterCoords",
    "AddGaussianNoise",
    "MaskAtoms",
    "EpochRandomSampler",
    "OffsetSampler",
    "GeometricStructureTask",
    "GeometricStructureLoss",
    "PretrainTaskModel",
    "PretrainTrainer",
    "build_geometric_model",
]
