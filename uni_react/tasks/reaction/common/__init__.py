"""Common reaction builders."""

from .dataset import ReactionTripletH5Dataset, collate_reaction_triplet, split_dataset
from .metrics import binary_accuracy
from .model import ReactionPretrainNet
from .trainer import ReactionPretrainTrainer


def build_reaction_model(cfg, model_spec, task_spec=None):
    del task_spec
    return ReactionPretrainNet(
        descriptor=model_spec.build_backbone(cfg),
        emb_dim=cfg.emb_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        teacher_momentum=cfg.teacher_momentum,
    )


__all__ = [
    "ReactionTripletH5Dataset",
    "collate_reaction_triplet",
    "split_dataset",
    "binary_accuracy",
    "ReactionPretrainNet",
    "ReactionPretrainTrainer",
    "build_reaction_model",
]
