"""Common QM9 builders."""

from .loss import QM9RegressionLoss
from .model import QM9FineTuneNet
from .trainer import FinetuneQM9Trainer


def build_qm9_model(cfg, model_spec, targets, task_spec=None, **kwargs):
    del task_spec, kwargs
    return QM9FineTuneNet(
        emb_dim=cfg.emb_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        head_dropout=cfg.head_dropout,
        num_targets=len(targets),
        descriptor=model_spec.build_backbone(cfg),
    )


__all__ = ["QM9RegressionLoss", "QM9FineTuneNet", "FinetuneQM9Trainer", "build_qm9_model"]
