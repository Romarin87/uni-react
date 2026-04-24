"""GotenNet variant-specific QM9 builders."""

from .model import GotenNetQM9Metadata, GotenNetQM9Net, build_gotennet_qm9_metadata
from .trainer import GotenNetQM9Trainer


def build_qm9_model(cfg, model_spec, targets, task_spec=None, *, atomref=None, mean=None, std=None):
    del task_spec
    if len(targets) != 1:
        raise ValueError("gotennet_* with official QM9 heads only supports single-target runs.")
    target_name = targets[0]
    metadata = build_gotennet_qm9_metadata(
        target=target_name,
        mean=mean,
        std=std,
        atomref=atomref,
    )
    return GotenNetQM9Net(
        descriptor=model_spec.build_backbone(cfg),
        target=target_name,
        metadata=metadata,
    )


__all__ = [
    "GotenNetQM9Metadata",
    "GotenNetQM9Net",
    "build_gotennet_qm9_metadata",
    "GotenNetQM9Trainer",
    "build_qm9_model",
]
