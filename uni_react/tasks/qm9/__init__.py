"""QM9 task helpers."""

from .common import build_qm9_model as build_common_qm9_model
from .dataset import QM9PyGDataset, QM9_SPLIT_MODES, QM9_TARGETS, build_qm9_pyg_splits, collate_fn_qm9, get_qm9_atomref, get_qm9_target_index_map
from .entry import run_qm9_entry
from .gotennet_l import build_qm9_model as build_gotennet_l_qm9_model
from .results import write_qm9_outputs
from .runtime import build_qm9_trainer, finalize_qm9_training, parse_targets, prepare_qm9_config
from .spec import QM9TaskSpec, resolve_qm9_task_spec


def build_qm9_model(cfg, model_spec, targets, task_spec, **kwargs):
    if cfg.model_name == "gotennet_l":
        return build_gotennet_l_qm9_model(cfg, model_spec, targets, task_spec, **kwargs)
    return build_common_qm9_model(cfg, model_spec, targets, task_spec, **kwargs)


__all__ = [
    "QM9TaskSpec",
    "QM9PyGDataset",
    "QM9_SPLIT_MODES",
    "QM9_TARGETS",
    "build_qm9_pyg_splits",
    "collate_fn_qm9",
    "get_qm9_atomref",
    "get_qm9_target_index_map",
    "run_qm9_entry",
    "resolve_qm9_task_spec",
    "write_qm9_outputs",
    "build_qm9_model",
    "parse_targets",
    "prepare_qm9_config",
    "build_qm9_trainer",
    "finalize_qm9_training",
]
