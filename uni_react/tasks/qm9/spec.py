"""QM9 task specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QM9TaskSpec:
    name: str = "qm9"
    variant: str = "default"
    split: str = "egnn"
    target_index_variant: str = "default"
    center_coords: bool = True


def resolve_qm9_task_spec(cfg) -> QM9TaskSpec:
    variant = cfg.task_variant or ("gotennet" if cfg.model_name == "gotennet_l" else "default")
    if variant == "gotennet":
        if cfg.task_variant:
            split = cfg.split or "gotennet"
            target_index_variant = cfg.qm9_target_variant or "gotennet"
            center_coords = not bool(cfg.no_center_coords)
        else:
            split = "gotennet"
            target_index_variant = "gotennet"
            center_coords = False
    else:
        split = cfg.split or "egnn"
        target_index_variant = cfg.qm9_target_variant or "default"
        center_coords = not bool(cfg.no_center_coords)
    return QM9TaskSpec(
        variant=variant,
        split=split,
        target_index_variant=target_index_variant,
        center_coords=center_coords,
    )
