"""Geometric task training task specification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometricTaskSpec:
    name: str = "geometric"
    train_mode: str = "geometric_structure"


def resolve_geometric_task_spec(cfg) -> GeometricTaskSpec:
    return GeometricTaskSpec()
