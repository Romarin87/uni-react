"""Density pretraining task specification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DensityTaskSpec:
    name: str = "density"
    train_mode: str = "electron_density"


def resolve_density_task_spec(cfg) -> DensityTaskSpec:
    return DensityTaskSpec()
