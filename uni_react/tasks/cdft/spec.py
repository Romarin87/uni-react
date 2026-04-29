"""CDFT task training task specification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CDFTTaskSpec:
    name: str = "cdft"
    train_mode: str = "cdft"


def resolve_cdft_task_spec(cfg) -> CDFTTaskSpec:
    return CDFTTaskSpec()
