"""Compatibility exports for joint task adapters."""
from __future__ import annotations

from ..atom_mask import AtomMaskAdapter
from ..charge import ChargeAdapter
from ..common import DatasetBuildResult, MoleculeTaskAdapter, TaskAdapter
from ..coord_denoise import CoordDenoiseAdapter
from ..electron_density import ElectronDensityAdapter
from ..fukui import FukuiAdapter
from ..registry import build_adapter, supported_task_names
from ..vea import VeaAdapter
from ..vip import VipAdapter

__all__ = [
    "AtomMaskAdapter",
    "ChargeAdapter",
    "CoordDenoiseAdapter",
    "DatasetBuildResult",
    "ElectronDensityAdapter",
    "FukuiAdapter",
    "MoleculeTaskAdapter",
    "TaskAdapter",
    "VeaAdapter",
    "VipAdapter",
    "build_adapter",
    "supported_task_names",
]
