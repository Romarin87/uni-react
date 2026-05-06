"""Registry for independently implemented joint tasks."""
from __future__ import annotations

from typing import Dict, Iterable

from .common import TaskAdapter, TaskSpec
from .atom_mask import TASK_SPEC as ATOM_MASK
from .charge import TASK_SPEC as CHARGE
from .coord_denoise import TASK_SPEC as COORD_DENOISE
from .electron_density import TASK_SPEC as ELECTRON_DENSITY
from .fukui import TASK_SPEC as FUKUI
from .vea import TASK_SPEC as VEA
from .vip import TASK_SPEC as VIP


TASK_SPECS: Dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        ATOM_MASK,
        COORD_DENOISE,
        CHARGE,
        ELECTRON_DENSITY,
        VIP,
        VEA,
        FUKUI,
    )
}


def get_task_spec(task_name: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported joint task: {task_name}") from exc


def build_adapter(
    task_name: str,
    task_cfg: Dict,
    *,
    run_cfg: Dict,
    model_cfg: Dict,
    advanced_cfg: Dict,
) -> TaskAdapter:
    spec = get_task_spec(task_name)
    return spec.adapter_cls(task_name, task_cfg, run_cfg, model_cfg, advanced_cfg)


def supported_task_names() -> Iterable[str]:
    return tuple(TASK_SPECS)
