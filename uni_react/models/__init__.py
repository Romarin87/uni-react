"""High-level backbone builders."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

__all__ = ["ModelSpec", "build_model_spec", "build_qm9_model_spec"]


@dataclass(frozen=True)
class ModelSpec:
    """High-level backbone definition used by task builders."""

    name: str
    build_backbone: Callable


def build_model_spec(model_name: str) -> ModelSpec:
    if model_name not in {"single_mol", "gotennet_l"}:
        raise ValueError(f"Unsupported model '{model_name}'.")
    module = import_module(f".{model_name}", __name__)
    return ModelSpec(
        name=model_name,
        build_backbone=module.build_backbone,
    )


def build_qm9_model_spec(model_name: str, task_variant: str) -> ModelSpec:
    supported_variants = {
        "single_mol": {"default"},
        "gotennet_l": {"gotennet"},
    }
    if model_name not in supported_variants:
        raise ValueError(f"Unsupported QM9 model '{model_name}'.")
    if task_variant not in supported_variants[model_name]:
        raise ValueError(
            f"Model '{model_name}' does not support QM9 variant '{task_variant}'. "
            f"Supported: {tuple(sorted(supported_variants[model_name]))}"
        )
    return build_model_spec(model_name)
