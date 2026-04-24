"""High-level backbone builders."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

__all__ = ["ModelSpec", "build_model_spec", "build_qm9_model_spec"]

GOTENNET_MODELS = {
    "gotennet_s",
    "gotennet_b",
    "gotennet_l",
    "gotennet_s_hat",
    "gotennet_b_hat",
    "gotennet_l_hat",
}
SUPPORTED_MODELS = {"single_mol", *GOTENNET_MODELS}


@dataclass(frozen=True)
class ModelSpec:
    """High-level backbone definition used by task builders."""

    name: str
    build_backbone: Callable


def build_model_spec(model_name: str) -> ModelSpec:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'.")
    module_name = "gotennet" if model_name in GOTENNET_MODELS else model_name
    module = import_module(f".{module_name}", __name__)
    return ModelSpec(
        name=model_name,
        build_backbone=module.build_backbone,
    )


def build_qm9_model_spec(model_name: str, task_variant: str) -> ModelSpec:
    supported_variants = {
        "single_mol": {"default"},
        "gotennet_s": {"gotennet"},
        "gotennet_b": {"gotennet"},
        "gotennet_l": {"gotennet"},
        "gotennet_s_hat": {"gotennet"},
        "gotennet_b_hat": {"gotennet"},
        "gotennet_l_hat": {"gotennet"},
    }
    if model_name not in supported_variants:
        raise ValueError(f"Unsupported QM9 model '{model_name}'.")
    if task_variant not in supported_variants[model_name]:
        raise ValueError(
            f"Model '{model_name}' does not support QM9 variant '{task_variant}'. "
            f"Supported: {tuple(sorted(supported_variants[model_name]))}"
        )
    return build_model_spec(model_name)
