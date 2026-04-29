"""Training utilities shared across all training entry-points.

Exports are resolved lazily so lightweight modules such as
``uni_react.training.logger`` and ``uni_react.training.losses`` do not require
importing torch-only trainer infrastructure.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MetricBag": ".accumulator",
    "ScalarAccumulator": ".accumulator",
    "BaseTrainer": ".base",
    "move_batch_to_device": ".batch",
    "build_checkpoint_dict": ".checkpoint",
    "load_restart_checkpoint": ".checkpoint",
    "validate_restart_config": ".checkpoint",
    "cleanup_distributed": ".distributed",
    "init_distributed": ".distributed",
    "is_main_process": ".distributed",
    "ConsoleLogger": ".logger",
    "LoggerProtocol": ".logger",
    "ResultWriter": ".logger",
    "build_event_logger": ".logger",
    "RegressionLoss": ".losses",
    "regression_loss": ".losses",
    "build_optimizer": ".optimizer",
    "ConstantScheduler": ".scheduler",
    "WarmupCosineScheduler": ".scheduler",
    "WarmupLinearScheduler": ".scheduler",
    "build_scheduler": ".scheduler",
    "set_seed": ".seed",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
