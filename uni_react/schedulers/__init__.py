"""LR scheduler implementations.

All schedulers are registered in :data:`~uni_react.registry.SCHEDULER_REGISTRY`.

Available schedulers
--------------------
``cosine``   :class:`~uni_react.schedulers.cosine.WarmupCosineScheduler`
``linear``   :class:`~uni_react.schedulers.linear.WarmupLinearScheduler`
``none``     :class:`~uni_react.schedulers.constant.ConstantScheduler`
"""
from .constant import ConstantScheduler
from .cosine import WarmupCosineScheduler
from .linear import WarmupLinearScheduler

__all__ = [
    "WarmupCosineScheduler",
    "WarmupLinearScheduler",
    "ConstantScheduler",
]
