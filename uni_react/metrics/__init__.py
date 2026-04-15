"""Evaluation metrics for uni_react.

All metrics are pure functions or lightweight stateful accumulators.
Trainers call these instead of computing metrics inline.

Available
---------
:func:`mae`                   Mean absolute error
:func:`rmse`                  Root mean squared error
:func:`binary_accuracy`       Binary classification accuracy from logits
:class:`ScalarAccumulator`    Running mean accumulator for scalar metrics
:class:`MetricBag`            Named collection of ScalarAccumulators
"""
from .regression import mae, rmse
from .classification import binary_accuracy
from .accumulator import MetricBag, ScalarAccumulator

__all__ = [
    "mae",
    "rmse",
    "binary_accuracy",
    "ScalarAccumulator",
    "MetricBag",
]
