"""Unit tests for shared and task-specific metrics."""
import pytest
import torch

from uni_react.training import MetricBag, ScalarAccumulator
from uni_react.tasks.reaction.common.metrics import binary_accuracy

class TestBinaryAccuracy:
    def test_all_correct(self):
        logits = torch.tensor([5.0, -5.0])
        labels = torch.tensor([1.0,  0.0])
        assert float(binary_accuracy(logits, labels)) == 1.0

    def test_all_wrong(self):
        logits = torch.tensor([-5.0, 5.0])
        labels = torch.tensor([ 1.0, 0.0])
        assert float(binary_accuracy(logits, labels)) == 0.0

    def test_half_correct(self):
        logits = torch.tensor([5.0, 5.0])
        labels = torch.tensor([1.0, 0.0])
        assert abs(float(binary_accuracy(logits, labels)) - 0.5) < 1e-5

    def test_with_mask(self):
        logits = torch.tensor([5.0, 5.0, -5.0])
        labels = torch.tensor([1.0, 0.0,  1.0])
        mask   = torch.tensor([True, False, False])
        assert float(binary_accuracy(logits, labels, mask=mask)) == 1.0


# ------------------------------------------------------------------
# accumulators
# ------------------------------------------------------------------

class TestScalarAccumulator:
    def test_basic(self):
        acc = ScalarAccumulator()
        acc.update(2.0, weight=2)
        acc.update(4.0, weight=2)
        assert abs(acc.compute() - 3.0) < 1e-6

    def test_empty_returns_zero(self):
        acc = ScalarAccumulator()
        assert acc.compute() == 0.0

    def test_reset(self):
        acc = ScalarAccumulator()
        acc.update(10.0, weight=1)
        acc.reset()
        assert acc.compute() == 0.0


class TestMetricBag:
    def test_update_and_compute(self):
        bag = MetricBag(["loss", "mae"])
        bag.update("loss", 2.0, weight=4)
        bag.update("loss", 6.0, weight=4)
        bag.update("mae",  1.0, weight=8)
        metrics = bag.compute()
        assert abs(metrics["loss"] - 4.0) < 1e-6
        assert abs(metrics["mae"]  - 1.0) < 1e-6

    def test_update_dict(self):
        bag = MetricBag(["a", "b"])
        bag.update_dict({"a": 1.0, "b": 2.0}, weight=1)
        bag.update_dict({"a": 3.0, "b": 4.0}, weight=1)
        m = bag.compute()
        assert abs(m["a"] - 2.0) < 1e-6
        assert abs(m["b"] - 3.0) < 1e-6

    def test_reset(self):
        bag = MetricBag(["x"])
        bag.update("x", 5.0, weight=1)
        bag.reset()
        assert bag.compute()["x"] == 0.0

    def test_unknown_key_auto_creates(self):
        bag = MetricBag(["a"])
        bag.update("new_key", 7.0, weight=1)
        assert abs(bag.compute()["new_key"] - 7.0) < 1e-6
