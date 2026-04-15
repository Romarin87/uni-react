"""Stateful metric accumulators for training loops."""
from typing import Dict, List, Optional


class ScalarAccumulator:
    """Running weighted sum accumulator for a single scalar metric.

    Accumulates ``value * weight`` and a total weight, then reports the
    weighted mean via :meth:`compute`.  Suitable for per-sample averaging
    where each batch may have a different number of valid elements.

    Example::

        acc = ScalarAccumulator()
        for batch in loader:
            loss = compute_loss(batch)
            acc.update(loss.item(), weight=len(batch))
        print(acc.compute())   # weighted mean loss
        acc.reset()
    """

    def __init__(self) -> None:
        self._sum: float = 0.0
        self._weight: float = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        """Add a new observation.

        Args:
            value: The metric value (already reduced over the batch).
            weight: Number of samples this value represents.
        """
        self._sum    += float(value) * float(weight)
        self._weight += float(weight)

    def compute(self) -> float:
        """Return the weighted mean, or 0.0 if no updates have been made."""
        if self._weight == 0.0:
            return 0.0
        return self._sum / self._weight

    def reset(self) -> None:
        """Reset to initial state."""
        self._sum    = 0.0
        self._weight = 0.0

    def __repr__(self) -> str:
        return f"ScalarAccumulator(sum={self._sum:.4f}, weight={self._weight:.1f})"


class MetricBag:
    """A named collection of :class:`ScalarAccumulator` instances.

    Provides a dict-like interface so trainers can update and read metrics
    without managing individual accumulators.

    Example::

        bag = MetricBag(["loss", "mae", "consistency_acc"])
        for batch in loader:
            out = model(batch)
            bag.update("loss", loss.item(), weight=bs)
            bag.update("mae",  mae_val,      weight=bs)
        metrics = bag.compute()   # {"loss": 0.42, "mae": 0.07, ...}
        bag.reset()
    """

    def __init__(self, keys: List[str]) -> None:
        self._accs: Dict[str, ScalarAccumulator] = {
            k: ScalarAccumulator() for k in keys
        }

    def update(self, key: str, value: float, weight: float = 1.0) -> None:
        """Update a named accumulator.

        Silently creates the accumulator if *key* was not in the initial list.
        """
        if key not in self._accs:
            self._accs[key] = ScalarAccumulator()
        self._accs[key].update(value, weight)

    def update_dict(
        self,
        values: Dict[str, float],
        weight: float = 1.0,
        keys: Optional[List[str]] = None,
    ) -> None:
        """Batch-update multiple accumulators from a dict.

        Args:
            values: Mapping of metric name → value.
            weight: Shared weight applied to all values.
            keys: If provided, only update these keys (subset of *values*).
        """
        selected = keys if keys is not None else list(values.keys())
        for k in selected:
            if k in values:
                self.update(k, values[k], weight)

    def compute(self) -> Dict[str, float]:
        """Return a dict of all weighted means."""
        return {k: acc.compute() for k, acc in self._accs.items()}

    def reset(self) -> None:
        """Reset all accumulators."""
        for acc in self._accs.values():
            acc.reset()

    def keys(self) -> List[str]:
        return list(self._accs.keys())

    def get_sums(self) -> Dict[str, float]:
        """Return the raw sum from each accumulator.
        
        Returns:
            Dict mapping metric name to accumulated sum.
        """
        return {k: acc._sum for k, acc in self._accs.items()}

    def get_weight(self) -> float:
        """Return the weight from the first accumulator.
        
        Assumes all accumulators have the same weight (typical in training loops).
        
        Returns:
            The weight value, or 0.0 if no accumulators exist.
        """
        if not self._accs:
            return 0.0
        return next(iter(self._accs.values()))._weight

    def __repr__(self) -> str:
        computed = self.compute()
        parts = ", ".join(f"{k}={v:.4f}" for k, v in computed.items())
        return f"MetricBag({parts})"
