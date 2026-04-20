"""Stateful metric accumulators shared by training loops."""
from typing import Dict, List, Optional


class ScalarAccumulator:
    """Running weighted sum accumulator for a single scalar metric."""

    def __init__(self) -> None:
        self._sum: float = 0.0
        self._weight: float = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        self._sum += float(value) * float(weight)
        self._weight += float(weight)

    def compute(self) -> float:
        if self._weight == 0.0:
            return 0.0
        return self._sum / self._weight

    def reset(self) -> None:
        self._sum = 0.0
        self._weight = 0.0

    def __repr__(self) -> str:
        return f"ScalarAccumulator(sum={self._sum:.4f}, weight={self._weight:.1f})"


class MetricBag:
    """A named collection of :class:`ScalarAccumulator` instances."""

    def __init__(self, keys: List[str]) -> None:
        self._accs: Dict[str, ScalarAccumulator] = {
            key: ScalarAccumulator() for key in keys
        }

    def update(self, key: str, value: float, weight: float = 1.0) -> None:
        if key not in self._accs:
            self._accs[key] = ScalarAccumulator()
        self._accs[key].update(value, weight)

    def update_dict(
        self,
        values: Dict[str, float],
        weight: float = 1.0,
        keys: Optional[List[str]] = None,
    ) -> None:
        selected = keys if keys is not None else list(values.keys())
        for key in selected:
            if key in values:
                self.update(key, values[key], weight)

    def compute(self) -> Dict[str, float]:
        return {key: acc.compute() for key, acc in self._accs.items()}

    def reset(self) -> None:
        for acc in self._accs.values():
            acc.reset()

    def keys(self) -> List[str]:
        return list(self._accs.keys())

    def get_sums(self) -> Dict[str, float]:
        return {key: acc._sum for key, acc in self._accs.items()}

    def get_weight(self) -> float:
        if not self._accs:
            return 0.0
        return next(iter(self._accs.values()))._weight

    def __repr__(self) -> str:
        computed = self.compute()
        parts = ", ".join(f"{key}={value:.4f}" for key, value in computed.items())
        return f"MetricBag({parts})"
