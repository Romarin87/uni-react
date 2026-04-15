"""Protocol for training-run loggers."""
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class LoggerProtocol(Protocol):
    """Structural contract for logging backends.

    Any backend (console, W&B, TensorBoard, MLflow …) that satisfies this
    interface can be plugged into a Trainer without further changes.
    """

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        """Log a dict of scalar metrics.

        Args:
            metrics: Mapping of metric name → scalar value.
            step: Global step counter (epoch or batch index).
            phase: One of ``"train"`` or ``"val"``.
        """
        ...

    def log_config(self, config: Dict[str, Any]) -> None:
        """Persist the experiment hyper-parameters."""
        ...

    def set_rank(self, rank: int) -> None:
        """Inform the logger of the current distributed rank."""
        ...

    def finish(self) -> None:
        """Finalise the logging session (flush buffers, close connections)."""
        ...
