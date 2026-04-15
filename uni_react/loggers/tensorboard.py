"""TensorBoard logger."""
import json
from typing import Any, Dict, Optional

from ..registry import LOGGER_REGISTRY


@LOGGER_REGISTRY.register("tensorboard")
class TensorBoardLogger:
    """Logs metrics to TensorBoard.

    Requires ``tensorboard``:  ``pip install tensorboard``.

    Config example::

        logger:
          type: tensorboard
          log_dir: runs/my_exp/tb
    """

    def __init__(self, log_dir: str = "runs/tb", rank: int = 0) -> None:
        self._rank = rank
        self._current_rank: Optional[int] = None
        self._log_dir = log_dir
        self._writer = None

    def set_rank(self, rank: int) -> None:
        self._current_rank = rank
        if rank == self._rank:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=self._log_dir)
            except ImportError:
                raise ImportError("pip install tensorboard")

    def _should_log(self) -> bool:
        return (self._current_rank is None) or (self._current_rank == self._rank)

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        if not self._should_log() or self._writer is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"{phase}/{k}", v, global_step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        if not self._should_log() or self._writer is None:
            return
        self._writer.add_text("config", json.dumps(config, indent=2, default=str))

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
