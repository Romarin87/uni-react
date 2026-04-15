"""Weights & Biases logger."""
from typing import Any, Dict, Optional

from ..registry import LOGGER_REGISTRY


@LOGGER_REGISTRY.register("wandb")
class WandbLogger:
    """Logs metrics to Weights & Biases.

    Requires ``wandb``:  ``pip install wandb``.

    Config example::

        logger:
          type: wandb
          project: uni-react
          name: pretrain_base
    """

    def __init__(
        self,
        project: str = "uni-react",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        rank: int = 0,
    ) -> None:
        self._rank = rank
        self._current_rank: Optional[int] = None
        self._project = project
        self._name = name
        self._entity = entity
        self._run = None

    def set_rank(self, rank: int) -> None:
        self._current_rank = rank
        if rank == self._rank:
            try:
                import wandb
                self._run = wandb.init(
                    project=self._project,
                    name=self._name,
                    entity=self._entity,
                    resume="allow",
                )
            except ImportError:
                raise ImportError("pip install wandb")

    def _should_log(self) -> bool:
        return (self._current_rank is None) or (self._current_rank == self._rank)

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        if not self._should_log() or self._run is None:
            return
        import wandb
        prefixed = {f"{phase}/{k}": v for k, v in metrics.items()}
        wandb.log(prefixed, step=step)

    def log_config(self, config: Dict[str, Any]) -> None:
        if not self._should_log() or self._run is None:
            return
        import wandb
        wandb.config.update(config, allow_val_change=True)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
