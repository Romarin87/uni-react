"""Protocol for Trainer implementations."""
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class TrainerProtocol(Protocol):
    """Minimal contract every Trainer must satisfy.

    The ``fit`` method encapsulates the complete train + validate loop.
    """

    def fit(
        self,
        start_epoch: int = 1,
        end_epoch: Optional[int] = None,
    ) -> None:
        """Run the training loop from *start_epoch* to *end_epoch* inclusive.

        Args:
            start_epoch: First epoch to train (1-indexed, inclusive).
            end_epoch: Last epoch (inclusive).  Defaults to the value set
                during construction.
        """
        ...

    def save_checkpoint(
        self,
        epoch: int,
        tag: str = "latest",
        include_optimizer: bool = True,
    ) -> None:
        """Persist a checkpoint to disk.

        Args:
            epoch: Current epoch number.
            tag: Filename suffix (e.g. ``"best"``, ``"latest"``).
            include_optimizer: Whether to include optimizer state.
        """
        ...
