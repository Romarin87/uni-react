"""Protocol for learning-rate schedulers."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class LRSchedulerProtocol(Protocol):
    """Minimal interface for LR schedulers used by Trainers.

    PyTorch's built-in schedulers already satisfy this protocol via duck
    typing.  Custom schedulers (e.g. warmup-cosine) must expose the same
    ``step`` method signature.
    """

    def step(self) -> None:
        """Advance the scheduler by one step (called once per training step)."""
        ...

    def get_last_lr(self) -> list:
        """Return the last computed LR for each param group."""
        ...
