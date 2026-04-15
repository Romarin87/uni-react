"""Protocols for loss-function implementations."""
from typing import Dict, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class LossFnProtocol(Protocol):
    """Generic loss function contract.

    A loss function takes the merged prediction dict and the raw batch dict
    and returns a dict whose ``"loss"`` key holds the total scalar loss.
    Additional per-component keys (e.g. ``"atom_loss"``) may also be present
    and will be logged automatically by the Trainer.
    """

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute losses.

        Returns:
            Dict with at least key ``"loss"`` (scalar).
        """
        ...


@runtime_checkable
class PretrainLossFnProtocol(LossFnProtocol, Protocol):
    """Extended loss contract for pretraining pipelines.

    Adds the ability to query which metric keys this loss function exposes so
    that the Trainer can pre-allocate metric accumulators.
    """

    def metric_keys(self) -> tuple:
        """Return the ordered tuple of scalar metric names this loss reports.

        The first element must always be ``"loss"`` (the total).
        """
        ...
