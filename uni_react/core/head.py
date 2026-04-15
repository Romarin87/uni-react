"""Protocol for task-head implementations."""
from typing import Dict, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class TaskHeadProtocol(Protocol):
    """Structural contract every prediction head must satisfy.

    A head receives the encoder's output dict and returns a dict of
    predictions.  The returned keys are head-specific (e.g.
    ``"atom_logits"``, ``"coords_denoised"``, ``"charge_pred"``).

    The head is also responsible for computing its own loss via
    :meth:`compute_loss`.  This keeps the loss co-located with the
    prediction logic and avoids spreading loss code across training scripts.
    """

    name: str
    """Unique identifier used by the registry."""

    def forward(
        self,
        descriptors: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Produce predictions from encoder output.

        Args:
            descriptors: Output dict from :class:`MolecularEncoderProtocol`.

        Returns:
            Dict of prediction tensors.
        """
        ...

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Tensor:
        """Compute a scalar loss for this head's predictions.

        Args:
            outputs: Merged prediction dict (from all heads).
            batch: The original data batch.

        Returns:
            Scalar loss tensor.
        """
        ...

    def __call__(
        self,
        descriptors: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        ...
