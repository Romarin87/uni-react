"""Classification metrics."""
from typing import Optional

import torch
from torch import Tensor


def binary_accuracy(
    logits: Tensor,
    labels: Tensor,
    threshold: float = 0.5,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Binary classification accuracy from raw logits.

    Args:
        logits: Raw (pre-sigmoid) model outputs, shape ``(N,)``.
        labels: Ground-truth binary labels (0.0 or 1.0), shape ``(N,)``.
        threshold: Decision threshold applied after sigmoid.
        mask: Boolean mask (``True`` = valid element).

    Returns:
        Scalar accuracy tensor in ``[0, 1]``.
    """
    pred = (torch.sigmoid(logits) >= threshold).float()
    correct = (pred == labels).float()
    if mask is not None:
        correct = correct[mask]
    return correct.mean() if correct.numel() > 0 else correct.new_zeros([])
