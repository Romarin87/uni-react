"""Batch-level utility helpers for training loops."""
from typing import Dict

import torch


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move all tensor values in *batch* to *device* (non-blocking).

    Non-tensor values are passed through unchanged.
    """
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
