"""Collate utilities for variable-length molecular batches."""
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


def collate_mol_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of per-molecule sample dicts into a padded batch.

    Tensors with shape ``(N, ...)`` (atom-level) are padded to the maximum
    atom count in the batch.  Scalar tensors and non-tensor values are
    stacked / collected into lists as appropriate.

    Args:
        samples: List of sample dicts from a :class:`MolDatasetProtocol`.

    Returns:
        A single batch dict ready for the model.
    """
    if not samples:
        return {}

    keys = samples[0].keys()
    batch: Dict[str, Any] = {}

    for key in keys:
        values = [s[key] for s in samples]
        if not isinstance(values[0], Tensor):
            batch[key] = values
            continue

        # Scalar tensors – just stack
        if values[0].ndim == 0:
            batch[key] = torch.stack(values)
            continue

        # 1-D atom-level tensors – pad to max length
        if values[0].ndim >= 1:
            max_len = max(v.shape[0] for v in values)
            if all(v.shape[0] == max_len for v in values):
                batch[key] = torch.stack(values)
            else:
                batch[key] = _pad_sequence(values, max_len)

    # Build atom_padding mask if not already present
    if "atom_padding" not in batch:
        if "atomic_numbers" in batch and isinstance(batch["atomic_numbers"], Tensor):
            # True where we padded (all-zero rows as sentinel)
            batch["atom_padding"] = (batch["atomic_numbers"] == 0)

    return batch


def _pad_sequence(tensors: List[Tensor], max_len: int) -> Tensor:
    """Pad a list of tensors along dim-0 to *max_len* with zeros."""
    out = torch.zeros(
        len(tensors), max_len, *tensors[0].shape[1:],
        dtype=tensors[0].dtype,
    )
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out
