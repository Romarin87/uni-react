"""Composable data transforms for molecular pretraining."""
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ..registry import TRANSFORM_REGISTRY


class Compose:
    """Apply a sequence of transforms in order.

    Args:
        transforms: List of callables that accept and return a sample dict.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = ["Compose("]
        for t in self.transforms:
            lines.append(f"  {t!r}")
        lines.append(")")
        return "\n".join(lines)


@TRANSFORM_REGISTRY.register("center_coords")
class CenterCoords:
    """Subtract the centroid from all atom coordinates.

    Config example::

        transforms:
          - type: center_coords
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        coords: Tensor = sample["coords"]
        pad: Optional[Tensor] = sample.get("atom_padding")
        if pad is not None:
            valid = (~pad).float().unsqueeze(-1)  # (N, 1)
            centroid = (coords * valid).sum(0) / (valid.sum() + 1e-8)
        else:
            centroid = coords.mean(0)
        sample["coords"] = coords - centroid
        return sample

    def __repr__(self) -> str:
        return "CenterCoords()"


@TRANSFORM_REGISTRY.register("add_gaussian_noise")
class AddGaussianNoise:
    """Add i.i.d. Gaussian noise to atom coordinates.

    Config example::

        transforms:
          - type: add_gaussian_noise
            std: 0.1
    """

    def __init__(self, std: float = 0.1) -> None:
        self.std = float(std)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        coords: Tensor = sample["coords"]
        noise = torch.randn_like(coords) * self.std
        sample["coords_noisy"] = coords + noise
        if "coords_target" not in sample:
            sample["coords_target"] = coords
        return sample

    def __repr__(self) -> str:
        return f"AddGaussianNoise(std={self.std})"


@TRANSFORM_REGISTRY.register("mask_atoms")
class MaskAtoms:
    """Randomly mask a fraction of atoms by replacing their atomic numbers.

    Config example::

        transforms:
          - type: mask_atoms
            ratio: 0.15
            mask_token_id: 94
            min_masked: 1
    """

    def __init__(
        self,
        ratio: float = 0.15,
        mask_token_id: int = 94,
        min_masked: int = 1,
        max_masked: Optional[int] = None,
    ) -> None:
        self.ratio = float(ratio)
        self.mask_token_id = int(mask_token_id)
        self.min_masked = int(min_masked)
        self.max_masked = max_masked

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        atomic_numbers: Tensor = sample["atomic_numbers"]  # (N,)
        pad: Optional[Tensor] = sample.get("atom_padding")

        n_total = int(atomic_numbers.shape[0])
        valid_idx = [
            i for i in range(n_total)
            if pad is None or not pad[i].item()
        ]
        n_valid = len(valid_idx)
        n_mask = max(self.min_masked, int(round(n_valid * self.ratio)))
        if self.max_masked is not None:
            n_mask = min(n_mask, self.max_masked)
        n_mask = min(n_mask, n_valid)

        masked_idx = random.sample(valid_idx, n_mask)
        input_atomic_numbers = atomic_numbers.clone()
        input_atomic_numbers[masked_idx] = self.mask_token_id

        mask = torch.zeros(n_total, dtype=torch.bool)
        mask[masked_idx] = True

        sample["input_atomic_numbers"] = input_atomic_numbers
        sample["target_atomic_numbers"] = atomic_numbers
        sample["mask"] = mask
        return sample

    def __repr__(self) -> str:
        return (
            f"MaskAtoms(ratio={self.ratio}, mask_token_id={self.mask_token_id}, "
            f"min_masked={self.min_masked}, max_masked={self.max_masked})"
        )
