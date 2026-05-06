"""Composable data transforms for geometric pretraining."""

import random
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor


class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        lines = ["Compose("]
        for transform in self.transforms:
            lines.append(f"  {transform!r}")
        lines.append(")")
        return "\n".join(lines)


class CenterCoords:
    """Subtract the centroid from all atom coordinates."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        coords: Tensor = sample["coords"]
        pad: Optional[Tensor] = sample.get("atom_padding")
        if pad is not None:
            valid = (~pad).float().unsqueeze(-1)
            centroid = (coords * valid).sum(0) / (valid.sum() + 1e-8)
        else:
            centroid = coords.mean(0)
        sample["coords"] = coords - centroid
        return sample

    def __repr__(self) -> str:
        return "CenterCoords()"


class AddGaussianNoise:
    """Add i.i.d. Gaussian noise to atom coordinates."""

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


class MaskAtoms:
    """Randomly mask a fraction of atoms by replacing their atomic numbers."""

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
        atomic_numbers: Tensor = sample["atomic_numbers"]
        pad: Optional[Tensor] = sample.get("atom_padding")

        n_total = int(atomic_numbers.shape[0])
        valid_idx = [i for i in range(n_total) if pad is None or not pad[i].item()]
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
