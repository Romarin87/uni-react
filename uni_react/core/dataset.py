"""Protocol for molecular datasets."""
from typing import Dict, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class MolDatasetProtocol(Protocol):
    """Minimal contract for datasets consumed by uni_react Trainers.

    Concrete datasets (HDF5-backed, PyG QM9, …) must implement ``__len__``
    and ``__getitem__``.  The returned sample must be a dict whose tensor
    values are assembled by a collate function into batch dicts.
    """

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Dict:
        """Return a single sample dict."""
        ...
