"""Protocol for molecular encoder (backbone) implementations."""
from typing import Dict, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class MolecularEncoderProtocol(Protocol):
    """Structural contract every backbone encoder must satisfy.

    An encoder receives raw atomic features + 3-D coordinates and returns a
    dict of intermediate representations.  The dict **must** contain at least:

    - ``"node_feats"``  – per-atom embeddings  ``(B, N, D)``
    - ``"graph_feats"`` – pooled graph-level embedding  ``(B, D)``
    - ``"atom_padding"`` – boolean padding mask  ``(B, N)``  (``True`` = pad)

    Additional keys (e.g. equivariant vectors) are allowed and will be
    forwarded to task heads transparently.
    """

    emb_dim: int

    def forward(
        self,
        input_atomic_numbers: Tensor,
        coords_noisy: Tensor,
        atom_padding: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Encode a batch of molecules.

        Args:
            input_atomic_numbers: Atomic number tokens ``(B, N)``.
            coords_noisy: 3-D coordinates (possibly noisy) ``(B, N, 3)``.
            atom_padding: Boolean mask ``(B, N)``, ``True`` = padding atom.

        Returns:
            Dict with at minimum ``node_feats``, ``graph_feats``, ``atom_padding``.
        """
        ...

    def __call__(
        self,
        input_atomic_numbers: Tensor,
        coords_noisy: Tensor,
        atom_padding: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Make encoders callable (mirrors nn.Module.__call__)."""
        ...
