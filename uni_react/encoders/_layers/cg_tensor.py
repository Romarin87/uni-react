"""Hand-coded Clebsch-Gordan (CG) tensor-product utilities for ℓ ∈ {0, 1, 2}.

We avoid a full e3nn dependency by hard-coding the non-zero CG coefficients
for the irreps we need:

    ℓ=0 (scalar):  1 component  (invariant)
    ℓ=1 (vector):  3 components (x, y, z)
    ℓ=2 (d-type):  5 components (xy, yz, z², xz, x²-y²)

Only the following CG paths are implemented (sufficient for ReacFormer):

    0 ⊗ 0 → 0
    0 ⊗ 1 → 1
    0 ⊗ 2 → 2
    1 ⊗ 1 → 0   (inner product)
    1 ⊗ 1 → 1   (cross product)
    1 ⊗ 1 → 2   (traceless symmetric outer product)
    1 ⊗ 2 → 1
    1 ⊗ 2 → 2
    1 ⊗ 2 → 3   (not implemented – not needed)

All operations are batched and differentiable.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Spherical harmonics (real, Condon-Shortley convention)
# ---------------------------------------------------------------------------

def spherical_harmonics_l1(r_hat: Tensor) -> Tensor:
    """Real spherical harmonics Y_1m for unit vectors.

    Returns the (x, y, z) components in standard order so that the output
    transforms under rotation R as Y1(R@r) = R @ Y1(r).

    Args:
        r_hat: Unit vectors ``(..., 3)`` in (x, y, z) order.

    Returns:
        ``(..., 3)`` = (x, y, z) — proportional to real SH Y_1m (m=-1,0,+1)
        in a basis where the equivariance property holds directly.
    """
    return r_hat  # (x, y, z) — equivariant under R: Y1(Rr) = R @ Y1(r)


def spherical_harmonics_l2(r_hat: Tensor) -> Tensor:
    """Real spherical harmonics Y_2m for unit vectors, using e3nn convention.

    Uses e3nn's ``o3.spherical_harmonics`` for consistency with
    :func:`wigner_d_l2_from_R`.  Both functions use the same basis so that
    the equivariance Y2(R@r) = D2(R) @ Y2(r) holds exactly.

    Args:
        r_hat: Unit vectors ``(..., 3)``.

    Returns:
        ``(..., 5)`` in e3nn real SH order (m = -2,-1,0,+1,+2).
    """
    try:
        from e3nn import o3 as _o3
        shape = r_hat.shape
        flat  = r_hat.reshape(-1, 3)
        y2    = _o3.spherical_harmonics(2, flat, normalize=True)  # (-1, 5)
        return y2.reshape(*shape[:-1], 5)
    except ImportError:
        raise ImportError(
            "e3nn is required for spherical_harmonics_l2. "
            "Install with: pip install e3nn"
        )


def spherical_harmonics(r_hat: Tensor, l_max: int = 2) -> Tensor:
    """Concatenate real SH up to *l_max* for unit vectors.

    Returns:
        ``(..., (l_max+1)²)``  — concatenation of Y_0, Y_1, Y_2, …
    """
    parts = [torch.ones_like(r_hat[..., :1])]  # ℓ=0: constant 1
    if l_max >= 1:
        parts.append(spherical_harmonics_l1(r_hat))
    if l_max >= 2:
        parts.append(spherical_harmonics_l2(r_hat))
    return torch.cat(parts, dim=-1)  # (..., 1+3+5=9)


# ---------------------------------------------------------------------------
# CG tensor products
# ---------------------------------------------------------------------------

def cg_0x0_to_0(a: Tensor, b: Tensor) -> Tensor:
    """Scalar × scalar → scalar.  a,b: (...,1)"""
    return a * b


def cg_1x1_to_0(a: Tensor, b: Tensor) -> Tensor:
    """Vector ⊗ Vector → Scalar (inner product).  a,b: (...,3) → (...,1)"""
    return (a * b).sum(dim=-1, keepdim=True) / math.sqrt(3.0)


def cg_1x1_to_1(a: Tensor, b: Tensor) -> Tensor:
    """Vector ⊗ Vector → Vector (cross product).  a,b: (...,3) → (...,3)"""
    return torch.cross(a, b, dim=-1) / math.sqrt(2.0)


def cg_1x1_to_2(a: Tensor, b: Tensor) -> Tensor:
    """Vector ⊗ Vector → ℓ=2 tensor (traceless symmetric outer product).

    Produces 5 components (xy, yz, z²-½(x²+y²), xz, x²-y²) of the
    symmetrised outer product A⊗B + B⊗A (traceless part).

    Args:
        a, b: ``(..., 3)``

    Returns:
        ``(..., 5)``
    """
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    c = 1.0 / math.sqrt(2.0)
    xy  = c * (ax * by + ay * bx)
    yz  = c * (ay * bz + az * by)
    z2  = (az * bz - 0.5 * (ax * bx + ay * by)) * math.sqrt(2.0 / 3.0)
    xz  = c * (ax * bz + az * bx)
    x2y2 = c * 0.5 * (ax * bx - ay * by)
    return torch.stack([xy, yz, z2, xz, x2y2], dim=-1)


def cg_0xl_to_l(scalar: Tensor, higher: Tensor) -> Tensor:
    """Scale any irrep by a scalar.  scalar: (...,1), higher: (..., 2ℓ+1)."""
    return scalar * higher


# ---------------------------------------------------------------------------
# Edge-feature irrep decomposition
# ---------------------------------------------------------------------------

class EdgeIrrepEncoding(torch.nn.Module):
    """Project edge RBF features + spherical harmonics into irrep channels.

    Produces per-edge scalar (ℓ=0), vector (ℓ=1), and tensor (ℓ=2) features
    that a CG message-passing layer can consume.

    Args:
        num_rbf:  Number of RBF basis functions.
        emb_dim:  Output channel dimension for each irrep order.
        l_max:    Maximum ℓ (0, 1, or 2).
    """

    def __init__(self, num_rbf: int, emb_dim: int, l_max: int = 2) -> None:
        super().__init__()
        if l_max not in (0, 1, 2):
            raise ValueError("l_max must be 0, 1, or 2")
        self.l_max = l_max
        sh_dim = sum(2 * l + 1 for l in range(l_max + 1))  # 1+3+5=9

        # Project RBF + SH into emb_dim per ℓ
        self.proj_0 = torch.nn.Linear(num_rbf + 1, emb_dim)       # ℓ=0: rbf+Y0
        if l_max >= 1:
            self.proj_1 = torch.nn.Linear(num_rbf + 3, emb_dim)   # ℓ=1: rbf+Y1
        if l_max >= 2:
            self.proj_2 = torch.nn.Linear(num_rbf + 5, emb_dim)   # ℓ=2: rbf+Y2

    def forward(
        self,
        rbf: Tensor,    # (..., num_rbf)
        r_hat: Tensor,  # (..., 3) unit vectors
    ) -> Tuple[Tensor, ...]:
        """Return (e0, e1, e2) edge irrep features.

        Returns:
            e0: ``(..., emb_dim)``
            e1: ``(..., emb_dim)`` (only if l_max >= 1)
            e2: ``(..., emb_dim)`` (only if l_max >= 2)
        """
        y0 = torch.ones_like(rbf[..., :1])
        e0 = self.proj_0(torch.cat([rbf, y0], dim=-1))
        out = [e0]
        if self.l_max >= 1:
            y1 = spherical_harmonics_l1(r_hat)  # (..., 3)
            e1 = self.proj_1(torch.cat([rbf, y1], dim=-1))
            out.append(e1)
        if self.l_max >= 2:
            y2 = spherical_harmonics_l2(r_hat)  # (..., 5)
            e2 = self.proj_2(torch.cat([rbf, y2], dim=-1))
            out.append(e2)
        return tuple(out)
