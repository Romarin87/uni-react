"""Three-body (angular) invariant feature aggregation.

For each center atom i, aggregates over all ordered neighbor pairs (j, k)
within the cutoff to produce an angular descriptor that captures the local
bond-angle distribution.

Computational cost: O(N · K²) where K is the average neighbour count.
For K≈15 and N≈30 this is O(30·225) = O(6750) – perfectly acceptable
for molecule-scale systems.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Legendre polynomials (for angular basis)
# ---------------------------------------------------------------------------

def legendre_basis(cos_theta: Tensor, max_order: int = 4) -> Tensor:
    """Evaluate Legendre polynomials P_0 … P_{max_order} at *cos_theta*.

    Args:
        cos_theta: ``(...)`` tensor of cosine values in [-1, 1].
        max_order: Highest polynomial order (inclusive).

    Returns:
        ``(..., max_order+1)`` where channel k is P_k(cos_theta).
    """
    polys = [torch.ones_like(cos_theta)]          # P_0 = 1
    if max_order >= 1:
        polys.append(cos_theta)                   # P_1 = x
    for n in range(2, max_order + 1):             # Bonnet recursion
        p_n = (
            (2 * n - 1) * cos_theta * polys[-1] - (n - 1) * polys[-2]
        ) / n
        polys.append(p_n)
    return torch.stack(polys, dim=-1)             # (..., max_order+1)


# ---------------------------------------------------------------------------
# Three-body layer
# ---------------------------------------------------------------------------

class ThreeBodyAggregation(torch.nn.Module):
    """Invariant three-body (angle) feature aggregation.

    For each center atom i this layer computes:

    .. math::
        a_i = \\sum_{j \\in \\mathcal{N}(i)} \\sum_{k \\in \\mathcal{N}(i)}
              w(d_{ij}) \\cdot w(d_{ik}) \\cdot P(\\cos \\theta_{ijk})

    where P is a Legendre polynomial basis and w is a learned radial weight.
    The result is projected to *emb_dim* scalars.

    Args:
        emb_dim:       Output channel dimension.
        num_rbf:       Number of RBF radial basis functions.
        legendre_order: Highest Legendre polynomial P_n to include.
        cutoff:        Cutoff radius (Å) – used only for soft-mask.
        hidden_dim:    Hidden dimension of the output MLP.
    """

    def __init__(
        self,
        emb_dim: int,
        num_rbf: int = 64,
        legendre_order: int = 3,
        cutoff: float = 5.0,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.emb_dim       = emb_dim
        self.legendre_order = int(legendre_order)
        self.cutoff        = float(cutoff)

        hidden = hidden_dim or emb_dim
        ang_dim = legendre_order + 1     # number of Legendre polynomials
        # Each pair (j,k) contributes: radial_j * radial_k * angular
        # We project the pair-level angular features per-i via an MLP.
        self.radial_proj = torch.nn.Linear(num_rbf, 1)          # scalar weight per edge
        self.angular_mlp = torch.nn.Sequential(
            torch.nn.Linear(ang_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, emb_dim),
        )
        self.out_norm = torch.nn.LayerNorm(emb_dim)

    def forward(
        self,
        rbf: Tensor,          # (B, N, N, num_rbf)
        r_hat: Tensor,        # (B, N, N, 3) unit vectors r_ij
        access_mask: Tensor,  # (B, N, N)  float, 1 = valid neighbour pair
    ) -> Tensor:
        """Compute three-body invariant features for each atom.

        Args:
            rbf:         Pairwise RBF distance features.
            r_hat:       Unit displacement vectors.
            access_mask: Float mask (1 = within cutoff, 0 = outside/padding).

        Returns:
            ``(B, N, emb_dim)`` per-atom three-body features.
        """
        # Radial weight per edge: (B, N, N, 1)
        w = torch.sigmoid(self.radial_proj(rbf)) * access_mask.unsqueeze(-1)

        # Cosine of bond angle: cos_θ_ijk = r̂_ij · r̂_ik
        # r_hat: (B, N, N, 3)
        # cos_theta[b, i, j, k] = r_hat[b,i,j,:] · r_hat[b,i,k,:]
        # Efficient batch outer product via einsum:
        # (B, i, j, 3) × (B, i, k, 3) → (B, i, j, k)
        cos_theta = torch.einsum("bijx,bikx->bijk", r_hat, r_hat)  # (B,N,N,N)
        cos_theta = cos_theta.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        # Legendre polynomial basis: (B, N, N, N, L+1)
        ang_feat = legendre_basis(cos_theta, self.legendre_order)  # (B,N,N,N,L+1)

        # Weight pairs (j,k) and aggregate over j and k:
        # w_ij * w_ik: (B,N,N,1) outer → (B,N,N,N)
        w_jk = (w.squeeze(-1).unsqueeze(-1)) * (w.squeeze(-1).unsqueeze(-2))  # (B,N,N,N)

        # Weighted sum over j and k: (B,N,L+1)
        agg = torch.einsum("bijk,bijkl->bil", w_jk, ang_feat)

        # Project to emb_dim and normalise
        out = self.out_norm(self.angular_mlp(agg))  # (B,N,emb_dim)
        return out
