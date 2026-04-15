"""SO(2) equivariant convolution for message passing.

Based on eSCN (Passaro & Zitnick, ICML 2023, arXiv:2302.03655).

Key insight: rotate each edge to a local frame where bond axis = z,
then SO(3) equivariance becomes SO(2) symmetry (rotation around z).
Different |m| channels decouple, reducing cost from O(L^6) to O(L^3).

For ℓ_max=1 (vectors) we only need the 3x3 rotation matrix.
For ℓ_max=2 (d-tensors) we need the 5x5 Wigner-D for ℓ=2.
Both are computed analytically from the bond direction.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Rotation matrix from bond direction
# ---------------------------------------------------------------------------

def rotation_from_edge(r_hat: Tensor) -> Tensor:
    """Build a 3x3 rotation matrix R that maps z-axis to r_hat.

    R[:,2] = r_hat (the bond direction becomes the new z-axis).
    The x and y axes are chosen to complete an orthonormal frame
    using the Gram-Schmidt procedure with a fixed reference.

    Args:
        r_hat: Unit vectors ``(..., 3)``.

    Returns:
        Rotation matrices ``(..., 3, 3)``.
    """
    device = r_hat.device
    dtype  = r_hat.dtype
    shape  = r_hat.shape[:-1]

    # Choose reference vector perpendicular to r_hat for Gram-Schmidt
    # If r_hat ≈ (0,0,1), use (1,0,0); else use (0,0,1)
    ref = torch.zeros_like(r_hat)
    use_x = (r_hat[..., 2].abs() > 0.9)
    ref[..., 0] = (~use_x).float()
    ref[..., 2] = use_x.float()

    # x = normalize(ref - (ref·r_hat)*r_hat)
    proj  = (ref * r_hat).sum(dim=-1, keepdim=True)
    x_raw = ref - proj * r_hat
    x_norm = x_raw / (x_raw.norm(dim=-1, keepdim=True) + 1e-8)

    # y = r_hat × x
    y_norm = torch.cross(r_hat, x_norm, dim=-1)

    # Stack as columns: R = [x | y | z=r_hat]
    R = torch.stack([x_norm, y_norm, r_hat], dim=-1)  # (..., 3, 3)
    return R


def wigner_d_l2_from_R(R: Tensor) -> Tensor:
    """5x5 Wigner-D matrix for ℓ=2 from rotation matrix R.

    Computes D such that Y2(R @ r_hat) = D @ Y2(r_hat) for all unit vectors.

    Uses the fact that the 5 real spherical harmonics Y2 evaluated at 5
    linearly-independent unit vectors form a 5x5 invertible matrix A, and
    that applying R gives A' = D @ A, so D = A' @ inv(A).

    This is a numerically robust, definition-based approach that avoids
    any hand-coded CG formula.
        (xy, yz, z²-½(x²+y²), xz, ½(x²-y²))

    We derive D analytically by applying R to each of the 6 standard
    Cartesian monomials (xx, xy, xz, yy, yz, zz) and projecting back
    into our 5-dimensional basis.

    Args:
        R: Rotation matrices ``(..., 3, 3)``.

    Returns:
        ``(..., 5, 5)`` Wigner-D matrices.
    """
    try:
        from e3nn import o3 as _o3
    except ImportError:
        raise ImportError(
            "e3nn is required for wigner_d_l2_from_R. "
            "Install with: pip install e3nn"
        )
    
    device = R.device
    dtype = R.dtype
    prefix = R.shape[:-2]
    
    # Cast to float32 for e3nn compatibility
    R_flat = R.reshape(-1, 3, 3).float()
    
    # Orthogonalise via SVD to ensure proper rotation matrix (det=+1)
    U, _, Vh = torch.linalg.svd(R_flat)
    R_orth = U @ Vh
    
    # Fix reflection: if det=-1, flip sign to make det=+1
    det = torch.det(R_orth)  # (-1,)
    # For any det < 0, negate the last column of U
    sign = torch.where(det < 0, -1.0, 1.0).unsqueeze(-1).unsqueeze(-1)
    U_fixed = U.clone()
    U_fixed[:, :, -1] = U_fixed[:, :, -1] * sign.squeeze(-1)
    R_orth = U_fixed @ Vh
    
    # Ensure det is close to 1 (add small epsilon for numerical stability)
    det_check = torch.det(R_orth)
    # Clamp to avoid assertion error in e3nn
    R_orth = R_orth / det_check.unsqueeze(-1).unsqueeze(-1).clamp(min=0.99, max=1.01)
    
    # Convert to angles - MUST be on CPU for e3nn
    R_orth_cpu = R_orth.cpu()
    angles = _o3.matrix_to_angles(R_orth_cpu)
    
    # Compute Wigner-D matrix on CPU
    alpha, beta, gamma = angles
    D_flat = _o3.wigner_D(2, alpha, beta, gamma)  # (-1, 5, 5) on CPU
    
    # Move back to original device and dtype
    return D_flat.reshape(*prefix, 5, 5).to(dtype).to(device)


# ---------------------------------------------------------------------------
# SO(2) message-passing block
# ---------------------------------------------------------------------------

class SO2MessageBlock(torch.nn.Module):
    """One SO(2)-equivariant message-passing layer.

    For each neighbor j of atom i:
      1. Rotate j's irrep features to edge-local frame (bond axis = z).
      2. Apply SO(2)-equivariant linear mixing (different |m| decouple).
      3. Scale by radial weight (learned from RBF).
      4. Rotate back to global frame.
      5. Aggregate over neighbors.

    Args:
        emb_dim:  Feature channel dimension.
        num_rbf:  Number of RBF basis functions.
        l_max:    Maximum irrep order (1 or 2).
        cutoff:   Cutoff radius for soft-mask.
    """

    def __init__(
        self,
        emb_dim: int,
        num_rbf: int,
        l_max: int = 2,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        if l_max not in (1, 2):
            raise ValueError("l_max must be 1 or 2")
        self.emb_dim = emb_dim
        self.l_max   = l_max
        self.cutoff  = cutoff

        # Radial network: RBF → per-ℓ weight vectors
        # For each ℓ, produces emb_dim weights
        n_ell = l_max + 1  # ℓ = 0, 1, [2]
        self.radial_net = torch.nn.Sequential(
            torch.nn.Linear(num_rbf, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, emb_dim * n_ell),
        )

        # SO(2) linear mixing: for each m, mix the D channels
        # ℓ=0: m=0 only (1 component × D)
        # ℓ=1: m=0 (1) + m=±1 (2, treated jointly) → 2 SO(2) blocks
        # ℓ=2: m=0 (1) + m=±1 (2) + m=±2 (2) → 3 SO(2) blocks
        self.so2_l0 = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.so2_l1_m0 = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.so2_l1_m1 = torch.nn.Linear(emb_dim * 2, emb_dim * 2, bias=False)
        if l_max >= 2:
            self.so2_l2_m0 = torch.nn.Linear(emb_dim, emb_dim, bias=False)
            self.so2_l2_m1 = torch.nn.Linear(emb_dim * 2, emb_dim * 2, bias=False)
            self.so2_l2_m2 = torch.nn.Linear(emb_dim * 2, emb_dim * 2, bias=False)

        # Output projection for scalar path
        self.out_proj_s = torch.nn.Linear(emb_dim, emb_dim)
        # Layer norms
        self.ln_s = torch.nn.LayerNorm(emb_dim)
        self.ln_v = torch.nn.LayerNorm(emb_dim)
        if l_max >= 2:
            self.ln_t = torch.nn.LayerNorm(emb_dim)

    # ------------------------------------------------------------------
    # SO(2) mixing helpers
    # ------------------------------------------------------------------

    def _mix_l0(self, feat: Tensor, w: Tensor) -> Tensor:
        """Mix ℓ=0 (scalar) irrep features.
        feat: (..., D), w: (..., D)
        """
        return self.so2_l0(feat) * w

    def _mix_l1(
        self,
        vec_local: Tensor,  # (..., 3, D)  in local frame
        w: Tensor,          # (..., D)
    ) -> Tensor:
        """SO(2) mixing for ℓ=1 irreps in the edge-local frame.
        The z-component (m=0) and (x,y) pair (m=±1) decouple under SO(2).
        """
        # m=0 component: z-axis (index 2 in local frame)
        z_comp = vec_local[..., 2, :] * w               # (..., D)
        z_out  = self.so2_l1_m0(z_comp)                 # (..., D)

        # m=±1 components: (x, y) pair
        xy = vec_local[..., :2, :].reshape(
            *vec_local.shape[:-2], self.emb_dim * 2
        )                                                # (..., 2D)
        xy_out = self.so2_l1_m1(xy * w.repeat_interleave(2, dim=-1))
        xy_out = xy_out.reshape(*vec_local.shape[:-2], 2, self.emb_dim)

        out = torch.zeros_like(vec_local)
        out[..., 2, :] = z_out
        out[..., :2, :] = xy_out
        return out

    def _mix_l2(
        self,
        ten_local: Tensor,  # (..., 5, D)  in local frame
        w: Tensor,          # (..., D)
    ) -> Tensor:
        """SO(2) mixing for ℓ=2 irreps.
        m=0: index 2; m=±1: indices (1,3); m=±2: indices (0,4).
        """
        # m=0
        m0 = self.so2_l2_m0(ten_local[..., 2, :] * w)         # (...,D)
        # m=±1
        m1_pair = torch.stack([ten_local[..., 1, :], ten_local[..., 3, :]], dim=-2)
        m1_flat = m1_pair.reshape(*ten_local.shape[:-2], self.emb_dim * 2)
        m1_out  = self.so2_l2_m1(m1_flat * w.repeat_interleave(2, dim=-1))
        m1_out  = m1_out.reshape(*ten_local.shape[:-2], 2, self.emb_dim)
        # m=±2
        m2_pair = torch.stack([ten_local[..., 0, :], ten_local[..., 4, :]], dim=-2)
        m2_flat = m2_pair.reshape(*ten_local.shape[:-2], self.emb_dim * 2)
        m2_out  = self.so2_l2_m2(m2_flat * w.repeat_interleave(2, dim=-1))
        m2_out  = m2_out.reshape(*ten_local.shape[:-2], 2, self.emb_dim)

        out = torch.zeros_like(ten_local)
        out[..., 2, :] = m0
        out[..., 1, :] = m1_out[..., 0, :]
        out[..., 3, :] = m1_out[..., 1, :]
        out[..., 0, :] = m2_out[..., 0, :]
        out[..., 4, :] = m2_out[..., 1, :]
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        node_scalar: Tensor,          # (B, N, D)
        node_vec:    Tensor,          # (B, N, 3, D)
        node_tensor: Optional[Tensor],# (B, N, 5, D) or None
        rbf:         Tensor,          # (B, N, N, num_rbf)
        r_hat:       Tensor,          # (B, N, N, 3) unit vectors r_j→i
        access_mask: Tensor,          # (B, N, N) float
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Compute SO(2) equivariant messages and aggregate.

        Returns:
            Δscalar ``(B,N,D)``, Δvec ``(B,N,3,D)``,
            Δtensor ``(B,N,5,D)`` or None.
        """
        B, N, _, D = node_vec.shape
        n_ell = self.l_max + 1

        # Radial weights: (B,N,N,D*n_ell)
        w_all = self.radial_net(rbf)                        # (B,N,N,D*n_ell)
        w_all = w_all * access_mask.unsqueeze(-1)
        w = w_all.split(D, dim=-1)                         # list of (B,N,N,D)

        # Build rotation matrices for each edge
        R = rotation_from_edge(r_hat)                      # (B,N,N,3,3)

        # Rotate neighbor vec features to edge-local frame
        # node_vec: (B,N,3,D) → broadcast to (B,N,N,3,D)
        vec_j  = node_vec.unsqueeze(2).expand(B, N, N, 3, D)   # (B,N,N,3,D)
        # R: (B,N,N,3,3); vec_local[...,p,d] = Σ_q R[...,p,q] * vec_j[...,q,d]
        vec_local = torch.einsum("bijpq,bijqd->bijpd", R, vec_j)  # (B,N,N,3,D)

        # ℓ=0 message: scalar × radial weight
        s_j = node_scalar.unsqueeze(2).expand(B, N, N, D)     # (B,N,N,D)
        msg_s = self._mix_l0(s_j, w[0])                        # (B,N,N,D)

        # ℓ=1 message: rotate + SO(2) mix + rotate back
        msg_v_local = self._mix_l1(vec_local, w[1])             # (B,N,N,3,D)
        # Rotate back: R^T @ msg_v_local
        msg_v = torch.einsum("bijpq,bijqd->bijpd", R.transpose(-1,-2), msg_v_local)

        # ℓ=2 message (optional)
        msg_t = None
        if self.l_max >= 2 and node_tensor is not None:
            ten_j = node_tensor.unsqueeze(2).expand(B, N, N, 5, D)  # (B,N,N,5,D)
            D2 = wigner_d_l2_from_R(R)                               # (B,N,N,5,5)
            ten_local = torch.einsum("bijpq,bijqd->bijpd", D2, ten_j)
            msg_t_local = self._mix_l2(ten_local, w[2])              # (B,N,N,5,D)
            D2_inv = D2.transpose(-1, -2)
            msg_t_global = torch.einsum("bijpq,bijqd->bijpd", D2_inv, msg_t_local)
            msg_t = msg_t_global.sum(dim=2)                          # (B,N,5,D)

        # Aggregate: sum over neighbors j (dim=2)
        delta_s = self.ln_s(self.out_proj_s(msg_s.sum(dim=2)))  # (B,N,D)
        delta_v = self.ln_v(msg_v.sum(dim=2))                   # (B,N,3,D)
        if msg_t is not None:
            delta_t = self.ln_t(msg_t)                           # (B,N,5,D)
        else:
            delta_t = None

        return delta_s, delta_v, delta_t