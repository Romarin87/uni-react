"""Thin e3nn wrappers used by ReacFormer encoders.

All equivariant operations (spherical harmonics, Wigner-D, tensor products)
delegate to e3nn so that correctness is guaranteed by a well-tested library.
"""
from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor

try:
    from e3nn import o3
    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False


def _require_e3nn():
    if not HAS_E3NN:
        raise ImportError(
            "e3nn is required for ReacFormer equivariant layers. "
            "Install with: pip install e3nn"
        )


def sh_l1(r_hat: Tensor) -> Tensor:
    """Real spherical harmonics ℓ=1 (m=-1,0,+1) for unit vectors.

    Delegates to ``e3nn.o3.spherical_harmonics`` for a consistent basis
    that matches :func:`wigner_d_l1`.

    Args:
        r_hat: Unit vectors ``(..., 3)``.

    Returns:
        ``(..., 3)``.
    """
    _require_e3nn()
    shape = r_hat.shape
    return o3.spherical_harmonics(1, r_hat.reshape(-1, 3),
                                   normalize=True).reshape(*shape[:-1], 3)


def sh_l2(r_hat: Tensor) -> Tensor:
    """Real spherical harmonics ℓ=2 (m=-2,-1,0,+1,+2) for unit vectors.

    Args:
        r_hat: Unit vectors ``(..., 3)``.

    Returns:
        ``(..., 5)``.
    """
    _require_e3nn()
    shape = r_hat.shape
    return o3.spherical_harmonics(2, r_hat.reshape(-1, 3),
                                   normalize=True).reshape(*shape[:-1], 5)


def wigner_d2(R: Tensor) -> Tensor:
    """5×5 Wigner-D matrix for ℓ=2.

    Args:
        R: Rotation matrices ``(..., 3, 3)``.

    Returns:
        ``(..., 5, 5)``.
    """
    _require_e3nn()
    prefix = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3).float()
    U, _, Vh = torch.linalg.svd(R_flat)
    det = torch.det(U @ Vh).unsqueeze(-1).unsqueeze(-1)
    R_orth = U @ (Vh * det)
    angles = o3.matrix_to_angles(R_orth)
    D = o3.wigner_D(2, *angles)  # (-1, 5, 5)
    return D.reshape(*prefix, 5, 5).to(R.dtype)


class CGMessageBlock(torch.nn.Module):
    """SE(3)-equivariant message-passing block using e3nn TensorProduct.

    Computes messages for ℓ=0 (scalar), ℓ=1 (vector), and ℓ=2 (d-tensor)
    using verified e3nn tensor products.  Node features are represented as
    e3nn irreps with multiplicity = emb_dim:

        scalars : ``emb_dim x 0e``  → invariant
        vectors : ``emb_dim x 1o``  → pseudo-vector (odd parity)
        tensors : ``emb_dim x 2e``  → even rank-2 tensor

    The tensor product path used is ``0e ⊗ Lx → Lx`` (scalar × SH → irrep),
    which is always valid and guarantees equivariance.

    Args:
        emb_dim:  Number of channels per irrep (= multiplicity).
        num_rbf:  Number of RBF features for radial weighting.
        l_max:    Maximum irrep order (1 or 2).
    """

    def __init__(self, emb_dim: int, num_rbf: int, l_max: int = 2) -> None:
        super().__init__()
        _require_e3nn()
        self.emb_dim = emb_dim
        self.l_max   = l_max

        # e3nn irreps strings
        self.irreps_s = o3.Irreps(f"{emb_dim}x0e")  # scalars
        self.irreps_v = o3.Irreps(f"{emb_dim}x1o")  # vectors
        self.irreps_t = o3.Irreps(f"{emb_dim}x2e")  # d-tensors
        self.irreps_sh1 = o3.Irreps("1x1o")
        self.irreps_sh2 = o3.Irreps("1x2e")

        # Radial networks: RBF → per-TP weight vector
        def _radial(out_dim: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(num_rbf, emb_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(emb_dim, out_dim),
            )

        # TP: (emb_dim x 0e) ⊗ (1x1o) → (emb_dim x 1o)
        # instruction: "uuw" = per-channel weight, no weight sharing
        if l_max >= 1:
            self.tp_s_sh1 = o3.TensorProduct(
                self.irreps_s, self.irreps_sh1, self.irreps_v,
                instructions=[(i, 0, i, "uuw", True)
                               for i in range(emb_dim)],
                shared_weights=False,
            )
            self.radial_v = _radial(self.tp_s_sh1.weight_numel)

        if l_max >= 2:
            # TP: (emb_dim x 0e) ⊗ (1x2e) → (emb_dim x 2e)
            self.tp_s_sh2 = o3.TensorProduct(
                self.irreps_s, self.irreps_sh2, self.irreps_t,
                instructions=[(i, 0, i, "uuw", True)
                               for i in range(emb_dim)],
                shared_weights=False,
            )
            self.radial_t = _radial(self.tp_s_sh2.weight_numel)

        # Scalar message path: simple radial weighting
        self.radial_s = _radial(emb_dim)
        # 1⊗1→0 inner product for scalar update from vectors
        self.radial_11_0 = _radial(emb_dim)

        self.out_s      = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim), torch.nn.Linear(emb_dim, emb_dim)
        )
        self.out_v_scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.out_t_scale = torch.nn.Parameter(torch.ones(emb_dim))

    def forward(
        self,
        node_s: Tensor,  # (B,N,D)
        node_v: Tensor,  # (B,N,3,D)
        node_t: Tensor,  # (B,N,5,D)
        rbf:    Tensor,  # (B,N,N,num_rbf)
        r_hat:  Tensor,  # (B,N,N,3)
        mask:   Tensor,  # (B,N,N) float
    ):
        B, N, _, D = node_v.shape

        # Source scalars projected to message space
        h_j = node_s.unsqueeze(1).expand(B, N, N, D) * mask.unsqueeze(-1)  # (B,N,N,D)

        # --- Scalar message: h_j * w0 ---
        w0     = self.radial_s(rbf) * mask.unsqueeze(-1)   # (B,N,N,D)
        msg_s  = (h_j * w0).sum(dim=2)                      # (B,N,D)

        # --- 1⊗1→0: v_j · Y1 → scalar update ---
        Y1     = sh_l1(r_hat)                               # (B,N,N,3)
        v_j    = node_v.unsqueeze(2).expand(B, N, N, 3, D)
        dot_vY = (v_j * Y1.unsqueeze(-1)).sum(dim=-2)       # (B,N,N,D) invariant
        w110   = self.radial_11_0(rbf) * mask.unsqueeze(-1)
        msg_s_from_v = (dot_vY * w110).sum(dim=2)           # (B,N,D)

        # --- Vector message: h_j ⊗ Y1 via e3nn TP ---
        delta_v = torch.zeros(B, N, 3, D, device=rbf.device, dtype=rbf.dtype)
        if self.l_max >= 1:
            Y1_norm = sh_l1(r_hat)                          # (B,N,N,3)
            wv = self.radial_v(rbf)                         # (B,N,N, weight_numel)
            # Process each source atom j: flatten (B,N,N) batch
            h_flat  = h_j.reshape(B * N * N, D)             # (BNN, D)
            Y1_flat = Y1_norm.reshape(B * N * N, 3)         # (BNN, 3)
            wv_flat = wv.reshape(B * N * N, -1)             # (BNN, W)
            mask_flat = mask.reshape(B * N * N, 1)
            # e3nn TP expects (batch, irreps_dim): scalars=(D,), Y1=(3,)
            out_v_flat = self.tp_s_sh1(h_flat, Y1_flat, wv_flat)  # (BNN, D*3)
            # Reshape to (BNN, 3, D) and sum over j
            out_v = out_v_flat.reshape(B, N, N, 3, D) * mask_flat.reshape(B, N, N, 1, 1)
            delta_v = out_v.sum(dim=2)                       # (B,N,3,D)

        # --- Tensor message: h_j ⊗ Y2 via e3nn TP ---
        delta_t = torch.zeros(B, N, 5, D, device=rbf.device, dtype=rbf.dtype)
        if self.l_max >= 2:
            Y2_norm = sh_l2(r_hat)                          # (B,N,N,5)
            wt = self.radial_t(rbf)                         # (B,N,N, weight_numel)
            h_flat  = h_j.reshape(B * N * N, D)
            Y2_flat = Y2_norm.reshape(B * N * N, 5)
            wt_flat = wt.reshape(B * N * N, -1)
            mask_flat = mask.reshape(B * N * N, 1)
            out_t_flat = self.tp_s_sh2(h_flat, Y2_flat, wt_flat)  # (BNN, D*5)
            out_t = out_t_flat.reshape(B, N, N, 5, D) * mask_flat.reshape(B, N, N, 1, 1)
            delta_t = out_t.sum(dim=2)                       # (B,N,5,D)

        delta_s = self.out_s(msg_s + msg_s_from_v)
        delta_v = delta_v * self.out_v_scale
        delta_t = delta_t * self.out_t_scale
        return delta_s, delta_v, delta_t