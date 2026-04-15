"""ReacFormer-SE3: Full SE(3) CG equivariant encoder with global attention.

Architecture (per block):
  Path A – Invariant global attention (Uni-Mol style, full N×N)
  Path B – SE(3) CG equivariant message passing (MACE style, ℓ_max=2)
  Path C – Three-body angular invariant aggregation
  Fusion – Bidirectional gated fusion of all three paths

Registered as ``"reacformer_se3"`` in ENCODER_REGISTRY.

Output dict keys (same interface as SingleMolEncoder + node_tensor):
  node_feats   (B, N, D)
  node_vec     (B, N, 3, D)
  node_tensor  (B, N, 5, D)   ← new: ℓ=2 equivariant d-tensor
  graph_feats  (B, D)
  coords_input (B, N, 3)
  atom_padding (B, N)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ._layers import (
    RBFEmb, NonLinear, UniMolLayer, safe_normalization,
    create_access_mask, create_attn_mask,
    ThreeBodyAggregation,
)
from ._layers.e3nn_utils import sh_l1, sh_l2
from ..registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# SE(3) CG message-passing block (Path B)
# Uses e3nn sh_l1/sh_l2 for verified spherical harmonics.
# CG paths: 0⊗1→1 and 0⊗2→2 are trivially scalar*SH (no TensorProduct needed).
# ---------------------------------------------------------------------------

class SE3CGMessageBlock(torch.nn.Module):
    """SE(3)-equivariant message passing using e3nn spherical harmonics.

    CG paths implemented:
        scalar × Y1 → vector      (0 ⊗ 1 → 1)
        scalar × Y2 → d-tensor    (0 ⊗ 2 → 2)
        dot(vector, Y1) → scalar  (1 ⊗ 1 → 0)
    All radial weights learned via MLP(RBF) → scalar gate.
    """

    def __init__(self, emb_dim: int, num_rbf: int, cutoff: float = 5.0) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        def _radial(out_dim: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(num_rbf, emb_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(emb_dim, out_dim),
            )

        self.radial_0   = _radial(emb_dim)
        self.radial_1   = _radial(emb_dim)
        self.radial_2   = _radial(emb_dim)
        self.radial_11_0 = _radial(emb_dim)
        self.src_proj   = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_s      = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim), torch.nn.Linear(emb_dim, emb_dim)
        )
        self.out_v_scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.out_t_scale = torch.nn.Parameter(torch.ones(emb_dim))

    def forward(
        self,
        node_s: Tensor, node_v: Tensor, node_t: Tensor,
        rbf: Tensor, r_hat: Tensor, mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, N, _, D = node_v.shape
        h_j = self.src_proj(node_s).unsqueeze(1).expand(B, N, N, D) * mask.unsqueeze(-1)

        # e3nn-backed SH
        Y1 = sh_l1(r_hat)   # (B,N,N,3)
        Y2 = sh_l2(r_hat)   # (B,N,N,5)

        # scalar message
        msg_s = (h_j * self.radial_0(rbf)).sum(dim=2)

        # vector message: scalar * Y1
        vec_msg  = (h_j * self.radial_1(rbf)).unsqueeze(-2) * Y1.unsqueeze(-1)
        delta_v  = vec_msg.sum(dim=2) * self.out_v_scale

        # tensor message: scalar * Y2
        ten_msg  = (h_j * self.radial_2(rbf)).unsqueeze(-2) * Y2.unsqueeze(-1)
        delta_t  = ten_msg.sum(dim=2) * self.out_t_scale

        # 1⊗1→0: dot(v_j, Y1) → invariant scalar
        v_j = node_v.unsqueeze(2).expand(B, N, N, 3, D)
        dot_vY1 = (v_j * Y1.unsqueeze(-1)).sum(dim=-2)
        msg_s_from_v = (dot_vY1 * self.radial_11_0(rbf) * mask.unsqueeze(-1)).sum(dim=2)

        delta_s = self.out_s(msg_s + msg_s_from_v)
        return delta_s, delta_v, delta_t


# ---------------------------------------------------------------------------
# Gated cross-path fusion
# ---------------------------------------------------------------------------

class CrossPathFusion(torch.nn.Module):
    """Bidirectional gated fusion of invariant and equivariant paths.

    Scalars from all paths modulate the equivariant updates;
    equivariant norms feed back to refine the scalar update.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        # Gates for scalar path
        self.gate_s = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 3, emb_dim),
            torch.nn.Sigmoid(),
        )
        # Gate for vector path (uses scalar info)
        self.gate_v = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.Sigmoid(),
        )
        # Gate for tensor path
        self.gate_t = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.Sigmoid(),
        )
        self.ln_s = torch.nn.LayerNorm(emb_dim)
        # Equivariant features: use scalar scale only (no LayerNorm over D)
        self.scale_v = torch.nn.Parameter(torch.ones(emb_dim))
        self.scale_t = torch.nn.Parameter(torch.ones(emb_dim))

    def forward(
        self,
        h:        Tensor,   # (B,N,D) current scalar
        delta_sa: Tensor,   # (B,N,D) from path A (attention)
        delta_sb: Tensor,   # (B,N,D) from path B (CG)
        delta_sc: Tensor,   # (B,N,D) from path C (3-body)
        delta_v:  Tensor,   # (B,N,3,D) from path B
        delta_t:  Tensor,   # (B,N,5,D) from path B
        node_v:   Tensor,   # (B,N,3,D) current vectors
        node_t:   Tensor,   # (B,N,5,D) current tensors
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Scalar update: gated combination of all three paths
        # Back-inject equivariant norm (invariant) into scalar gate
        v_norm = node_v.norm(dim=-2)  # (B,N,D) – invariant under rotation
        t_norm = node_t.norm(dim=-2)  # (B,N,D)
        gate_input = torch.cat([delta_sa, delta_sb + delta_sc, v_norm + t_norm], dim=-1)
        g_s = self.gate_s(gate_input)                             # (B,N,D)
        h_new = h + self.ln_s(g_s * (delta_sa + delta_sb + delta_sc))

        # Vector update: gated by scalar (equivariant: scalar × vector = vector)
        g_v = self.gate_v(torch.cat([h_new, v_norm], dim=-1)).unsqueeze(-2)  # (B,N,1,D)
        v_new = node_v + g_v * delta_v * self.scale_v  # no LayerNorm on equivariant

        # Tensor update: gated by scalar
        g_t = self.gate_t(torch.cat([h_new, t_norm], dim=-1)).unsqueeze(-2)  # (B,N,1,D)
        t_new = node_t + g_t * delta_t * self.scale_t

        return h_new, v_new, t_new


# ---------------------------------------------------------------------------
# ReacFormer-SE3 Block
# ---------------------------------------------------------------------------

class ReacFormerSE3Block(torch.nn.Module):
    """One ReacFormer-SE3 block: three parallel paths + cross-path fusion."""

    def __init__(
        self,
        emb_dim: int,
        heads: int,
        num_rbf: int,
        cutoff: float = 5.0,
        path_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        legendre_order: int = 3,
    ) -> None:
        super().__init__()
        # Path A: invariant global attention
        self.path_a = UniMolLayer(
            dim=emb_dim, num_heads=heads,
            path_dropout=path_dropout,
            activation_dropout=activation_dropout,
            attn_dropout=attn_dropout,
        )
        # Path B: SE(3) CG message passing (e3nn SH, verified equivariant)
        self.path_b = SE3CGMessageBlock(emb_dim=emb_dim, num_rbf=num_rbf, cutoff=cutoff)
        # Path C: three-body angular features
        self.path_c = ThreeBodyAggregation(
            emb_dim=emb_dim, num_rbf=num_rbf,
            legendre_order=legendre_order, cutoff=cutoff,
        )
        # Edge bias projection for attention
        self.edge_bias_proj = NonLinear(num_rbf, heads, hidden=emb_dim)
        # Fusion
        self.fusion = CrossPathFusion(emb_dim=emb_dim)

    def forward(
        self,
        node_s: Tensor,       # (B,N,D)
        node_v: Tensor,       # (B,N,3,D)
        node_t: Tensor,       # (B,N,5,D)
        rbf:    Tensor,       # (B,N,N,num_rbf)
        r_hat:  Tensor,       # (B,N,N,3)
        access_mask: Tensor,  # (B,N,N) float
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        edge_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (node_s, node_v, node_t, edge_bias)."""
        # Path A: global attention with edge bias
        if edge_bias is None:
            edge_bias = self.edge_bias_proj(rbf)            # (B,N,N,heads)
        node_s_a, edge_bias = self.path_a(
            x=node_s, edge_bias=edge_bias,
            attn_mask=attn_mask, padding_mask=padding_mask,
        )

        # Path B: SE(3) CG message passing
        delta_sb, delta_v, delta_t = self.path_b(
            node_s=node_s, node_v=node_v, node_t=node_t,
            rbf=rbf, r_hat=r_hat, mask=access_mask,
        )

        # Path C: three-body angular features
        delta_sc = self.path_c(rbf=rbf, r_hat=r_hat, access_mask=access_mask)

        # Cross-path gated fusion
        node_s, node_v, node_t = self.fusion(
            h=node_s,
            delta_sa=node_s_a - node_s,  # delta from attention
            delta_sb=delta_sb,
            delta_sc=delta_sc,
            delta_v=delta_v,
            delta_t=delta_t,
            node_v=node_v,
            node_t=node_t,
        )
        return node_s, node_v, node_t, edge_bias


# ---------------------------------------------------------------------------
# ReacFormerSE3 Encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("reacformer_se3")
class ReacFormerSE3Encoder(torch.nn.Module):
    """ReacFormer with full SE(3) CG equivariance (ℓ_max=2).

    Three parallel paths per block:
      A – Invariant global attention (full N×N, no cutoff)
      B – SE(3) CG equivariant message passing (ℓ=0,1,2)
      C – Three-body Legendre angular features
    Fused by bidirectional gated coupling.

    Output dict keys:
      node_feats   (B, N, D)
      node_vec     (B, N, 3, D)
      node_tensor  (B, N, 5, D)   ℓ=2 equivariant d-tensor
      graph_feats  (B, D)
      coords_input (B, N, 3)
      atom_padding (B, N)

    Config example::

        encoder:
          type: reacformer_se3
          emb_dim: 256
          num_layers: 6
          heads: 8
          num_rbf: 128
          cutoff: 5.0
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_layers: int = 6,
        heads: int = 8,
        atom_vocab_size: int = 128,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        path_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        legendre_order: int = 3,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.cutoff  = cutoff

        self.atom_encoder = torch.nn.Embedding(atom_vocab_size, emb_dim)
        self.bond_encoder = RBFEmb(num_rbf, cutoff)
        self.dist_to_rbf_proj = torch.nn.Linear(num_rbf, num_rbf)  # refine RBF

        self.blocks = torch.nn.ModuleList([
            ReacFormerSE3Block(
                emb_dim=emb_dim, heads=heads, num_rbf=num_rbf,
                cutoff=cutoff,
                path_dropout=path_dropout,
                attn_dropout=attn_dropout,
                activation_dropout=activation_dropout,
                legendre_order=legendre_order,
            )
            for _ in range(num_layers)
        ])

        # Initial edge bias projection
        self.edge_bias_init = NonLinear(num_rbf, heads, hidden=emb_dim)

    @staticmethod
    def _masked_mean_pool(node_feats: Tensor, atom_padding: Tensor) -> Tensor:
        valid = (~atom_padding).unsqueeze(-1).float()
        return (node_feats * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def forward(
        self,
        input_atomic_numbers: Tensor,
        coords_noisy: Tensor,
        atom_padding: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        B, N = input_atomic_numbers.shape
        if atom_padding is None:
            atom_padding = torch.zeros(B, N, dtype=torch.bool, device=coords_noisy.device)

        # Node init
        node_s = self.atom_encoder(input_atomic_numbers)  # (B,N,D)
        node_s = node_s.masked_fill(atom_padding.unsqueeze(-1), 0)
        node_v = torch.zeros(B, N, 3, self.emb_dim, device=coords_noisy.device, dtype=node_s.dtype)
        node_t = torch.zeros(B, N, 5, self.emb_dim, device=coords_noisy.device, dtype=node_s.dtype)

        # Pairwise geometry
        ediff = coords_noisy.unsqueeze(2) - coords_noisy.unsqueeze(1)  # (B,N,N,3)
        dist  = ediff.norm(dim=-1).clamp_min(1e-6)                      # (B,N,N)
        r_hat = ediff / dist.unsqueeze(-1)                              # (B,N,N,3)

        # RBF + masks
        rbf = self.bond_encoder(dist)                              # (B,N,N,num_rbf)
        rbf = self.dist_to_rbf_proj(rbf)
        pad_2d = atom_padding.unsqueeze(2) | atom_padding.unsqueeze(1) | (dist >= self.cutoff)
        rbf    = rbf.masked_fill(pad_2d.unsqueeze(-1), 0)
        r_hat  = r_hat.masked_fill(pad_2d.unsqueeze(-1), 0)

        access_mask   = create_access_mask(dist, self.cutoff, atom_padding).float()
        attn_mask     = create_attn_mask(dist, self.cutoff, atom_padding)

        # Initial edge bias
        edge_bias = self.edge_bias_init(rbf)  # (B,N,N,heads)

        for block in self.blocks:
            node_s, node_v, node_t, edge_bias = block(
                node_s=node_s, node_v=node_v, node_t=node_t,
                rbf=rbf, r_hat=r_hat, access_mask=access_mask,
                attn_mask=attn_mask, padding_mask=atom_padding,
                edge_bias=edge_bias,
            )
            node_s = node_s.masked_fill(atom_padding.unsqueeze(-1), 0)
            node_v = node_v.masked_fill(atom_padding.unsqueeze(-1).unsqueeze(-1), 0)
            node_t = node_t.masked_fill(atom_padding.unsqueeze(-1).unsqueeze(-1), 0)

        graph_feats = self._masked_mean_pool(node_s, atom_padding)
        return {
            "node_feats":   node_s,
            "node_vec":     node_v,
            "node_tensor":  node_t,
            "graph_feats":  graph_feats,
            "coords_input": coords_noisy,
            "atom_padding": atom_padding,
        }
