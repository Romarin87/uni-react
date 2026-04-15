"""ReacFormer-SO2: SE(3) via SO(2) decomposition (eSCN style) + global attention.

Same three-path architecture as ReacFormer-SE3 but replaces the full CG
tensor products with the SO(2) decomposition of Passaro & Zitnick (ICML 2023).

Key difference vs SE3 version:
  - Rotates each edge to local frame (bond axis = z)
  - In local frame, SO(3) equivariance reduces to SO(2)
  - Different |m| channels decouple → O(L³) vs O(L⁶)
  - Numerically identical equivariance, 2-4× faster for ℓ_max=2

Registered as ``"reacformer_so2"`` in ENCODER_REGISTRY.

Output dict: same interface as ReacFormerSE3Encoder.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ._layers import (
    RBFEmb, NonLinear, UniMolLayer, safe_normalization,
    create_access_mask, create_attn_mask,
    ThreeBodyAggregation,
    SO2MessageBlock,
)
from ..registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# Gated cross-path fusion (reused from SE3 version via copy to avoid import)
# ---------------------------------------------------------------------------

class CrossPathFusion(torch.nn.Module):
    """Bidirectional gated fusion – identical to the SE3 version."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.gate_s = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 3, emb_dim), torch.nn.Sigmoid(),
        )
        self.gate_v = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim), torch.nn.Sigmoid(),
        )
        self.gate_t = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim), torch.nn.Sigmoid(),
        )
        self.ln_s    = torch.nn.LayerNorm(emb_dim)
        self.scale_v = torch.nn.Parameter(torch.ones(emb_dim))
        self.scale_t = torch.nn.Parameter(torch.ones(emb_dim))

    def forward(
        self,
        h: Tensor, delta_sa: Tensor, delta_sb: Tensor, delta_sc: Tensor,
        delta_v: Tensor, delta_t: Optional[Tensor],
        node_v: Tensor, node_t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        v_norm = node_v.norm(dim=-2)  # (B,N,D) – invariant
        t_norm = node_t.norm(dim=-2)
        g_s = self.gate_s(torch.cat([delta_sa, delta_sb + delta_sc, v_norm + t_norm], dim=-1))
        h_new = h + self.ln_s(g_s * (delta_sa + delta_sb + delta_sc))

        g_v = self.gate_v(torch.cat([h_new, v_norm], dim=-1)).unsqueeze(-2)
        v_new = node_v + g_v * delta_v * self.scale_v

        g_t = self.gate_t(torch.cat([h_new, t_norm], dim=-1)).unsqueeze(-2)
        if delta_t is not None:
            t_new = node_t + g_t * delta_t * self.scale_t
        else:
            t_new = node_t
        return h_new, v_new, t_new


# ---------------------------------------------------------------------------
# ReacFormer-SO2 Block
# ---------------------------------------------------------------------------

class ReacFormerSO2Block(torch.nn.Module):
    """One ReacFormer-SO2 block.

    Path A: invariant global attention (same as SE3 version)
    Path B: SO(2) equivariant message passing (eSCN style, faster)
    Path C: three-body Legendre angular features
    Fusion: bidirectional gated coupling
    """

    def __init__(
        self,
        emb_dim: int,
        heads: int,
        num_rbf: int,
        cutoff: float = 5.0,
        l_max: int = 2,
        path_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        legendre_order: int = 3,
    ) -> None:
        super().__init__()
        self.l_max = l_max
        # Path A
        self.path_a = UniMolLayer(
            dim=emb_dim, num_heads=heads,
            path_dropout=path_dropout,
            activation_dropout=activation_dropout,
            attn_dropout=attn_dropout,
        )
        # Path B: SO(2) equivariant message passing
        self.path_b = SO2MessageBlock(
            emb_dim=emb_dim, num_rbf=num_rbf, l_max=l_max, cutoff=cutoff,
        )
        # Path C: three-body
        self.path_c = ThreeBodyAggregation(
            emb_dim=emb_dim, num_rbf=num_rbf,
            legendre_order=legendre_order, cutoff=cutoff,
        )
        # Projections
        self.edge_bias_proj = NonLinear(num_rbf, heads, hidden=emb_dim)
        self.fusion = CrossPathFusion(emb_dim=emb_dim)

    def forward(
        self,
        node_s: Tensor,
        node_v: Tensor,
        node_t: Tensor,
        rbf:    Tensor,
        r_hat:  Tensor,
        access_mask: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        edge_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Path A
        if edge_bias is None:
            edge_bias = self.edge_bias_proj(rbf)
        node_s_a, edge_bias = self.path_a(
            x=node_s, edge_bias=edge_bias,
            attn_mask=attn_mask, padding_mask=padding_mask,
        )

        # Path B: SO(2) equivariant
        node_t_in = node_t if self.l_max >= 2 else None
        delta_sb, delta_v, delta_t = self.path_b(
            node_scalar=node_s,
            node_vec=node_v,
            node_tensor=node_t_in,
            rbf=rbf, r_hat=r_hat, access_mask=access_mask,
        )

        # Path C: three-body
        delta_sc = self.path_c(rbf=rbf, r_hat=r_hat, access_mask=access_mask)

        # Fusion
        node_s, node_v, node_t = self.fusion(
            h=node_s,
            delta_sa=node_s_a - node_s,
            delta_sb=delta_sb,
            delta_sc=delta_sc,
            delta_v=delta_v,
            delta_t=delta_t,
            node_v=node_v,
            node_t=node_t,
        )
        return node_s, node_v, node_t, edge_bias


# ---------------------------------------------------------------------------
# ReacFormerSO2 Encoder
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("reacformer_so2")
class ReacFormerSO2Encoder(torch.nn.Module):
    """ReacFormer with SO(2)-decomposed SE(3) equivariance (eSCN style).

    Mathematically equivalent to full CG SE(3) equivariance but 2-4× faster
    for ℓ_max=2 because the SO(2) decomposition decouples |m| channels.

    Three parallel paths per block:
      A – Invariant global attention
      B – SO(2)-equivariant message passing (ℓ_max configurable)
      C – Three-body Legendre angular features

    Config example::

        encoder:
          type: reacformer_so2
          emb_dim: 256
          num_layers: 6
          heads: 8
          num_rbf: 128
          l_max: 2
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
        l_max: int = 2,
        path_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        legendre_order: int = 3,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.cutoff  = cutoff
        self.l_max   = l_max

        self.atom_encoder    = torch.nn.Embedding(atom_vocab_size, emb_dim)
        self.bond_encoder    = RBFEmb(num_rbf, cutoff)
        self.dist_to_rbf_proj = torch.nn.Linear(num_rbf, num_rbf)
        self.edge_bias_init  = NonLinear(num_rbf, heads, hidden=emb_dim)

        self.blocks = torch.nn.ModuleList([
            ReacFormerSO2Block(
                emb_dim=emb_dim, heads=heads, num_rbf=num_rbf,
                cutoff=cutoff, l_max=l_max,
                path_dropout=path_dropout,
                attn_dropout=attn_dropout,
                activation_dropout=activation_dropout,
                legendre_order=legendre_order,
            )
            for _ in range(num_layers)
        ])

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

        node_s = self.atom_encoder(input_atomic_numbers)
        node_s = node_s.masked_fill(atom_padding.unsqueeze(-1), 0)
        node_v = torch.zeros(B, N, 3, self.emb_dim, device=coords_noisy.device, dtype=node_s.dtype)
        node_t = torch.zeros(B, N, 5, self.emb_dim, device=coords_noisy.device, dtype=node_s.dtype)

        ediff = coords_noisy.unsqueeze(2) - coords_noisy.unsqueeze(1)
        dist  = ediff.norm(dim=-1).clamp_min(1e-6)
        r_hat = ediff / dist.unsqueeze(-1)

        rbf     = self.bond_encoder(dist)
        rbf     = self.dist_to_rbf_proj(rbf)
        pad_2d  = atom_padding.unsqueeze(2) | atom_padding.unsqueeze(1) | (dist >= self.cutoff)
        rbf     = rbf.masked_fill(pad_2d.unsqueeze(-1), 0)
        r_hat   = r_hat.masked_fill(pad_2d.unsqueeze(-1), 0)

        access_mask = create_access_mask(dist, self.cutoff, atom_padding).float()
        attn_mask   = create_attn_mask(dist, self.cutoff, atom_padding)
        edge_bias   = self.edge_bias_init(rbf)

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
