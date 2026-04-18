"""Hybrid ReacFormer encoder with explicit triplet updates.

This encoder keeps the existing ReacFormer-SE3 backbone logic but adds two
practical ingredients motivated by recent QM9 architecture work:

1. Explicit triplet/line-graph style edge updates that aggregate angular
   neighbour context before feeding it back to node states.
2. A lightweight scalar-only global-context block that periodically injects
   graph-level information without turning the whole backbone into a heavy
   global transformer.

The implementation is intentionally conservative: it reuses the existing
SE(3) message-passing path and only augments the scalar path. This keeps the
model compatible with the current training code while making the local
geometry branch significantly stronger.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ._layers import (
    DropPath,
    NonLinear,
    RBFEmb,
    UniMolLayer,
    create_access_mask,
    create_attn_mask,
)
from ._layers.three_body import legendre_basis
from .reacformer_se3 import CrossPathFusion, SE3CGMessageBlock
from ..registry import ENCODER_REGISTRY


class TripletEdgeAggregation(torch.nn.Module):
    """Explicit triplet-aware edge update with node feedback.

    For every center atom ``i`` and neighbour edge ``i -> j`` this layer
    aggregates over the companion neighbours ``i -> k`` and encodes
    ``angle(j, i, k)`` into an edge state. The aggregated edge states are then
    summed back into node updates and pooled into a bond-graph summary.
    """

    def __init__(
        self,
        emb_dim: int,
        num_rbf: int,
        legendre_order: int = 3,
        hidden_dim: Optional[int] = None,
        topk: int = 16,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or emb_dim
        self.legendre_order = int(legendre_order)
        self.topk = max(1, int(topk))
        ang_dim = self.legendre_order + 1

        self.edge_proj = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2 + num_rbf, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, emb_dim),
        )
        self.triplet_proj = torch.nn.Sequential(
            torch.nn.Linear(ang_dim + num_rbf, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, emb_dim),
        )
        self.triplet_score = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.triplet_mix = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim * 2),
            torch.nn.Linear(emb_dim * 2, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, emb_dim),
        )
        self.edge_gate = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.Sigmoid(),
        )
        self.node_out = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.edge_norm = torch.nn.LayerNorm(emb_dim)

    def forward(
        self,
        node_s: Tensor,       # (B, N, D)
        dist: Tensor,         # (B, N, N)
        rbf: Tensor,          # (B, N, N, R)
        r_hat: Tensor,        # (B, N, N, 3)
        access_mask: Tensor,  # (B, N, N)
    ) -> Tuple[Tensor, Tensor]:
        B, N, D = node_s.shape
        access = access_mask.float()

        src = node_s.unsqueeze(2).expand(B, N, N, D)
        dst = node_s.unsqueeze(1).expand(B, N, N, D)
        edge_base = self.edge_proj(torch.cat([src, dst, rbf], dim=-1))
        edge_base = edge_base * access.unsqueeze(-1)

        max_dist = dist.detach().amax(dim=(1, 2), keepdim=True).clamp_min(1.0) + 1.0
        masked_dist = torch.where(access_mask > 0, dist, max_dist.expand_as(dist))
        topk = min(self.topk, N)
        topk_idx = masked_dist.topk(k=topk, dim=-1, largest=False).indices  # (B, N, K)
        topk_mask = access.gather(2, topk_idx)

        gather_idx_vec = topk_idx.unsqueeze(-1).expand(B, N, topk, 3)
        gather_idx_rbf = topk_idx.unsqueeze(-1).expand(B, N, topk, rbf.shape[-1])
        gather_idx_feat = topk_idx.unsqueeze(-1).expand(B, N, topk, D)

        companion_r_hat = r_hat.gather(2, gather_idx_vec)   # (B, N, K, 3)
        companion_rbf = rbf.gather(2, gather_idx_rbf)       # (B, N, K, R)
        companion_feat = node_s.unsqueeze(1).expand(B, N, N, D).gather(2, gather_idx_feat)  # (B, N, K, D)

        # angle(j, i, k): center atom i, anchor neighbour j, companion k
        cos_theta = torch.einsum("bijx,bikx->bijk", r_hat, companion_r_hat).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle_feat = legendre_basis(cos_theta, self.legendre_order)  # (B, N, N, K, L)

        radial_k = companion_rbf.unsqueeze(2).expand(B, N, N, topk, rbf.shape[-1])
        triplet_weight = self.triplet_proj(torch.cat([angle_feat, radial_k], dim=-1))
        neighbour_feat = companion_feat.unsqueeze(2).expand(B, N, N, topk, D)

        pair_mask = access.unsqueeze(-1) * topk_mask.unsqueeze(2)
        anchor_idx = torch.arange(N, device=node_s.device).view(1, 1, N, 1)
        pair_mask = pair_mask.masked_fill(anchor_idx == topk_idx.unsqueeze(2), 0.0)
        triplet_pair = triplet_weight * neighbour_feat
        pair_logits = self.triplet_score(torch.cat([triplet_weight, neighbour_feat], dim=-1)).squeeze(-1)
        pair_logits = pair_logits.masked_fill(pair_mask <= 0, float("-inf"))
        pair_attn = torch.softmax(pair_logits, dim=3)
        pair_attn = torch.where(pair_mask > 0, pair_attn, torch.zeros_like(pair_attn))
        triplet_attn = (triplet_pair * pair_attn.unsqueeze(-1)).sum(dim=3)

        masked_pair = triplet_pair.masked_fill(pair_mask.unsqueeze(-1) <= 0, float("-inf"))
        triplet_max = masked_pair.max(dim=3).values
        triplet_max = torch.where(torch.isfinite(triplet_max), triplet_max, torch.zeros_like(triplet_max))
        triplet_ctx = self.triplet_mix(torch.cat([triplet_attn, triplet_max], dim=-1))

        edge_feat = self.edge_norm(edge_base + triplet_ctx)
        edge_feat = self.edge_gate(edge_feat) * edge_feat

        node_delta = self.node_out((edge_feat * access.unsqueeze(-1)).sum(dim=2))
        denom = access.sum(dim=(1, 2), keepdim=False).clamp_min(1.0).unsqueeze(-1)
        bond_graph = edge_feat.sum(dim=(1, 2)) / denom
        return node_delta, bond_graph


class ScalarGlobalContextBlock(torch.nn.Module):
    """Low-frequency scalar-only global update.

    A lightweight graph token is constructed from node pooling and bond-summary
    features. It attends to the node sequence and then modulates node scalars
    through FiLM-style gating.
    """

    def __init__(
        self,
        emb_dim: int,
        heads: int,
        path_dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim * 2),
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        self.node_attn = torch.nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            batch_first=True,
            dropout=attn_dropout,
        )
        self.token_attn = torch.nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            batch_first=True,
            dropout=attn_dropout,
        )
        self.film = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim * 2),
            torch.nn.Linear(emb_dim * 2, emb_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim * 2),
        )
        self.drop = DropPath(path_dropout)
        self.node_ln = torch.nn.LayerNorm(emb_dim)
        self.token_ln = torch.nn.LayerNorm(emb_dim)

    @staticmethod
    def _masked_mean(node_s: Tensor, atom_padding: Tensor) -> Tensor:
        valid = (~atom_padding).unsqueeze(-1).float()
        return (node_s * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        node_s: Tensor,
        atom_padding: Tensor,
        bond_graph: Tensor,
        prev_token: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        pooled = self._masked_mean(node_s, atom_padding)
        if prev_token is None:
            prev_token = torch.zeros_like(pooled)
        token = self.token_proj(torch.cat([pooled + prev_token, bond_graph], dim=-1)).unsqueeze(1)

        token_update, _ = self.token_attn(
            query=token,
            key=node_s,
            value=node_s,
            key_padding_mask=atom_padding,
        )
        token = self.token_ln(token + self.drop(token_update))

        node_update, _ = self.node_attn(
            query=node_s,
            key=torch.cat([token, node_s], dim=1),
            value=torch.cat([token, node_s], dim=1),
            key_padding_mask=torch.cat(
                [torch.zeros(atom_padding.shape[0], 1, dtype=torch.bool, device=atom_padding.device), atom_padding],
                dim=1,
            ),
        )

        token_ctx = token.squeeze(1)
        film = self.film(torch.cat([node_s, token_ctx.unsqueeze(1).expand_as(node_s)], dim=-1))
        gamma, beta = film.chunk(2, dim=-1)
        node_s = self.node_ln(node_s + self.drop(node_update) + torch.sigmoid(gamma) * beta)
        node_s = node_s.masked_fill(atom_padding.unsqueeze(-1), 0)
        return node_s, token_ctx


class HybridQM9SE3Block(torch.nn.Module):
    """One hybrid QM9 block: local scalar attention + SE(3) + triplet update."""

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
        triplet_topk: int = 16,
    ) -> None:
        super().__init__()
        self.path_a = UniMolLayer(
            dim=emb_dim,
            num_heads=heads,
            path_dropout=path_dropout,
            activation_dropout=activation_dropout,
            attn_dropout=attn_dropout,
        )
        self.path_b = SE3CGMessageBlock(emb_dim=emb_dim, num_rbf=num_rbf, cutoff=cutoff)
        self.path_d = TripletEdgeAggregation(
            emb_dim=emb_dim,
            num_rbf=num_rbf,
            legendre_order=legendre_order,
            hidden_dim=emb_dim,
            topk=triplet_topk,
        )
        self.edge_bias_proj = NonLinear(num_rbf, heads, hidden=emb_dim)
        self.fusion = CrossPathFusion(emb_dim=emb_dim)
        self.scalar_ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_dim),
            torch.nn.Linear(emb_dim, emb_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Dropout(path_dropout),
            torch.nn.Linear(emb_dim * 4, emb_dim),
        )
        self.scalar_drop = DropPath(path_dropout)

    def forward(
        self,
        node_s: Tensor,
        node_v: Tensor,
        node_t: Tensor,
        dist: Tensor,
        rbf: Tensor,
        r_hat: Tensor,
        access_mask: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        edge_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if edge_bias is None:
            edge_bias = self.edge_bias_proj(rbf)
        node_s_a, edge_bias = self.path_a(
            x=node_s,
            edge_bias=edge_bias,
            attn_mask=attn_mask,
            padding_mask=padding_mask,
        )
        delta_sb, delta_v, delta_t = self.path_b(
            node_s=node_s,
            node_v=node_v,
            node_t=node_t,
            rbf=rbf,
            r_hat=r_hat,
            mask=access_mask,
        )
        delta_sd, bond_graph = self.path_d(
            node_s=node_s,
            dist=dist,
            rbf=rbf,
            r_hat=r_hat,
            access_mask=access_mask,
        )

        node_s, node_v, node_t = self.fusion(
            h=node_s,
            delta_sa=node_s_a - node_s,
            delta_sb=delta_sb,
            delta_sc=delta_sd,
            delta_v=delta_v,
            delta_t=delta_t,
            node_v=node_v,
            node_t=node_t,
        )
        node_s = node_s + self.scalar_drop(self.scalar_ffn(node_s))
        return node_s, node_v, node_t, edge_bias, bond_graph


@ENCODER_REGISTRY.register("reacformer_hybrid")
class ReacFormerHybridEncoder(torch.nn.Module):
    """Hybrid ReacFormer encoder with triplet edge updates and global context."""

    _GLOBAL_EVERY = 2
    _TRIPLET_TOPK = 16

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
        self.cutoff = cutoff
        self.global_every = self._GLOBAL_EVERY

        self.atom_encoder = torch.nn.Embedding(atom_vocab_size, emb_dim)
        self.bond_encoder = RBFEmb(num_rbf, cutoff)
        self.dist_to_rbf_proj = torch.nn.Linear(num_rbf, num_rbf)
        self.blocks = torch.nn.ModuleList([
            HybridQM9SE3Block(
                emb_dim=emb_dim,
                heads=heads,
                num_rbf=num_rbf,
                cutoff=cutoff,
                path_dropout=path_dropout,
                activation_dropout=activation_dropout,
                attn_dropout=attn_dropout,
                legendre_order=legendre_order,
                triplet_topk=self._TRIPLET_TOPK,
            )
            for _ in range(num_layers)
        ])
        self.global_blocks = torch.nn.ModuleList([
            ScalarGlobalContextBlock(
                emb_dim=emb_dim,
                heads=heads,
                path_dropout=path_dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(max(1, (num_layers + self.global_every - 1) // self.global_every))
        ])
        self.edge_bias_init = NonLinear(num_rbf, heads, hidden=emb_dim)

    @staticmethod
    def _masked_mean_pool(node_feats: Tensor, atom_padding: Tensor) -> Tensor:
        valid = (~atom_padding).unsqueeze(-1).float()
        return (node_feats * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

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
        dist = ediff.norm(dim=-1).clamp_min(1e-6)
        r_hat = ediff / dist.unsqueeze(-1)

        rbf = self.bond_encoder(dist)
        rbf = self.dist_to_rbf_proj(rbf)
        pad_2d = atom_padding.unsqueeze(2) | atom_padding.unsqueeze(1) | (dist >= self.cutoff)
        rbf = rbf.masked_fill(pad_2d.unsqueeze(-1), 0)
        r_hat = r_hat.masked_fill(pad_2d.unsqueeze(-1), 0)

        access_mask = create_access_mask(dist, self.cutoff, atom_padding).float()
        attn_mask = create_attn_mask(dist, self.cutoff, atom_padding)
        edge_bias = self.edge_bias_init(rbf)

        global_token = torch.zeros(B, self.emb_dim, device=coords_noisy.device, dtype=node_s.dtype)
        global_idx = 0
        for block_idx, block in enumerate(self.blocks, start=1):
            node_s, node_v, node_t, edge_bias, bond_graph = block(
                node_s=node_s,
                node_v=node_v,
                node_t=node_t,
                dist=dist,
                rbf=rbf,
                r_hat=r_hat,
                access_mask=access_mask,
                attn_mask=attn_mask,
                padding_mask=atom_padding,
                edge_bias=edge_bias,
            )
            if block_idx % self.global_every == 0 or block_idx == len(self.blocks):
                node_s, global_token = self.global_blocks[global_idx](
                    node_s=node_s,
                    atom_padding=atom_padding,
                    bond_graph=bond_graph,
                    prev_token=global_token,
                )
                global_idx += 1

            node_s = node_s.masked_fill(atom_padding.unsqueeze(-1), 0)
            node_v = node_v.masked_fill(atom_padding.unsqueeze(-1).unsqueeze(-1), 0)
            node_t = node_t.masked_fill(atom_padding.unsqueeze(-1).unsqueeze(-1), 0)

        graph_feats = self._masked_mean_pool(node_s, atom_padding)
        return {
            "node_feats": node_s,
            "node_vec": node_v,
            "node_tensor": node_t,
            "graph_feats": graph_feats,
            "coords_input": coords_noisy,
            "atom_padding": atom_padding,
        }
