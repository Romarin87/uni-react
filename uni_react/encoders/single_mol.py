"""SingleMolEncoder – invariant MPNN + SE(3)-equivariant backbone.

The full implementation lives here; the old ``descriptor/`` package has been
removed. This encoder is registered in ENCODER_REGISTRY under ``"single_mol"``.
"""
import math
from typing import Dict, Optional

import torch
from torch import Tensor

from ._layers import FCEqMPLayer, FCSVec, FTE, NonLinear, RBFEmb, UniMolLayer, safe_normalization, create_access_mask, create_attn_mask
from ..registry import ENCODER_REGISTRY


@ENCODER_REGISTRY.register("single_mol")
class SingleMolEncoder(torch.nn.Module):
    """Single-molecule 3D descriptor encoder (invariant MPNN + SE3 equivariant layers).

    Returns a dict with:
    - ``node_feats``   (B, N, D)  – per-atom embeddings
    - ``node_vec``     (B, N, 3, D) – equivariant vectors
    - ``graph_feats``  (B, D)    – mean-pooled graph embedding
    - ``coords_input`` (B, N, 3) – original (possibly noisy) coordinates
    - ``atom_padding`` (B, N)    – bool mask, True = padding

    Config example::

        encoder:
          type: single_mol
          emb_dim: 256
          inv_layer: 2
          se3_layer: 4
          heads: 8
    """

    def __init__(
        self,
        emb_dim: int,
        inv_layer: int,
        se3_layer: int,
        heads: int,
        atom_vocab_size: int = 128,
        cutoff: float = 5.0,
        path_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_kernel: int = 128,
    ) -> None:
        super().__init__()
        if inv_layer < 1:
            raise ValueError("inv_layer must be >= 1")
        if heads <= 0:
            raise ValueError("heads must be > 0")
        if emb_dim % heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by heads ({heads})")
        if atom_vocab_size <= 0:
            raise ValueError("atom_vocab_size must be > 0")

        self.emb_dim = emb_dim
        self.inv_layer = inv_layer
        self.se3_layer = se3_layer
        self.heads = heads
        self.cutoff = cutoff

        self.inv_mpnns = torch.nn.ModuleList([
            UniMolLayer(
                dim=emb_dim,
                num_heads=heads,
                path_dropout=path_dropout,
                activation_dropout=activation_dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(inv_layer)
        ])

        wdim = emb_dim * 3 + num_kernel
        self.message_layers = torch.nn.ModuleList()
        self.FTEs = torch.nn.ModuleList()
        self.SAs = torch.nn.ModuleList()
        self.lns_before_attn = torch.nn.ModuleList()
        self.lns_after_attn = torch.nn.ModuleList()
        self.lns_after_mp = torch.nn.ModuleList()

        for _ in range(se3_layer):
            self.message_layers.append(FCEqMPLayer(emb_dim, emb_dim, wdim))
            self.FTEs.append(FTE(emb_dim))
            self.SAs.append(torch.nn.MultiheadAttention(
                embed_dim=emb_dim, num_heads=heads,
                batch_first=True, dropout=attn_dropout,
            ))
            self.lns_before_attn.append(torch.nn.LayerNorm(emb_dim))
            self.lns_after_attn.append(torch.nn.LayerNorm(emb_dim))
            self.lns_after_mp.append(torch.nn.LayerNorm(emb_dim))

        self.atom_encoder = torch.nn.Embedding(atom_vocab_size, emb_dim)
        bond_encoder = RBFEmb(num_kernel, cutoff)
        try:
            self.bond_encoder = torch.jit.script(bond_encoder)
        except Exception:
            # TorchScript can fail on some local PyTorch/Python combinations.
            # Fall back to the eager module so training and smoke tests still run.
            self.bond_encoder = bond_encoder

        self.dist_to_bond = NonLinear(num_kernel, emb_dim, dropout=activation_dropout, hidden=emb_dim)
        self.src_to_bond  = NonLinear(emb_dim,    emb_dim, dropout=activation_dropout, hidden=emb_dim)
        self.tgt_to_bond  = NonLinear(emb_dim,    emb_dim, dropout=activation_dropout, hidden=emb_dim)
        self.bond_to_bias = NonLinear(emb_dim,    heads,   dropout=activation_dropout, hidden=emb_dim)

        # Project edge RBF features to weight_dim for FCEqMPLayer.
        self.Svec = NonLinear(emb_dim, wdim, hidden=emb_dim)
        # Project 3 frame directions to 1 for node_frame construction.
        self.lin  = NonLinear(3, 1, hidden=max(emb_dim // 4, 8))
        self.inv_sqrt_2 = 1.0 / math.sqrt(2.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _masked_mean_pool(node_feats: Tensor, atom_padding: Tensor) -> Tensor:
        valid = (~atom_padding).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1)
        return (node_feats * valid).sum(dim=1) / denom

    @staticmethod
    def _safe_distance(ediff: Tensor) -> Tensor:
        """Compute safe Euclidean distance with dtype-aware epsilon.
        
        Args:
            ediff: Edge difference vectors (B, N, N, 3).
            
        Returns:
            Distance tensor (B, N, N) with numerical stability.
        """
        # Use larger epsilon for lower precision
        eps = 1e-6 if ediff.dtype == torch.float32 else 1e-4
        return torch.sqrt((ediff * ediff).sum(dim=-1) + eps)

    def _feature_initialization(self, atomic_numbers: Tensor, pos: Tensor,
                                 atom_padding: Optional[Tensor] = None):
        node_feats = self.atom_encoder(atomic_numbers)
        if atom_padding is not None:
            node_feats = node_feats.masked_fill(atom_padding[..., None], 0)

        ediff = pos[:, :, None] - pos[:, None]
        dist  = self._safe_distance(ediff)

        access_mask    = create_access_mask(dist, self.cutoff, atom_padding)
        unimol_attn_mask = create_attn_mask(dist, self.cutoff, atom_padding)

        if atom_padding is not None:
            pad_2d = atom_padding[:, :, None] | atom_padding[:, None, :]
            pad_2d = pad_2d | (dist >= self.cutoff)
        else:
            pad_2d = dist >= self.cutoff

        radial_emb = self.bond_encoder(dist)
        radial_emb = radial_emb.masked_fill(pad_2d[..., None], 0)

        radial_hidden = (
            self.dist_to_bond(radial_emb)
            + self.src_to_bond(node_feats)[:, :, None]
            + self.tgt_to_bond(node_feats)[:, None]
        )
        radial_hidden = radial_hidden.masked_fill(pad_2d[..., None], 0)

        soft_cutoff = dist * torch.pi / self.cutoff
        soft_cutoff = 0.5 * (torch.cos(soft_cutoff) + 1.0)
        soft_cutoff = soft_cutoff.masked_fill(dist >= self.cutoff, 0)

        unimol_bias = self.bond_to_bias(radial_hidden)
        for i in range(self.inv_layer):
            node_feats, unimol_bias = self.inv_mpnns[i](
                x=node_feats, edge_bias=unimol_bias,
                attn_mask=unimol_attn_mask, padding_mask=atom_padding,
            )

        vec = torch.zeros(
            (*node_feats.shape[:2], 3, self.emb_dim),
            device=pos.device, dtype=node_feats.dtype,
        )

        edgef_diff  = ediff / (dist + 1e-6)[..., None]
        edgef_cross = safe_normalization(
            torch.cross(
                pos[:, :, None].expand_as(ediff),
                ediff, dim=-1,
            ), dim=-1,
        )
        edgef_third = safe_normalization(
            torch.cross(edgef_diff, edgef_cross, dim=-1), dim=-1,
        )
        edge_frame = torch.stack([edgef_diff, edgef_cross, edgef_third], dim=-1)

        access_mask_float = access_mask.float()

        # A_i_j: soft-cutoff-weighted edge weights for FCEqMPLayer (B, N, N, wdim)
        A_i_j = soft_cutoff[..., None] * self.Svec(radial_hidden)

        # Build per-node local frame by weighted-averaging edge frame directions.
        # edge_frame: (B, N, N, 3, 3), access_mask_float: (B, N, N)
        # node_frame_raw: (B, N, 3, 3)
        node_frame_raw = torch.einsum("bqkdc,bqk->bqdc", edge_frame, access_mask_float)
        # lin: NonLinear(3,1) collapses last dim 3→1 to produce weighted combination.
        # node_frame_raw permuted: (B, N, 3, 3) → lin input (..., 3) → (..., 1)
        lin_out = self.lin(node_frame_raw)          # (B, N, 3, 1)
        lin_out = lin_out.squeeze(-1).unsqueeze(-1)  # (B, N, 3, 1)
        node_frame = safe_normalization(
            lin_out + node_frame_raw, dim=-2
        )
        return node_feats, vec, radial_hidden, A_i_j, edge_frame, node_frame, access_mask_float

    def _mpnn_one(self, layer: int, feats: Tensor, vec: Tensor,
                  access_mask: Tensor, edge_vector: Tensor, A_i_j: Tensor,
                  radial_hidden: Tensor, node_frame: Tensor,
                  atom_padding: Optional[Tensor] = None):
        feats = self.lns_before_attn[layer](feats)
        feats = self.SAs[layer](
            query=feats, key=feats, value=feats,
            key_padding_mask=atom_padding,
        )[0] + feats
        feats = self.lns_after_attn[layer](feats)

        ds, dvec = self.message_layers[layer](
            x=feats, vec=vec, access_mask=access_mask,
            weight=A_i_j, edge_rbf=radial_hidden, edge_vector=edge_vector,
        )
        feats = self.lns_after_mp[layer](feats + ds)
        vec   = vec + dvec

        ds, dvec = self.FTEs[layer](feats, vec, node_frame)
        feats = (feats + ds) * self.inv_sqrt_2
        vec   = vec + dvec
        return feats, vec

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_atomic_numbers: Tensor,
        coords_noisy: Tensor,
        atom_padding: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if atom_padding is None:
            atom_padding = torch.zeros_like(input_atomic_numbers, dtype=torch.bool)

        feats, vec, radial_hidden, A_i_j, edge_frame, node_frame, access_mask = \
            self._feature_initialization(
                atomic_numbers=input_atomic_numbers,
                pos=coords_noisy,
                atom_padding=atom_padding,
            )

        for i in range(self.se3_layer):
            feats, vec = self._mpnn_one(
                layer=i, feats=feats, vec=vec,
                access_mask=access_mask,
                edge_vector=edge_frame[..., 0],
                A_i_j=A_i_j, radial_hidden=radial_hidden,
                node_frame=node_frame, atom_padding=atom_padding,
            )
            feats = feats.masked_fill(atom_padding[..., None], 0)
            vec   = vec.masked_fill(atom_padding[..., None, None], 0)

        graph_feats = self._masked_mean_pool(feats, atom_padding)
        return {
            "node_feats":   feats,
            "node_vec":     vec,
            "graph_feats":  graph_feats,
            "coords_input": coords_noisy,
            "atom_padding": atom_padding,
        }


# Backward-compat alias so old checkpoints that referenced SingleMolDescriptor
# still import cleanly.
SingleMolDescriptor = SingleMolEncoder
