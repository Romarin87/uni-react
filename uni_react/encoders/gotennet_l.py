"""GotenNet-L encoder adapted to the uni_react backbone interface."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from types import SimpleNamespace

from ._layers.gotennet_vendor import GotenNetWrapper
from ._layers.gotennet_vendor.gotennet_layers import CosineCutoff
from ..registry import ENCODER_REGISTRY


def _flatten_padded_batch(
    atomic_numbers: Tensor,
    coords: Tensor,
    atom_padding: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    valid = ~atom_padding
    batch_size, max_atoms = atomic_numbers.shape
    batch_index = (
        torch.arange(batch_size, device=atomic_numbers.device)
        .unsqueeze(1)
        .expand(batch_size, max_atoms)
    )
    flat_z = atomic_numbers[valid]
    flat_pos = coords[valid]
    flat_batch = batch_index[valid]
    counts = valid.sum(dim=1)
    return flat_z, flat_pos, flat_batch, counts


def _unflatten_nodes(
    flat_h: Tensor,
    flat_x: Tensor,
    counts: Tensor,
    atom_padding: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size, max_atoms = atom_padding.shape
    hidden_dim = flat_h.shape[-1]
    device = flat_h.device
    dtype = flat_h.dtype
    node_feats = torch.zeros(batch_size, max_atoms, hidden_dim, device=device, dtype=dtype)
    node_vec = torch.zeros(batch_size, max_atoms, 3, hidden_dim, device=device, dtype=dtype)
    node_tensor = torch.zeros(batch_size, max_atoms, 5, hidden_dim, device=device, dtype=dtype)

    start = 0
    for i in range(batch_size):
        n = int(counts[i].item())
        end = start + n
        if n > 0:
            node_feats[i, :n] = flat_h[start:end]
            if flat_x.shape[1] >= 3:
                node_vec[i, :n] = flat_x[start:end, :3]
            if flat_x.shape[1] >= 8:
                node_tensor[i, :n] = flat_x[start:end, 3:8]
        start = end
    return node_feats, node_vec, node_tensor


@ENCODER_REGISTRY.register("gotennet_l")
class GotenNetLEncoder(torch.nn.Module):
    """Original GotenNet architecture exposed as a uni_react backbone.

    This uses the official GotenNet representation with the QM9-style settings
    from the public repository: hidden size 256, four interaction blocks, and
    lmax=2. External arguments are accepted to stay source-compatible with the
    existing training entrypoints; QM9-specific fixed settings are validated
    rather than silently remapped.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_layers: int = 4,
        heads: int = 8,
        atom_vocab_size: int = 128,
        cutoff: float = 5.0,
        num_rbf: int = 64,
        path_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if path_dropout != 0.1:
            raise ValueError(
                "gotennet_l does not support path_dropout; keep the official fixed setting "
                f"(expected 0.1 placeholder, got {path_dropout})."
            )
        if activation_dropout != 0.1:
            raise ValueError(
                "gotennet_l does not support activation_dropout; keep the official fixed setting "
                f"(expected 0.1 placeholder, got {activation_dropout})."
            )
        self.emb_dim = emb_dim
        self.cutoff = cutoff
        self.lmax = 2
        self.representation = GotenNetWrapper(
            n_atom_basis=emb_dim,
            n_interactions=num_layers,
            radial_basis="expnorm",
            n_rbf=num_rbf,
            cutoff_fn=CosineCutoff(cutoff),
            activation="swish",
            max_z=max(atom_vocab_size, 100),
            layernorm="layer",
            steerable_norm="tensor",
            num_heads=heads,
            attn_dropout=attn_dropout,
            edge_updates=True,
            scale_edge=False,
            lmax=self.lmax,
            aggr="add",
            sep_htr=True,
            sep_dir=True,
            sep_tensor=True,
            edge_ln="",
            max_num_neighbors=32,
        )

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
        if atom_padding is None:
            atom_padding = torch.zeros_like(input_atomic_numbers, dtype=torch.bool)

        flat_z, flat_pos, flat_batch, counts = _flatten_padded_batch(
            input_atomic_numbers, coords_noisy, atom_padding
        )
        data = SimpleNamespace(z=flat_z, pos=flat_pos, batch=flat_batch)
        flat_h, flat_x = self.representation(data)
        node_feats, node_vec, node_tensor = _unflatten_nodes(flat_h, flat_x, counts, atom_padding)
        node_feats = node_feats.masked_fill(atom_padding.unsqueeze(-1), 0)
        node_vec = node_vec.masked_fill(atom_padding.unsqueeze(-1).unsqueeze(-1), 0)
        node_tensor = node_tensor.masked_fill(atom_padding.unsqueeze(-1).unsqueeze(-1), 0)
        graph_feats = self._masked_mean_pool(node_feats, atom_padding)
        return {
            "node_feats": node_feats,
            "node_vec": node_vec,
            "node_tensor": node_tensor,
            "graph_feats": graph_feats,
            "coords_input": coords_noisy,
            "atom_padding": atom_padding,
        }
