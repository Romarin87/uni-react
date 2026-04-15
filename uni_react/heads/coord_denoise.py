"""Coordinate-denoising prediction head."""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from ..registry import HEAD_REGISTRY


class _GatedEquivariantBlock(torch.nn.Module):
    """Single gated equivariant update block (internal to CoordDenoiseHead)."""

    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.vec1_proj   = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj   = torch.nn.Linear(hidden_channels, out_channels,   bias=False)
        self.update_net  = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, out_channels * 2),
        )
        torch.nn.init.xavier_uniform_(self.vec1_proj.weight)
        torch.nn.init.xavier_uniform_(self.vec2_proj.weight)
        torch.nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x: Tensor, v: Tensor):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)
        x    = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v    = torch.einsum("...c,...dc->...dc", v, vec2)
        return x, v


class _EquiOutput(torch.nn.Module):
    """SE(3)-equivariant output MLP that maps (scalar, vector) → coordinate delta."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.output_network = torch.nn.ModuleList([
            _GatedEquivariantBlock(hidden_channels, hidden_channels // 2),
            _GatedEquivariantBlock(hidden_channels // 2, 1),
        ])
        self.actf = torch.nn.SiLU()

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        for i, layer in enumerate(self.output_network):
            x, vec = layer(x, vec)
            if i != len(self.output_network) - 1:
                x = self.actf(x)
        return vec.squeeze(dim=-1)


@HEAD_REGISTRY.register("coord_denoise")
class CoordDenoiseHead(torch.nn.Module):
    """Predicts denoised 3-D coordinates from noisy inputs.

    Config example::

        heads:
          - type: coord_denoise
            emb_dim: 256
    """

    name = "coord_denoise"

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.head = _EquiOutput(emb_dim)

    def forward(self, descriptors: Dict[str, Tensor]) -> Dict[str, Tensor]:
        node_feats   = descriptors["node_feats"]
        node_vec     = descriptors["node_vec"]
        coords_input = descriptors["coords_input"]
        atom_padding = descriptors["atom_padding"]

        coords_delta    = self.head(node_feats, node_vec)
        coords_denoised = coords_input + coords_delta
        coords_delta    = coords_delta.masked_fill(atom_padding[..., None], 0)
        coords_denoised = coords_denoised.masked_fill(atom_padding[..., None], 0)
        return {"coords_delta": coords_delta, "coords_denoised": coords_denoised}

    def compute_loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        atom_padding = batch["atom_padding"]
        valid_atom   = ~atom_padding
        if valid_atom.any():
            return F.mse_loss(
                outputs["coords_denoised"][valid_atom],
                batch["coords"][valid_atom],
            )
        return outputs["coords_denoised"].sum() * 0.0
