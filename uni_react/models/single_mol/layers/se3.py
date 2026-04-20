import math
from typing import Tuple

import torch


class FTE(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.equi_proj = torch.nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xequi_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden_channels)
        # MLP that maps 3 local-frame scalars → 1 scalar gate.
        # Hidden dims (48, 8) are fixed architectural constants; changing them
        # requires retraining from scratch.
        _FRAME_HIDDEN_1 = 48
        _FRAME_HIDDEN_2 = 8
        self.lin3 = torch.nn.Sequential(
            torch.nn.Linear(3, _FRAME_HIDDEN_1),
            torch.nn.SiLU(),
            torch.nn.Linear(_FRAME_HIDDEN_1, _FRAME_HIDDEN_2),
            torch.nn.SiLU(),
            torch.nn.Linear(_FRAME_HIDDEN_2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.equi_proj.weight)
        torch.nn.init.xavier_uniform_(self.xequi_proj[0].weight)
        self.xequi_proj[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.xequi_proj[2].weight)
        self.xequi_proj[2].bias.data.fill_(0)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, node_frame: torch.Tensor):
        vec = self.equi_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)

        scalrization = torch.einsum("...dc,...dx->...xc", vec1, node_frame)
        scalar = self.lin3(scalrization.transpose(-1, -2)).squeeze(dim=-1)

        vec_dot = (vec1 * vec2).sum(dim=-2) * self.inv_sqrt_h
        x_vec_h = self.xequi_proj(torch.cat([x, scalar], dim=-1))
        xvecs = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = (xvecs[0] + xvecs[1] * vec_dot) * self.inv_sqrt_2
        dvec = torch.einsum("...c,...dc->...dc", xvecs[2], vec2)
        return dx, dvec * self.inv_sqrt_h


class GatedEquivariantBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.vec1_proj = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.update_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, out_channels * 2),
        )

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.vec1_proj.weight)
        torch.nn.init.xavier_uniform_(self.vec2_proj.weight)
        torch.nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x: torch.Tensor, v: torch.Tensor):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = torch.einsum("...c,...dc->...dc", v, vec2)
        return x, v


class EquiOutput(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.output_network = torch.nn.ModuleList(
            [
                GatedEquivariantBlock(hidden_channels, hidden_channels // 2),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )
        self.actf = torch.nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.output_network):
            x, vec = layer(x, vec)
            if i != len(self.output_network) - 1:
                x = self.actf(x)
        return vec.squeeze(dim=-1)


class FCSVec(torch.nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor, ev: torch.Tensor, ef: torch.Tensor, access_mask: torch.Tensor):
        x = self.proj(x)
        ev = torch.einsum("bqkd,bqkc->bqkdc", ev, ef)
        return torch.einsum("bkc,bqkdc,bqk->bqdc", x, ev, access_mask)


class FCEqMPLayer(torch.nn.Module):
    def __init__(self, hidden_channels: int, edge_dim: int, weight_dim: int):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.inv_proj = torch.nn.Sequential(
            torch.nn.Linear(weight_dim, hidden_channels * 3),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels * 3, hidden_channels * 3),
        )

        self.x_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.edge_proj = torch.nn.Linear(edge_dim, hidden_channels * 3)

        self.inv_sqrt_3 = 1.0 / math.sqrt(3.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        access_mask: torch.Tensor,
        edge_rbf: torch.Tensor,
        weight: torch.Tensor,
        edge_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xh = self.x_proj(x)
        rbfh = self.edge_proj(edge_rbf)
        weight = self.inv_proj(weight)

        x, xh2, xh3 = torch.split(
            torch.einsum("bkd,bqkd->bqkd", xh, rbfh * weight),
            self.hidden_channels,
            dim=-1,
        )
        xh2 = xh2 * self.inv_sqrt_3
        vec = (
            torch.einsum("bkdc,bqkc->bqkdc", vec, xh2)
            + torch.einsum("bqkc,bqkd->bqkdc", xh3, edge_vector)
        ) * self.inv_sqrt_h

        s_aggr = torch.einsum("bqkd,bqk->bqd", x, access_mask)
        v_aggr = torch.einsum("bqkdc,bqk->bqdc", vec, access_mask)
        return s_aggr, v_aggr
