"""Density pretraining model components."""
from __future__ import annotations

from typing import Dict

import torch

from .reacformer_se3 import ReacFormerSE3Encoder
from .reacformer_so2 import ReacFormerSO2Encoder
from .single_mol import SingleMolEncoder
from .reacformer_hybrid import ReacFormerHybridEncoder
from .gotennet_l import GotenNetLEncoder


class QueryPointDensityHead(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        point_hidden_dim: int = 128,
        cond_hidden_dim: int = 64,
        head_hidden_dim: int = 512,
        radial_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        if radial_sigma <= 0:
            raise ValueError("radial_sigma must be > 0")
        self.radial_sigma = float(radial_sigma)

        self.point_mlp = torch.nn.Sequential(
            torch.nn.Linear(4, point_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(point_hidden_dim, point_hidden_dim),
            torch.nn.SiLU(),
        )
        self.cond_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, cond_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(cond_hidden_dim, cond_hidden_dim),
            torch.nn.SiLU(),
        )
        in_dim = emb_dim + emb_dim + point_hidden_dim + cond_hidden_dim
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, head_hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        graph_feats: torch.Tensor,
        coords: torch.Tensor,
        atom_padding: torch.Tensor,
        query_points: torch.Tensor,
        total_charge: torch.Tensor,
        spin_multiplicity: torch.Tensor,
    ) -> torch.Tensor:
        diff = query_points[:, :, None, :] - coords[:, None, :, :]
        dist2 = (diff * diff).sum(dim=-1)
        valid = (~atom_padding).unsqueeze(1)

        w = torch.exp(-dist2 / (2.0 * (self.radial_sigma ** 2)))
        w = w * valid.float()
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        local_feat = torch.einsum("bpn,bnd->bpd", w, node_feats)

        dist2_masked = dist2.masked_fill(~valid, float("inf"))
        min_dist2 = dist2_masked.min(dim=-1).values
        min_dist2 = torch.where(torch.isfinite(min_dist2), min_dist2, torch.zeros_like(min_dist2))
        min_dist = torch.sqrt(min_dist2 + 1e-8).unsqueeze(-1)

        point_feat = self.point_mlp(torch.cat([query_points, min_dist], dim=-1))
        global_feat = graph_feats.unsqueeze(1).expand(-1, query_points.shape[1], -1)
        cond_input = torch.stack([total_charge, spin_multiplicity], dim=-1)
        cond_feat = self.cond_mlp(cond_input).unsqueeze(1).expand(-1, query_points.shape[1], -1)

        fused = torch.cat([local_feat, global_feat, point_feat, cond_feat], dim=-1)
        return self.out_mlp(fused).squeeze(-1)


class DensityPretrainNet(torch.nn.Module):
    def __init__(
        self,
        encoder_type: str = "single_mol",
        emb_dim: int = 256,
        inv_layer: int = 2,
        se3_layer: int = 4,
        heads: int = 8,
        atom_vocab_size: int = 128,
        cutoff: float = 5.0,
        num_kernel: int = 128,
        path_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        point_hidden_dim: int = 128,
        cond_hidden_dim: int = 64,
        head_hidden_dim: int = 512,
        radial_sigma: float = 1.5,
    ) -> None:
        super().__init__()
        if encoder_type == "single_mol":
            self.descriptor = SingleMolEncoder(
                emb_dim=emb_dim,
                inv_layer=inv_layer,
                se3_layer=se3_layer,
                heads=heads,
                atom_vocab_size=atom_vocab_size,
                cutoff=cutoff,
                path_dropout=path_dropout,
                activation_dropout=activation_dropout,
                attn_dropout=attn_dropout,
                num_kernel=num_kernel,
            )
        elif encoder_type == "reacformer_se3":
            self.descriptor = ReacFormerSE3Encoder(
                emb_dim=emb_dim,
                num_layers=se3_layer,
                heads=heads,
                atom_vocab_size=atom_vocab_size,
                cutoff=cutoff,
                num_rbf=num_kernel,
                path_dropout=path_dropout,
                activation_dropout=activation_dropout,
                attn_dropout=attn_dropout,
            )
        elif encoder_type == "reacformer_so2":
            self.descriptor = ReacFormerSO2Encoder(
                emb_dim=emb_dim,
                num_layers=se3_layer,
                heads=heads,
                atom_vocab_size=atom_vocab_size,
                cutoff=cutoff,
                num_rbf=num_kernel,
                path_dropout=path_dropout,
                activation_dropout=activation_dropout,
                attn_dropout=attn_dropout,
            )
        elif encoder_type == "reacformer_hybrid":
            self.descriptor = ReacFormerHybridEncoder(
                emb_dim=emb_dim,
                num_layers=se3_layer,
                heads=heads,
                atom_vocab_size=atom_vocab_size,
                cutoff=cutoff,
                num_rbf=num_kernel,
                path_dropout=path_dropout,
                activation_dropout=activation_dropout,
                attn_dropout=attn_dropout,
            )
        elif encoder_type == "gotennet_l":
            self.descriptor = GotenNetLEncoder(
                emb_dim=emb_dim,
                num_layers=se3_layer,
                heads=heads,
                atom_vocab_size=atom_vocab_size,
                cutoff=cutoff,
                num_rbf=num_kernel,
                path_dropout=path_dropout,
                activation_dropout=activation_dropout,
                attn_dropout=attn_dropout,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.density_head = QueryPointDensityHead(
            emb_dim=emb_dim,
            point_hidden_dim=point_hidden_dim,
            cond_hidden_dim=cond_hidden_dim,
            head_hidden_dim=head_hidden_dim,
            radial_sigma=radial_sigma,
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coords: torch.Tensor,
        atom_padding: torch.Tensor,
        query_points: torch.Tensor,
        total_charge: torch.Tensor,
        spin_multiplicity: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        desc = self.descriptor(
            input_atomic_numbers=atomic_numbers,
            coords_noisy=coords,
            atom_padding=atom_padding,
        )
        density_pred = self.density_head(
            node_feats=desc["node_feats"],
            graph_feats=desc["graph_feats"],
            coords=coords,
            atom_padding=atom_padding,
            query_points=query_points,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
        )
        return {"density_pred": density_pred}
