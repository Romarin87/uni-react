"""ReactionPretrainNet – EMA teacher-student model for reaction triplet pretraining."""
import copy
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


class ReactionPretrainNet(torch.nn.Module):
    """Online + EMA-teacher encoder for stage-3 reaction consistency pretraining.

    Args:
        descriptor: Pre-built backbone encoder (online copy).
        emb_dim: Embedding dimension.
        head_hidden_dim: Hidden dim for MLP heads.
        teacher_momentum: EMA momentum m; teacher <- m*teacher + (1-m)*online.
    """

    def __init__(
        self,
        descriptor: torch.nn.Module,
        emb_dim: int,
        head_hidden_dim: int = 512,
        teacher_momentum: float = 0.999,
    ) -> None:
        super().__init__()
        self.online_descriptor  = descriptor
        self.teacher_descriptor = copy.deepcopy(descriptor)
        for p in self.teacher_descriptor.parameters():
            p.requires_grad = False

        self.teacher_momentum = float(teacher_momentum)
        pair_dim    = emb_dim * 4
        triplet_dim = emb_dim * 9

        def _mlp(in_dim: int, out_dim: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.LayerNorm(in_dim),
                torch.nn.Linear(in_dim, head_hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(head_hidden_dim, out_dim),
            )

        self.consistency_head  = _mlp(triplet_dim, 1)
        self.complete_ts_head  = _mlp(pair_dim, emb_dim)
        self.complete_p_head   = _mlp(pair_dim, emb_dim)
        self.complete_r_head   = _mlp(pair_dim, emb_dim)

    @staticmethod
    def _pair_feat(a: Tensor, b: Tensor) -> Tensor:
        return torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)

    @staticmethod
    def _triplet_feat(r: Tensor, ts: Tensor, p: Tensor) -> Tensor:
        return torch.cat(
            [r, ts, p,
             torch.abs(r - ts), torch.abs(ts - p),
             r * ts, ts * p, r * p, torch.abs(r - p)],
            dim=-1,
        )

    @staticmethod
    def _encode_graph(
        encoder: torch.nn.Module,
        atomic_numbers: Tensor,
        coords: Tensor,
        atom_padding: Tensor,
    ) -> Tensor:
        out = encoder(input_atomic_numbers=atomic_numbers, coords_noisy=coords, atom_padding=atom_padding)
        return out["graph_feats"]

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Online: encode all six molecules
        h_r  = self._encode_graph(self.online_descriptor, batch["R_atomic_numbers"],  batch["R_coords"],  batch["R_padding"])
        h_ts = self._encode_graph(self.online_descriptor, batch["TS_atomic_numbers"], batch["TS_coords"], batch["TS_padding"])
        h_p  = self._encode_graph(self.online_descriptor, batch["P_atomic_numbers"],  batch["P_coords"],  batch["P_padding"])
        h_r_c  = self._encode_graph(self.online_descriptor, batch["R_cons_atomic_numbers"],  batch["R_cons_coords"],  batch["R_cons_padding"])
        h_ts_c = self._encode_graph(self.online_descriptor, batch["TS_cons_atomic_numbers"], batch["TS_cons_coords"], batch["TS_cons_padding"])
        h_p_c  = self._encode_graph(self.online_descriptor, batch["P_cons_atomic_numbers"],  batch["P_cons_coords"],  batch["P_cons_padding"])

        # Consistency classification on the consistency triplet
        cons_logits = self.consistency_head(
            self._triplet_feat(h_r_c, h_ts_c, h_p_c)
        ).squeeze(-1)

        # Completion: predict teacher targets from online pair features
        pred_ts = self.complete_ts_head(self._pair_feat(h_r, h_p))
        pred_p  = self.complete_p_head(self._pair_feat(h_r, h_ts))
        pred_r  = self.complete_r_head(self._pair_feat(h_p, h_ts))

        with torch.no_grad():
            self.teacher_descriptor.eval()
            tgt_r  = self._encode_graph(self.teacher_descriptor, batch["R_atomic_numbers"],  batch["R_coords"],  batch["R_padding"])
            tgt_ts = self._encode_graph(self.teacher_descriptor, batch["TS_atomic_numbers"], batch["TS_coords"], batch["TS_padding"])
            tgt_p  = self._encode_graph(self.teacher_descriptor, batch["P_atomic_numbers"],  batch["P_coords"],  batch["P_padding"])

        comp_ts_loss = F.mse_loss(F.normalize(pred_ts, dim=-1), F.normalize(tgt_ts, dim=-1))
        comp_p_loss  = F.mse_loss(F.normalize(pred_p,  dim=-1), F.normalize(tgt_p,  dim=-1))
        comp_r_loss  = F.mse_loss(F.normalize(pred_r,  dim=-1), F.normalize(tgt_r,  dim=-1))
        comp_loss = (comp_ts_loss + comp_p_loss + comp_r_loss) / 3.0

        return {
            "cons_logits":   cons_logits,
            "comp_loss":     comp_loss,
            "comp_ts_loss":  comp_ts_loss,
            "comp_p_loss":   comp_p_loss,
            "comp_r_loss":   comp_r_loss,
        }

    @torch.no_grad()
    def update_teacher(self) -> None:
        """EMA update: teacher <- momentum * teacher + (1 - momentum) * online."""
        m = float(self.teacher_momentum)
        for t, o in zip(
            self.teacher_descriptor.parameters(),
            self.online_descriptor.parameters(),
        ):
            t.data.mul_(m).add_(o.data, alpha=1.0 - m)
