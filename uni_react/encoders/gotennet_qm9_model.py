"""QM9 model using official GotenNet property-specific output heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ._layers.gotennet_vendor.outputs import Atomwise, Dipole, ElectronicSpatialExtentV2
from .gotennet_l import GotenNetLEncoder
from ..utils.qm9_dataset import QM9_TARGETS


@dataclass
class GotenNetQM9Metadata:
    target: str
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    atomref: Optional[torch.Tensor] = None


class _AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class GotenNetQM9Net(torch.nn.Module):
    """QM9 fine-tuning model aligned with the official GotenNet task heads."""

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
        descriptor: Optional[torch.nn.Module] = None,
        target: str = "gap",
        metadata: Optional[GotenNetQM9Metadata] = None,
    ) -> None:
        super().__init__()
        del inv_layer, path_dropout, activation_dropout
        self.target = target
        self.descriptor = descriptor or GotenNetLEncoder(
            emb_dim=emb_dim,
            num_layers=se3_layer,
            heads=heads,
            atom_vocab_size=atom_vocab_size,
            cutoff=cutoff,
            num_rbf=num_kernel,
            attn_dropout=attn_dropout,
        )
        self.metadata = metadata or GotenNetQM9Metadata(target=target)
        self.output_head = self._build_output_head(emb_dim, target, self.metadata)

    def set_target_stats(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> None:
        self.metadata.mean = mean
        self.metadata.std = std
        self.output_head = self._build_output_head(
            self.descriptor.emb_dim,
            self.target,
            self.metadata,
        ).to(next(self.parameters()).device)

    @staticmethod
    def _build_output_head(emb_dim: int, target: str, meta: GotenNetQM9Metadata):
        target_lower = target.lower()
        if target_lower == "mu":
            return Dipole(
                n_in=emb_dim,
                predict_magnitude=True,
                property="property",
                mean=meta.mean,
                stddev=meta.std,
            )
        if target_lower == "r2":
            return ElectronicSpatialExtentV2(
                n_in=emb_dim,
                property="property",
                mean=meta.mean,
                stddev=meta.std,
            )
        return Atomwise(
            n_in=emb_dim,
            property="property",
            activation=torch.nn.functional.silu,
            mean=meta.mean,
            stddev=meta.std,
            atomref=meta.atomref,
        )

    @staticmethod
    def _flatten_masked(
        atomic_numbers: torch.Tensor,
        coords: torch.Tensor,
        atom_padding: torch.Tensor,
        node_feats: torch.Tensor,
        node_vec: torch.Tensor,
    ) -> _AttrDict:
        valid = ~atom_padding
        batch_size, max_atoms = atomic_numbers.shape
        batch = (
            torch.arange(batch_size, device=atomic_numbers.device)
            .unsqueeze(1)
            .expand(batch_size, max_atoms)[valid]
        )
        return _AttrDict(
            z=atomic_numbers[valid],
            pos=coords[valid],
            batch=batch,
            representation=node_feats[valid],
            vector_representation=node_vec[valid],
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        coords: torch.Tensor,
        atom_padding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if atom_padding is None:
            atom_padding = torch.zeros_like(atomic_numbers, dtype=torch.bool)
        backbone_out = self.descriptor(
            input_atomic_numbers=atomic_numbers,
            coords_noisy=coords,
            atom_padding=atom_padding,
        )
        head_inputs = self._flatten_masked(
            atomic_numbers=atomic_numbers,
            coords=coords,
            atom_padding=atom_padding,
            node_feats=backbone_out["node_feats"],
            node_vec=backbone_out["node_vec"],
        )
        result = self.output_head(head_inputs)
        pred = result["property"]
        if pred.ndim == 2 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        return {
            "pred": pred,
            "pred_is_normalized": False,
            "node_feats": backbone_out["node_feats"],
        }


def build_gotennet_qm9_metadata(
    target: str,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    atomref: Optional[torch.Tensor] = None,
) -> GotenNetQM9Metadata:
    if target not in QM9_TARGETS:
        raise ValueError(f"Unsupported QM9 target for GotenNetQM9Net: {target!r}")
    return GotenNetQM9Metadata(target=target, mean=mean, std=std, atomref=atomref)
