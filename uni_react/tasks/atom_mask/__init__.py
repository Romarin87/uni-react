"""Atom masking task for joint training."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from ..common import DatasetBuildResult, MoleculeTaskAdapter, TaskSpec, extract_descriptors
from ..components.molecule_dataset import H5SingleMolPretrainDataset
from .head import AtomMaskHead


class AtomMaskAdapter(MoleculeTaskAdapter):
    required_keys = ("frames/offsets|mol_offsets", "atoms/Z|atom_numbers", "atoms/R|coords")

    def build_dataset(self, files: Sequence[str], split: str) -> DatasetBuildResult:
        kwargs = self._dataset_kwargs(split=split, require_reactivity=False)
        kwargs["mask_ratio"] = float(self.params.get("mask_ratio", 0.15))
        kwargs["min_masked"] = int(self.params.get("min_masked", 1))
        dataset = H5SingleMolPretrainDataset(files, **kwargs)
        return DatasetBuildResult(dataset, self.schema, self.required_keys, files)

    def compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = outputs["atom_logits"]
        targets = batch.get("target_atomic_numbers", batch["atomic_numbers"])
        mask = batch.get("mask_positions")
        loss_flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
        pred = logits.argmax(dim=-1)
        correct = pred.eq(targets)
        if mask is not None and mask.any():
            mask_flat = mask.reshape(-1).float()
            loss = (loss_flat * mask_flat).sum() / mask_flat.sum().clamp_min(1.0)
            acc = correct[mask].float().mean()
        else:
            loss = logits.sum() * 0.0
            acc = logits.new_tensor(0.0)
        return {"loss": loss, "acc": acc}

    def metric_names(self) -> Sequence[str]:
        return ("loss", "acc")


def build_head(*, emb_dim: int, atom_vocab_size: int, params: Dict) -> torch.nn.Module:
    del params
    return AtomMaskHead(emb_dim=emb_dim, atom_vocab_size=atom_vocab_size)


def forward(model, head: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    desc = extract_descriptors(
        model,
        batch["input_atomic_numbers"],
        batch["coords_noisy"],
        batch["atom_padding"],
    )
    out = dict(desc)
    out.update(head(desc))
    return out


TASK_SPEC = TaskSpec(
    name="atom_mask",
    adapter_cls=AtomMaskAdapter,
    build_head=build_head,
    forward=forward,
)
