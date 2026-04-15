"""Geometric-structure pretraining loss."""
from typing import Dict, Tuple

import torch
from torch import Tensor

from ..registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("geometric_structure")
class GeometricStructureLoss:
    """Weighted sum of atom-mask CE, coord-denoise MSE, and charge MSE.

    Config example::

        loss:
          type: geometric_structure
          atom_weight: 1.0
          coord_weight: 1.0
          charge_weight: 1.0
    """

    def __init__(
        self,
        atom_weight: float = 1.0,
        coord_weight: float = 1.0,
        charge_weight: float = 1.0,
    ) -> None:
        self.atom_weight = float(atom_weight)
        self.coord_weight = float(coord_weight)
        self.charge_weight = float(charge_weight)

    def metric_keys(self) -> Tuple[str, ...]:
        return ("loss", "atom_loss", "coord_loss", "charge_loss")

    def __call__(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute the geometric-structure loss.

        Expects *outputs* to contain (when available):
        - ``atom_logits``      – from atom-mask head
        - ``coords_denoised``  – from coord-denoise head
        - ``charge_pred``      – from charge head

        Returns:
            Dict with keys: ``loss``, ``atom_loss``, ``coord_loss``, ``charge_loss``.
        """
        zero = self._zero(outputs)

        atom_loss = zero
        if "atom_logits" in outputs and (
            "target_atomic_numbers" in batch or "atomic_numbers" in batch
        ):
            atom_loss = self._atom_loss(outputs, batch)

        coord_loss = zero
        if "coords_denoised" in outputs and (
            "coords_target" in batch or "coords" in batch
        ):
            coord_loss = self._coord_loss(outputs, batch)

        charge_loss = zero
        if "charge_pred" in outputs and "charges" in batch:
            charge_loss = self._charge_loss(outputs, batch)

        total = (
            self.atom_weight * atom_loss
            + self.coord_weight * coord_loss
            + self.charge_weight * charge_loss
        )
        return {
            "loss": total,
            "atom_loss": atom_loss,
            "coord_loss": coord_loss,
            "charge_loss": charge_loss,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero(outputs: Dict[str, Tensor]) -> Tensor:
        for v in outputs.values():
            if isinstance(v, Tensor):
                return v.sum() * 0.0
        return torch.tensor(0.0)

    @staticmethod
    def _atom_loss(outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        import torch.nn.functional as F
        import warnings
        
        logits = outputs["atom_logits"]   # (B, N, vocab)
        targets = batch.get("target_atomic_numbers", batch["atomic_numbers"])  # (B, N)
        mask = batch.get("mask", batch.get("mask_positions", None))    # (B, N) bool, True = masked position
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            num_masked = mask_flat.sum()
            
            # Validate that we have at least one masked position
            if num_masked == 0:
                warnings.warn(
                    "No masked positions found in batch for atom_loss, returning zero loss. "
                    "This may indicate a data pipeline issue.",
                    RuntimeWarning,
                    stacklevel=3
                )
                return loss.sum() * 0.0
            
            return (loss * mask_flat).sum() / num_masked
        return loss.mean()

    @staticmethod
    def _coord_loss(outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        import torch.nn.functional as F
        import warnings
        
        pred = outputs["coords_denoised"]   # (B, N, 3)
        target = batch.get("coords_target", batch["coords"])     # (B, N, 3)
        pad = batch.get("atom_padding", None)  # (B, N) True = pad
        diff = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # (B, N)
        if pad is not None:
            valid = (~pad).float()
            num_valid = valid.sum()
            
            # Validate that we have at least one valid atom
            if num_valid == 0:
                warnings.warn(
                    "No valid atoms found in batch for coord_loss, returning zero loss. "
                    "This may indicate a data pipeline issue.",
                    RuntimeWarning,
                    stacklevel=3
                )
                return diff.sum() * 0.0
            
            return (diff * valid).sum() / num_valid
        return diff.mean()

    @staticmethod
    def _charge_loss(outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        import torch.nn.functional as F
        import warnings
        
        pred = outputs["charge_pred"].squeeze(-1)  # (B, N)
        target = batch["charges"]                  # (B, N)
        pad = batch.get("atom_padding", None)
        loss = F.mse_loss(pred, target, reduction="none")  # (B, N)
        if pad is not None:
            valid = (~pad).float()
            num_valid = valid.sum()
            
            # Validate that we have at least one valid atom
            if num_valid == 0:
                warnings.warn(
                    "No valid atoms found in batch for charge_loss, returning zero loss. "
                    "This may indicate a data pipeline issue.",
                    RuntimeWarning,
                    stacklevel=3
                )
                return loss.sum() * 0.0
            
            return (loss * valid).sum() / num_valid
        return loss.mean()
