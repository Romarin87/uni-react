"""Private low-level PyTorch layers used only by encoders.

Nothing outside the ``encoders/`` package should import from here directly.
"""
from .common import (
    BiasedAttention,
    BaisedAttention,  # backward-compat typo alias
    DropPath,
    NonLinear,
    RBFEmb,
    UniMolLayer,
    safe_normalization,
)
from .se3 import EquiOutput, FCEqMPLayer, FCSVec, FTE, GatedEquivariantBlock
from .tensor_utils import create_access_mask, create_attn_mask
from .cg_tensor import (
    spherical_harmonics_l1, spherical_harmonics_l2, spherical_harmonics,
    cg_1x1_to_0, cg_1x1_to_1, cg_1x1_to_2, EdgeIrrepEncoding,
)
from .three_body import ThreeBodyAggregation
from .so2_conv import SO2MessageBlock, rotation_from_edge, wigner_d_l2_from_R
from .e3nn_utils import CGMessageBlock, sh_l1, sh_l2, wigner_d2

__all__ = [
    # original layers
    "RBFEmb", "NonLinear", "DropPath",
    "BiasedAttention", "BaisedAttention",
    "UniMolLayer", "safe_normalization",
    "FTE", "GatedEquivariantBlock", "EquiOutput", "FCSVec", "FCEqMPLayer",
    "create_access_mask", "create_attn_mask",
    # equivariant layers (custom)
    "spherical_harmonics_l1", "spherical_harmonics_l2", "spherical_harmonics",
    "cg_1x1_to_0", "cg_1x1_to_1", "cg_1x1_to_2", "EdgeIrrepEncoding",
    "ThreeBodyAggregation",
    "SO2MessageBlock", "rotation_from_edge", "wigner_d_l2_from_R",
    # e3nn-backed equivariant layers
    "CGMessageBlock", "sh_l1", "sh_l2", "wigner_d2",
]
