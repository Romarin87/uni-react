"""Encoder (backbone) implementations + composed model classes.

All encoders are registered in ENCODER_REGISTRY.

Public API
----------
``SingleMolEncoder``      – SE3 equivariant backbone (registered as ``"single_mol"``)
``SingleMolPretrainNet``  – backbone + geometric/electronic pretraining heads
``QM9FineTuneNet``        – backbone + QM9 regression head
``SingleMolDescriptor``   – backward-compat alias for SingleMolEncoder
"""
from .single_mol import SingleMolDescriptor, SingleMolEncoder
from .density_model import DensityPretrainNet, QueryPointDensityHead
from .pretrain_model import SingleMolPretrainNet
from .qm9_model import QM9FineTuneNet
from .reaction_model import ReactionPretrainNet
from .reacformer_se3 import ReacFormerSE3Encoder
from .reacformer_so2 import ReacFormerSO2Encoder

__all__ = [
    "SingleMolEncoder",
    "SingleMolDescriptor",
    "DensityPretrainNet",
    "QueryPointDensityHead",
    "SingleMolPretrainNet",
    "QM9FineTuneNet",
    "ReactionPretrainNet",
    "ReacFormerSE3Encoder",
    "ReacFormerSO2Encoder",
]
