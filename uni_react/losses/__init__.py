"""Loss-function implementations.

All losses are registered in :data:`~uni_react.registry.LOSS_REGISTRY`.

Available losses
----------------
``geometric_structure``
    :class:`~uni_react.losses.geometric.GeometricStructureLoss`
``electronic_structure``
    :class:`~uni_react.losses.electronic.ElectronicStructureLoss`
``qm9_regression``
    :class:`~uni_react.losses.qm9.QM9RegressionLoss`

Adding a new loss
-----------------
1. Create ``uni_react/losses/my_loss.py`` satisfying
   :class:`~uni_react.core.loss.LossFnProtocol`.
2. Decorate with ``@LOSS_REGISTRY.register("my_loss")``.
3. Import it below.
"""
from .electronic import ElectronicStructureLoss
from .geometric import GeometricStructureLoss
from .qm9 import QM9RegressionLoss

__all__ = [
    "GeometricStructureLoss",
    "ElectronicStructureLoss",
    "QM9RegressionLoss",
]
