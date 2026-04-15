"""Logging backend implementations.

All loggers are registered in :data:`~uni_react.registry.LOGGER_REGISTRY`.

Available loggers
-----------------
``console``     :class:`~uni_react.loggers.console.ConsoleLogger`  (default)
``wandb``       :class:`~uni_react.loggers.wandb.WandbLogger`
``tensorboard`` :class:`~uni_react.loggers.tensorboard.TensorBoardLogger`

Adding a new logger
-------------------
1. Create ``uni_react/loggers/my_logger.py`` satisfying
   :class:`~uni_react.core.logger.LoggerProtocol`.
2. Decorate with ``@LOGGER_REGISTRY.register("my_logger")``.
3. Import it below.
"""
from .console import ConsoleLogger
from .wandb import WandbLogger
from .tensorboard import TensorBoardLogger

__all__ = [
    "ConsoleLogger",
    "WandbLogger",
    "TensorBoardLogger",
]
