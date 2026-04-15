"""Component registry for uni_react.

All registries are module-level singletons.  Import them directly::

    from uni_react.registry import ENCODER_REGISTRY, HEAD_REGISTRY

Register a new implementation with the decorator::

    @ENCODER_REGISTRY.register("my_encoder")
    class MyEncoder: ...

Build an instance from a config dict::

    encoder = ENCODER_REGISTRY.build({"type": "my_encoder", "emb_dim": 256})
"""
from .registry import Registry

# ---------------------------------------------------------------------------
# Global registry singletons – one per abstraction layer
# ---------------------------------------------------------------------------
ENCODER_REGISTRY: Registry = Registry("encoder")
HEAD_REGISTRY: Registry = Registry("head")
LOSS_REGISTRY: Registry = Registry("loss")
LOGGER_REGISTRY: Registry = Registry("logger")
SCHEDULER_REGISTRY: Registry = Registry("scheduler")
TRANSFORM_REGISTRY: Registry = Registry("transform")

__all__ = [
    "Registry",
    "ENCODER_REGISTRY",
    "HEAD_REGISTRY",
    "LOSS_REGISTRY",
    "LOGGER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "TRANSFORM_REGISTRY",
]
