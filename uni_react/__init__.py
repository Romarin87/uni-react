"""Top-level package exports with lazy loading.

Keep lightweight tools importable without immediately importing torch-heavy
encoder modules.
"""

from importlib import import_module
from typing import Any

__all__ = ["SingleMolPretrainNet", "QM9FineTuneNet"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        encoders = import_module(".encoders", __name__)
        return getattr(encoders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
