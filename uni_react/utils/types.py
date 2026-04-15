"""Shared type aliases used across the uni_react package."""
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

# Filesystem paths accepted everywhere in the codebase.
PathLike = Union[str, Path]

# Convenience re-exports so callers only need to import from here.
__all__ = [
    "PathLike",
]
