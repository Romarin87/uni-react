"""Generic component registry with decorator-based registration."""
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A name → class mapping with optional factory support.

    Usage
    -----
    Register a class::

        ENCODER_REGISTRY = Registry("encoder")

        @ENCODER_REGISTRY.register("single_mol")
        class SingleMolDescriptor:
            def __init__(self, emb_dim: int, ...):
                ...

    Build an instance from a config dict::

        encoder = ENCODER_REGISTRY.build({"type": "single_mol", "emb_dim": 256})

    Retrieve the class without instantiating::

        cls = ENCODER_REGISTRY.get("single_mol")

    The ``"type"`` key is consumed by :meth:`build` and never forwarded to
    ``__init__``.  All other keys are passed as keyword arguments.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Type] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, alias: str) -> Callable[[Type[T]], Type[T]]:
        """Class decorator that registers *cls* under *alias*.

        Args:
            alias: The lookup key used in config files / ``build`` calls.

        Returns:
            The original class (unmodified) so the decorator is transparent.

        Raises:
            KeyError: If *alias* is already registered.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if alias in self._registry:
                raise KeyError(
                    f"[{self._name} registry] '{alias}' is already registered "
                    f"(existing: {self._registry[alias].__qualname__}). "
                    "Use a different alias or unregister the old entry first."
                )
            self._registry[alias] = cls
            return cls
        return decorator

    def register_or_replace(self, alias: str) -> Callable[[Type[T]], Type[T]]:
        """Like :meth:`register` but silently overwrites an existing entry.

        Useful when monkey-patching during tests.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self._registry[alias] = cls
            return cls
        return decorator

    # ------------------------------------------------------------------
    # Lookup & instantiation
    # ------------------------------------------------------------------

    def get(self, alias: str) -> Type:
        """Return the class registered under *alias*.

        Raises:
            KeyError: With a helpful message listing available aliases.
        """
        if alias not in self._registry:
            available = sorted(self._registry)
            raise KeyError(
                f"[{self._name} registry] '{alias}' not found. "
                f"Available: {available}"
            )
        return self._registry[alias]

    def build(self, cfg: Dict[str, Any], **extra_kwargs) -> Any:
        """Instantiate the class registered under ``cfg['type']``.

        Args:
            cfg: Config dict.  Must contain a ``"type"`` key whose value is a
                registered alias.  All other keys are forwarded to ``__init__``.
            **extra_kwargs: Additional kwargs merged into *cfg* (take precedence).

        Returns:
            A new instance of the registered class.

        Raises:
            KeyError: If ``cfg['type']`` is not registered.
            ValueError: If *cfg* does not contain a ``"type"`` key.
        """
        cfg = dict(cfg)  # shallow copy – avoid mutating caller's dict
        alias = cfg.pop("type", None)
        if alias is None:
            raise ValueError(
                f"[{self._name} registry] 'type' key is missing from config: {cfg}"
            )
        cfg.update(extra_kwargs)
        cls = self.get(alias)
        return cls(**cfg)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def list_aliases(self) -> list:
        """Return a sorted list of all registered aliases."""
        return sorted(self._registry)

    def __contains__(self, alias: str) -> bool:
        return alias in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, entries={self.list_aliases()})"

    def __len__(self) -> int:
        return len(self._registry)
