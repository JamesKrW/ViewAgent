"""Registry for standalone-evaluation model backends."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

BACKENDS: dict[str, type] = {}


def register_backend(*names: str) -> Callable[[type], type]:
    """Register a backend class under one or more configuration names."""

    def decorator(cls: type) -> type:
        _require_backend(cls, ", ".join(names))
        for name in names:
            if not name:
                raise ValueError("backend names must be non-empty strings")
            existing = BACKENDS.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(f"backend {name!r} is already registered")
            BACKENDS[name] = cls
        return cls

    return decorator


def resolve_backend(name: str) -> type:
    """Resolve a registered name or a ``module:Class`` import path."""

    if name in BACKENDS:
        return BACKENDS[name]
    if "." not in name and ":" not in name:
        raise ValueError(
            f"unknown evaluation backend {name!r}; choose from {sorted(BACKENDS)}."
        )
    module_name, _, attr = name.rpartition(":") if ":" in name else name.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"could not import evaluation backend {name!r}: {exc}") from exc
    _require_backend(cls, name)
    return cls


def build_backend(spec: Any):
    return resolve_backend(spec.backend)(spec)


def registered_backends() -> tuple[str, ...]:
    return tuple(sorted(BACKENDS))


def _require_backend(cls: type, name: str) -> None:
    if not isinstance(cls, type) or not callable(getattr(cls, "complete", None)):
        raise TypeError(f"{name} resolved to {cls!r}, which is not an evaluation backend")
    if not callable(getattr(cls, "close", None)):
        raise TypeError(f"{name} has no async close method")


__all__ = [
    "BACKENDS",
    "build_backend",
    "register_backend",
    "registered_backends",
    "resolve_backend",
]
