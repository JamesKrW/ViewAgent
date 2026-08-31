"""Algorithm declarations and the registry shared by all algorithm variants."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

RewardFolding = Literal["episode_total", "token_suffix", "identity"]


@dataclass(frozen=True)
class AlgorithmSpec:
    """How one VAGEN algorithm maps onto Slime's training interfaces."""

    slime_estimator: str
    custom_advantage_path: str | None = None
    needs_critic: bool = False
    spans_rows: bool = False
    requires_undiscounted: bool = False
    needs_value_mask: bool = False
    per_token_reward: bool = False
    reward_folding: RewardFolding = "identity"
    note: str = ""


ALGORITHMS: dict[str, AlgorithmSpec] = {}


def register_algorithm(name: str, spec: AlgorithmSpec) -> AlgorithmSpec:
    """Register one named algorithm without allowing silent replacement."""

    if not name or not isinstance(name, str):
        raise ValueError("algorithm name must be a non-empty string")
    if spec.reward_folding not in {"episode_total", "token_suffix", "identity"}:
        raise ValueError(f"algorithm {name!r} has unknown reward folding {spec.reward_folding!r}")
    existing = ALGORITHMS.get(name)
    if existing is not None and existing != spec:
        raise ValueError(f"algorithm {name!r} is already registered")
    ALGORITHMS[name] = spec
    return spec


def resolve_algorithm(name: str) -> AlgorithmSpec:
    """Resolve a registered name or a ``module:SPEC`` import path."""

    if name in ALGORITHMS:
        return ALGORITHMS[name]
    if "." not in name and ":" not in name:
        raise ValueError(
            f"unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}."
        )
    module_name, _, attr = name.rpartition(":") if ":" in name else name.rpartition(".")
    try:
        spec = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"could not import algorithm {name!r}: {exc}") from exc
    if not isinstance(spec, AlgorithmSpec):
        raise TypeError(f"{name} resolved to {spec!r}, which is not an AlgorithmSpec")
    return spec


def registered_algorithms() -> tuple[str, ...]:
    return tuple(sorted(ALGORITHMS))


__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "RewardFolding",
    "register_algorithm",
    "registered_algorithms",
    "resolve_algorithm",
]
