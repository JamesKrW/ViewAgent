"""Contracts and shared reward transformations for algorithm variants."""

from vagen_agent.algorithms._common.spec import (
    ALGORITHMS,
    AlgorithmSpec,
    RewardFolding,
    register_algorithm,
    registered_algorithms,
    resolve_algorithm,
)

__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "RewardFolding",
    "register_algorithm",
    "registered_algorithms",
    "resolve_algorithm",
]
