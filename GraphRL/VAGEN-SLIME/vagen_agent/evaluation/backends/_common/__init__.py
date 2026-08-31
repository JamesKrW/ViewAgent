"""Shared contracts and registry for evaluation backends."""

from vagen_agent.evaluation.backends._common.base import BackendReply, EvaluationBackend
from vagen_agent.evaluation.backends._common.registry import (
    BACKENDS,
    build_backend,
    register_backend,
    registered_backends,
    resolve_backend,
)

__all__ = [
    "BACKENDS",
    "BackendReply",
    "EvaluationBackend",
    "build_backend",
    "register_backend",
    "registered_backends",
    "resolve_backend",
]
