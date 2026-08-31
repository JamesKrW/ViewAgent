"""Standalone-evaluation backend extension axis."""

from vagen_agent.evaluation.backends._common import (
    BACKENDS,
    BackendReply,
    EvaluationBackend,
    build_backend,
    register_backend,
    registered_backends,
    resolve_backend,
)
from vagen_agent.evaluation.backends.openai import OpenAIChatBackend, render_message

__all__ = [
    "BACKENDS",
    "BackendReply",
    "EvaluationBackend",
    "OpenAIChatBackend",
    "build_backend",
    "register_backend",
    "registered_backends",
    "render_message",
    "resolve_backend",
]
