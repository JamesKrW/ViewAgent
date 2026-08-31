"""Model-family adapters. Importing this package registers the built-ins."""

from vagen_agent.models import (
    qwen as qwen,  # noqa: E402  (import for registration side effect)
)
from vagen_agent.models._common import (
    MODEL_ADAPTERS,
    ModelAdapter,
    ModelAdapterError,
    build_model_adapter,
    register_model,
    resolve_model_adapter,
)

__all__ = ["ModelAdapter", "ModelAdapterError", "MODEL_ADAPTERS", "register_model",
           "resolve_model_adapter", "build_model_adapter"]
