# Dynamic environment registry that loads from env_registry.yaml
from __future__ import annotations

import importlib
import logging
import os

import yaml

from vagen_agent.envs._common import (
    BaseCompactEnv,
    BaseConcatEnv,
    BaseEnv,
    BaseNoConcatEnv,
    BaseVagenEnv,
    EnvAction,
    Obs,
    Reward,
    StepResult,
)

logger = logging.getLogger(__name__)

_ENV_REGISTRY: dict[str, type] = {}
_FAILED_ENVS: dict[str, str] = {}
_LOADED = False


def _load_registry():
    global _LOADED
    if _LOADED:
        return

    # Find env_registry.yaml relative to this file
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "env_registry.yaml"
    )
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"env_registry.yaml not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_registry = config.get("env_registry", {})

    for env_name, module_path in env_registry.items():
        try:
            module_name, class_name = module_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            env_cls = getattr(module, class_name)
            _require_env_class(env_name, env_cls)
            _ENV_REGISTRY[env_name] = env_cls
        except Exception as e:  # noqa: BLE001 - one optional env must not hide the rest
            _FAILED_ENVS[env_name] = str(e)
            logger.warning(f"Failed to load env '{env_name}' from '{module_path}': {e}")

    _LOADED = True


def get_env_cls(name: str) -> type:
    """Resolve environment class from env_registry.yaml."""
    _load_registry()
    if name not in _ENV_REGISTRY:
        raise KeyError(f"Unknown env name: {name}. Available: {list(_ENV_REGISTRY.keys())}")
    return _ENV_REGISTRY[name]


def register_env(name: str, env_cls: type) -> None:
    """Manually register an environment class."""
    _load_registry()
    _require_env_class(name, env_cls)
    _ENV_REGISTRY[name] = env_cls


def _require_env_class(name: str, env_cls: type) -> None:
    if not isinstance(env_cls, type) or not issubclass(env_cls, BaseEnv):
        raise TypeError(f"environment {name!r} must inherit BaseEnv")


def build_env(
    env_cls: type[BaseEnv],
    env_config: dict | None,
    max_turns: int | None = None,
    *,
    required_type: type[BaseEnv] = BaseEnv,
) -> BaseEnv:
    """Construct and validate one direct Harness-facing environment."""
    env = env_cls(env_config=dict(env_config or {}))
    if not isinstance(env, required_type):
        raise TypeError(
            f"environment {env_cls.__name__} does not support "
            f"{required_type.__name__}"
        )
    env.configure(max_turns=max_turns)
    return env


def list_envs() -> list:
    """List all registered environment names."""
    _load_registry()
    return list(_ENV_REGISTRY.keys())


__all__ = [
    "BaseCompactEnv",
    "BaseConcatEnv",
    "BaseEnv",
    "BaseNoConcatEnv",
    "BaseVagenEnv",
    "EnvAction",
    "Obs",
    "Reward",
    "StepResult",
    "build_env",
    "get_env_cls",
    "list_envs",
    "register_env",
]
