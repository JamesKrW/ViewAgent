"""Context policies, by name.

A registry rather than an import in the trainer: which policy a run uses is a config
value, and adding one should not mean editing anything else.

Two ways to reach a policy that is not built in::

    @register_harness("mine")            # then `harness: mine`
    class MyHarness(BaseHarness): ...

    harness: mypkg.harnesses:MyHarness   # an import path, with nothing to register

The import path is there because a new policy is often tried from a config first, and a
config is a yaml rather than a package -- there is nowhere to put a decorator that would
have run by then.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from vagen_agent.harness._common import (
    BaseHarness,
    MissingModelRole,
    Msg,
    RoleClients,
    assistant,
    client_for_role,
    obs_to_message,
    user,
)
from vagen_agent.harness.compact import CompactHarness
from vagen_agent.harness.concat import ConcatHarness
from vagen_agent.harness.no_concat import NoConcatHarness

HARNESSES: dict[str, type[BaseHarness]] = {
    "concat": ConcatHarness,
    "no_concat": NoConcatHarness,
    "compact": CompactHarness,
}


def register_harness(name: str) -> Callable[[type], type]:
    """Register a ``BaseHarness`` subclass under ``name``.

    Refuses to overwrite a *different* class already holding the name: a silent rebinding
    means a run reports the policy it was configured with and executes another one.
    Re-registering the same class is fine -- a module is legitimately imported more than
    once.
    """
    def decorator(cls: type) -> type:
        _require_harness(cls, name)
        existing = HARNESSES.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"harness {name!r} is already registered to {existing.__qualname__}; "
                f"pick another name rather than shadowing it.")
        HARNESSES[name] = cls
        return cls
    return decorator


def resolve_harness(name: str) -> type[BaseHarness]:
    """The class for ``name``: a registered name, or a ``module:Class`` import path."""
    if name in HARNESSES:
        return HARNESSES[name]
    if "." not in name and ":" not in name:
        raise ValueError(
            f"unknown harness {name!r}; choose from {sorted(HARNESSES)}, or give an "
            f"import path like 'mypkg.harnesses:MyHarness'.")
    module_name, _, attr = name.rpartition(":") if ":" in name else name.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"could not import harness {name!r}: {exc}") from exc
    _require_harness(cls, name)
    return cls


def _require_harness(cls, name: str) -> None:
    if not (isinstance(cls, type) and issubclass(cls, BaseHarness)):
        raise TypeError(
            f"{name} resolved to {cls!r}, which does not subclass BaseHarness -- so "
            f"run_episode is not guaranteed. Checked here, at registration, rather than "
            f"at the first call.")
    roles = tuple(cls.model_roles)
    if not roles or any(not isinstance(role, str) or not role for role in roles):
        raise TypeError(f"{name} must declare non-empty string model_roles")
    if len(set(roles)) != len(roles):
        raise TypeError(f"{name} declares duplicate model_roles {list(roles)}")
    if cls.action_model_role not in roles:
        raise TypeError(
            f"{name}.action_model_role={cls.action_model_role!r} is not one of its "
            f"model_roles {list(roles)}"
        )
    environment_type = cls.environment_type
    if not (
        isinstance(environment_type, type)
        and issubclass(environment_type, BaseHarness.environment_type)
    ):
        raise TypeError(f"{name}.environment_type must inherit BaseEnv")


def build_harness(name: str, **kwargs) -> BaseHarness:
    return resolve_harness(name)(**kwargs)


__all__ = [
    "HARNESSES",
    "BaseHarness",
    "CompactHarness",
    "ConcatHarness",
    "MissingModelRole",
    "Msg",
    "NoConcatHarness",
    "RoleClients",
    "assistant",
    "build_harness",
    "client_for_role",
    "obs_to_message",
    "register_harness",
    "resolve_harness",
    "user",
]
