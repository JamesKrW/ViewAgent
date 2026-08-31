"""VAGEN-SLIME adapters for the environments implemented by ViewSuite.

The environments in :mod:`view_suite.envs` expose the legacy four-value
``(observation, reward, done, info)`` API.  This module is the single bridge to
VAGEN-SLIME's harness-facing API; environment behavior remains in ViewSuite.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

from vagen_agent.envs import register_env
from vagen_agent.envs._common import (
    BaseCompactEnv,
    BaseConcatEnv,
    BaseNoConcatEnv,
    Obs,
    StepResult,
)


def _action_text(action: Any) -> str:
    """Return model text from either a rollout/evaluation response or a string."""

    text = getattr(action, "text", None)
    return text if isinstance(text, str) else str(action)


def _normalized_info(raw_info: Any) -> dict[str, Any]:
    """Expose legacy step results through VAGEN-SLIME's metrics channel.

    Keep an environment's explicit ``metrics`` mapping authoritative.  When a
    legacy environment only publishes top-level numeric values, promote those
    values so the SLIME metric collector can still see them.  The historical
    ``traj_success`` spelling is added by the GraphRL logging seam rather than
    mutating every environment's native metric set here.
    """

    info = dict(raw_info or {}) if isinstance(raw_info, Mapping) else {}
    metrics = dict(info.get("metrics") or {})
    for key, value in info.items():
        if key != "metrics" and isinstance(key, str) and isinstance(value, Real):
            metrics.setdefault(key, float(value))

    success = bool(info.get("success", info.get("traj_success", False)))
    info["success"] = success
    # Keep the historical alias because adaptive configs and W&B panels use it.
    metrics.setdefault("success", float(success))
    info["metrics"] = metrics
    return info


def resolve_legacy_data_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve dataset names used by older checked-in examples.

    The final ViewSuite corpora use ``*_train.jsonl``, ``*_dev.jsonl`` and
    ``*_test.jsonl``.  Several historical entrypoints still say
    ``*_filter.jsonl`` for the same already-filtered files.  Keep those entrypoints
    runnable without rewriting their experiment definitions, while never masking a
    genuinely missing dataset: the fallback is used only when the requested path is
    absent and the canonical sibling exists.
    """

    raw_path = config.get("jsonl_path")
    if not isinstance(raw_path, str) or not raw_path.endswith("_filter.jsonl"):
        return config
    requested = Path(raw_path).expanduser()
    canonical = requested.with_name(requested.name.replace("_filter.jsonl", ".jsonl"))
    if not requested.is_file() and canonical.is_file():
        config["jsonl_path"] = str(canonical)
    return config


class ViewSuiteSlimeEnv(BaseNoConcatEnv, BaseConcatEnv, BaseCompactEnv):
    """Delegate one ViewSuite environment through all stock SLIME harnesses."""

    delegate_class: ClassVar[type]

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        super().__init__(env_config)
        self._delegate: Any | None = None

    def _require_delegate(self) -> Any:
        if self._delegate is None:
            raise RuntimeError("reset() must be called before using the environment")
        return self._delegate

    async def _reset(self, seed: int | None) -> tuple[Obs, Mapping[str, Any] | None]:
        config = resolve_legacy_data_paths(dict(self.env_config))
        if self.max_turns is not None:
            config["max_turns"] = self.max_turns
        self._delegate = self.delegate_class(config)
        observation, info = await self._delegate.reset(
            seed=0 if seed is None else int(seed)
        )
        # Reset metadata is not an episode result and must not leak into
        # ``last_metrics``.  BaseVagenEnv clears metrics after this return.
        return observation, dict(info or {})

    async def _system_prompt(self) -> Obs:
        return await self._require_delegate().system_prompt()

    async def _step(self, action: Any) -> StepResult:
        observation, reward, done, raw_info = await self._require_delegate().step(
            _action_text(action)
        )
        info = _normalized_info(raw_info)
        truncated = bool(info.get("truncated", False))
        terminated = bool(done and not truncated)
        return observation, reward, terminated, truncated, info

    async def close(self) -> None:
        try:
            if self._delegate is not None:
                await self._delegate.close()
        finally:
            self._delegate = None
            await super().close()


def adapt_legacy_env(name: str, delegate_class: type) -> type[ViewSuiteSlimeEnv]:
    """Create a named SLIME capability wrapper for a legacy async environment."""

    return type(
        name,
        (ViewSuiteSlimeEnv,),
        {"delegate_class": delegate_class, "__module__": __name__},
    )


from view_suite.envs.ai2thor_proxy_task.interactive_view_planning import (  # noqa: E402
    Ai2ThorInteractiveViewPlanning as _Ai2ThorInteractiveViewPlanning,
)
from view_suite.envs.ai2thor_proxy_task.path_to_view import (  # noqa: E402
    Ai2ThorPath2View as _Ai2ThorPath2View,
)
from view_suite.envs.ai2thor_proxy_task.view_to_path import (  # noqa: E402
    Ai2ThorView2Path as _Ai2ThorView2Path,
)
from view_suite.envs.habitat_gs_proxy_task.interactive_view_planning import (  # noqa: E402
    HabitatGSInteractiveViewPlanning as _HabitatGSInteractiveViewPlanning,
)
from view_suite.envs.habitat_gs_proxy_task.path_to_view import (  # noqa: E402
    HabitatGSPath2View as _HabitatGSPath2View,
)
from view_suite.envs.habitat_gs_proxy_task.view_to_path import (  # noqa: E402
    HabitatGSView2Path as _HabitatGSView2Path,
)
from view_suite.envs.scannet_proxy_task.interactive_view_planning import (  # noqa: E402
    InteractiveViewPlanning as _InteractiveViewPlanning,
)
from view_suite.envs.scannet_proxy_task.path_to_view import Path2View as _Path2View  # noqa: E402
from view_suite.envs.scannet_proxy_task.view_to_path import View2Path as _View2Path  # noqa: E402


Path2View = adapt_legacy_env("Path2View", _Path2View)
View2Path = adapt_legacy_env("View2Path", _View2Path)
InteractiveViewPlanning = adapt_legacy_env("InteractiveViewPlanning", _InteractiveViewPlanning)
Ai2ThorPath2View = adapt_legacy_env("Ai2ThorPath2View", _Ai2ThorPath2View)
Ai2ThorView2Path = adapt_legacy_env("Ai2ThorView2Path", _Ai2ThorView2Path)
Ai2ThorInteractiveViewPlanning = adapt_legacy_env(
    "Ai2ThorInteractiveViewPlanning", _Ai2ThorInteractiveViewPlanning
)
HabitatGSPath2View = adapt_legacy_env("HabitatGSPath2View", _HabitatGSPath2View)
HabitatGSView2Path = adapt_legacy_env("HabitatGSView2Path", _HabitatGSView2Path)
HabitatGSInteractiveViewPlanning = adapt_legacy_env(
    "HabitatGSInteractiveViewPlanning", _HabitatGSInteractiveViewPlanning
)


_ENVIRONMENTS = {
    cls.__name__: cls
    for cls in (
        Path2View,
        View2Path,
        InteractiveViewPlanning,
        Ai2ThorPath2View,
        Ai2ThorView2Path,
        Ai2ThorInteractiveViewPlanning,
        HabitatGSPath2View,
        HabitatGSView2Path,
        HabitatGSInteractiveViewPlanning,
    )
}


def register_viewsuite_envs() -> None:
    """Register every ViewSuite task used by shipped training/eval configs."""

    for name, env_class in _ENVIRONMENTS.items():
        register_env(name, env_class)


__all__ = [
    *_ENVIRONMENTS,
    "ViewSuiteSlimeEnv",
    "adapt_legacy_env",
    "register_viewsuite_envs",
    "resolve_legacy_data_paths",
]
