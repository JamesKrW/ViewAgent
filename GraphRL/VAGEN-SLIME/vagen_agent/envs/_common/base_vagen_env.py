"""Shared implementation for VAGEN's built-in Harness-facing environments."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Mapping
from numbers import Real
from typing import Any, final

from vagen_agent.envs._common.base_env import BaseEnv, Obs, StepResult
from vagen_agent.envs._common.utils.observation import validate_observation


class BaseVagenEnv(BaseEnv):
    """Implement VAGEN invariants while leaving task behavior to subclasses.

    Concrete environments implement only ``_reset``, ``_system_prompt`` and ``_step``.
    Harness-specific base classes are independent capability markers above this class.
    """

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        self.env_config = dict(env_config or {})
        self.max_turns: int | None = None
        self.turns_taken = 0
        self.reset_info: dict[str, Any] = {}
        self.last_metrics: dict[str, float] = {}
        self.success = False
        self._active = False

    @final
    def configure(self, *, max_turns: int | None = None) -> None:
        if max_turns is not None and int(max_turns) <= 0:
            raise ValueError(f"max_turns must be positive, got {max_turns!r}")
        self.max_turns = None if max_turns is None else int(max_turns)

    @final
    async def reset(self, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        result = await self._reset(seed)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("_reset() must return (observation, info)")
        observation, raw_info = result
        info = self._info(raw_info, source="_reset()")
        self.turns_taken = 0
        self.reset_info = info
        self.last_metrics = {}
        self.success = False
        self._active = True
        return validate_observation(observation, source="_reset()"), info

    @abstractmethod
    async def _reset(self, seed: int | None) -> tuple[Obs, Mapping[str, Any] | None]:
        """Start one deterministic episode."""

    @final
    async def system_prompt(self) -> Obs:
        return validate_observation(
            await self._system_prompt(),
            source="_system_prompt()",
        )

    @abstractmethod
    async def _system_prompt(self) -> Obs:
        """Return the standing task and action instructions."""

    @final
    async def step(self, action: Any) -> StepResult:
        if not self._active:
            raise RuntimeError("reset() must be called before step()")
        result = await self._step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError(
                "_step() must return "
                "(observation, reward, terminated, truncated, info)"
            )
        observation, reward, terminated, truncated, raw_info = result
        info = self._info(raw_info, source="_step()")
        terminated = bool(terminated)
        truncated = bool(truncated)
        self.turns_taken += 1
        if (
            self.max_turns is not None
            and self.turns_taken >= self.max_turns
            and not terminated
            and not truncated
        ):
            truncated = True
            info = {**info, "truncated": True}
        self._remember_metrics(info)
        self._remember_success(info)
        if terminated or truncated:
            self._active = False
        return (
            validate_observation(observation, source="_step()"),
            reward,
            terminated,
            truncated,
            info,
        )

    @abstractmethod
    async def _step(self, action: Any) -> StepResult:
        """Apply task-specific action semantics."""

    async def close(self) -> None:
        self._active = False

    @staticmethod
    def _info(value: Mapping[str, Any] | None, *, source: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"{source} info must be a mapping")
        return dict(value)

    def _remember_success(self, info: Mapping[str, Any]) -> None:
        for key in ("success", "is_success"):
            if key in info:
                self.success = bool(info[key])
                return
        if "success" in self.last_metrics:
            self.success = bool(self.last_metrics["success"])

    def _remember_metrics(self, info: Mapping[str, Any]) -> None:
        raw = info.get("metrics")
        if raw is None:
            return
        if not isinstance(raw, Mapping):
            raise TypeError("info['metrics'] must be a flat mapping of numeric values")
        snapshot: dict[str, float] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"invalid environment metric name: {key!r}")
            if not isinstance(value, Real):
                raise TypeError(
                    f"environment metric {key!r} is {type(value).__name__}; "
                    "expected a number"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    f"environment metric {key!r} is not finite: {number!r}"
                )
            snapshot[key] = number
        self.last_metrics = snapshot


__all__ = ["BaseVagenEnv"]
