"""Expose ViewAgent's Habitat-GS task through VAGEN-SLIME's environment contract.

The task remains in :mod:`view_suite`.  This adapter only translates its legacy
four-value ``step`` result into Gymnasium termination semantics, keeping all
environment-specific behavior outside the vendored training backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from view_suite.envs.habitat_gs_proxy_task.interactive_view_planning import (
    HabitatGSInteractiveViewPlanning as _LegacyHabitatGSInteractiveViewPlanning,
)

from vagen_agent.envs._common import BaseNoConcatEnv, Obs, StepResult


class HabitatGSInteractiveViewPlanning(BaseNoConcatEnv):
    """Expose the existing self-contained IVP task to the no-concat harness."""

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        super().__init__(env_config)
        self._delegate: _LegacyHabitatGSInteractiveViewPlanning | None = None

    def _require_delegate(self) -> _LegacyHabitatGSInteractiveViewPlanning:
        if self._delegate is None:
            raise RuntimeError("reset() must be called before using the Habitat-GS task")
        return self._delegate

    async def _reset(self, seed: int | None) -> tuple[Obs, Mapping[str, Any] | None]:
        # BaseVagenEnv.configure() runs after construction. Delay creation so its
        # framework-owned max_turns is also the limit used by the legacy task itself.
        config = dict(self.env_config)
        if self.max_turns is not None:
            config["max_turns"] = self.max_turns
        self._delegate = _LegacyHabitatGSInteractiveViewPlanning(config)
        return await self._delegate.reset(seed=0 if seed is None else int(seed))

    async def _system_prompt(self) -> Obs:
        return await self._require_delegate().system_prompt()

    async def _step(self, action: Any) -> StepResult:
        observation, reward, done, raw_info = await self._require_delegate().step(
            str(action)
        )
        info = dict(raw_info or {})
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


__all__ = ["HabitatGSInteractiveViewPlanning"]
