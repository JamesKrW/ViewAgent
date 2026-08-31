"""Minimal environment protocol consumed by VAGEN Harnesses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

Obs = dict[str, Any]
Reward = float | list[float]
StepResult = tuple[Obs, Reward, bool, bool, dict[str, Any]]


class BaseEnv(ABC):
    """Describe the complete Harness-facing environment lifecycle.

    This class intentionally contains no validation, counters, metrics, or storage.
    Implementations shared by the built-in Harness families begin in
    :class:`BaseVagenEnv`; a new interaction model may implement this protocol
    directly without inheriting those policies.
    """

    @abstractmethod
    def configure(self, *, max_turns: int | None = None) -> None:
        """Apply framework-owned episode limits before reset."""

    @abstractmethod
    async def reset(self, seed: int | None = None) -> tuple[Obs, dict[str, Any]]:
        """Start one episode and return its initial observation and metadata."""

    @abstractmethod
    async def system_prompt(self) -> Obs:
        """Return the standing task and action instructions."""

    @abstractmethod
    async def step(self, action: Any) -> StepResult:
        """Apply one Harness action using Gymnasium termination semantics."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the environment."""


__all__ = ["BaseEnv", "Obs", "Reward", "StepResult"]
