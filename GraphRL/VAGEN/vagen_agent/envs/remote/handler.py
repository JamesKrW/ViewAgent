"""Environment-agnostic session handler for the remote HTTP service."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from vagen_agent.envs._common import BaseEnv
from vagen_agent.envs.remote.observation_codec import pack_observation, to_json_value

LOGGER = logging.getLogger(__name__)


class SessionNotFoundError(KeyError):
    """Raised when a request references a missing or expired session."""


@dataclass
class HandlerResult:
    data: dict[str, Any]
    images: list[Image.Image] | None = None


@dataclass
class SessionContext:
    session_id: str
    env: Any
    created_at: float
    last_access: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class BaseGymHandler(ABC):
    """Manage stateful remote sessions; subclasses only implement ``create_env``."""

    def __init__(self, session_timeout: float = 3600.0, max_sessions: int = 0):
        if session_timeout <= 0:
            raise ValueError("session_timeout must be positive")
        if max_sessions < 0:
            raise ValueError("max_sessions must be non-negative")
        self.session_timeout = float(session_timeout)
        self.max_sessions = int(max_sessions)
        self._sessions: dict[str, SessionContext] = {}
        self._starting = 0
        self._sessions_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    @abstractmethod
    async def create_env(self, env_config: dict[str, Any]) -> BaseEnv:
        """Create one direct VAGEN ``BaseEnv`` implementation."""

    async def connect(
        self, env_config: dict[str, Any], seed: int | None = None
    ) -> HandlerResult:
        async with self._sessions_lock:
            if (
                self.max_sessions
                and len(self._sessions) + self._starting >= self.max_sessions
            ):
                raise RuntimeError(
                    f"maximum active sessions reached ({self.max_sessions})"
                )
            self._starting += 1

        env = None
        try:
            env = await self.create_env(dict(env_config or {}))
            if not isinstance(env, BaseEnv):
                raise TypeError("remote handler create_env() must return BaseEnv")
            result = None
            if seed is not None:
                obs, info = await env.reset(seed=int(seed))
                result = self._obs_to_result(obs)
                result.data["info"] = to_json_value(dict(info or {}), path="info")

            session_id = uuid.uuid4().hex
            now = time.time()
            async with self._sessions_lock:
                self._sessions[session_id] = SessionContext(
                    session_id=session_id,
                    env=env,
                    created_at=now,
                    last_access=now,
                )
            self._ensure_cleanup()
            if result is None:
                return HandlerResult({"session_id": session_id})
            result.data["session_id"] = session_id
            return result
        except BaseException:
            if env is not None:
                try:
                    await env.close()
                except Exception:
                    LOGGER.exception("failed to close environment after connect error")
            raise
        finally:
            async with self._sessions_lock:
                self._starting -= 1

    async def call(
        self,
        session_id: str,
        method: str,
        params: dict[str, Any],
        images: list[Image.Image] | None = None,
    ) -> HandlerResult:
        del images  # reserved for protocols whose actions include image inputs
        context = self._sessions.get(session_id)
        if context is None:
            raise SessionNotFoundError(session_id)
        async with context.lock:
            if self._sessions.get(session_id) is not context:
                raise SessionNotFoundError(session_id)
            context.last_access = time.time()
            if method == "system_prompt":
                result = await self._handle_system_prompt(context)
            elif method == "reset":
                result = await self._handle_reset(context, params)
            elif method == "step":
                result = await self._handle_step(context, params)
            elif method == "close":
                result = await self._handle_close(context)
            else:
                raise ValueError(f"unknown remote environment method {method!r}")
            context.last_access = time.time()
            return result

    async def _handle_system_prompt(self, context: SessionContext) -> HandlerResult:
        return self._obs_to_result(await context.env.system_prompt())

    async def _handle_reset(
        self, context: SessionContext, params: dict[str, Any]
    ) -> HandlerResult:
        obs, info = await context.env.reset(seed=params.get("seed"))
        result = self._obs_to_result(obs)
        result.data["info"] = to_json_value(dict(info or {}), path="info")
        return result

    async def _handle_step(
        self, context: SessionContext, params: dict[str, Any]
    ) -> HandlerResult:
        action = (
            params["action"]
            if "action" in params
            else str(params.get("action_str", ""))
        )
        obs, reward, terminated, truncated, info = await context.env.step(action)
        result = self._obs_to_result(obs)
        result.data.update(
            {
                "reward": to_json_value(reward, path="reward"),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": to_json_value(dict(info or {}), path="info"),
            }
        )
        return result

    async def _handle_close(self, context: SessionContext) -> HandlerResult:
        try:
            await context.env.close()
        finally:
            async with self._sessions_lock:
                self._sessions.pop(context.session_id, None)
        return HandlerResult({"closed": True})

    @classmethod
    def _obs_to_result(cls, obs: dict[str, Any]) -> HandlerResult:
        observation, images = pack_observation(obs)
        # ``obs`` is retained for protocol-v1 clients. Protocol-v2 clients prefer the
        # complete observation object and restore its binary image references.
        return HandlerResult(
            {
                "observation": observation,
                "obs": str(obs.get("obs_str", "")),
            },
            images,
        )

    def _ensure_cleanup(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        interval = max(1.0, min(60.0, self.session_timeout / 2))
        try:
            while self._sessions:
                await asyncio.sleep(interval)
                now = time.time()
                expired = [
                    session_id
                    for session_id, context in self._sessions.items()
                    if not context.lock.locked()
                    and now - context.last_access > self.session_timeout
                ]
                for session_id in expired:
                    context = self._sessions.get(session_id)
                    if context is None:
                        continue
                    async with context.lock:
                        if now - context.last_access <= self.session_timeout:
                            continue
                        try:
                            await context.env.close()
                        except Exception:
                            LOGGER.exception(
                                "failed to close expired session %s", session_id
                            )
                        async with self._sessions_lock:
                            self._sessions.pop(session_id, None)
        except asyncio.CancelledError:
            pass

    def get_session_stats(self) -> dict[str, Any]:
        now = time.time()
        return {
            "num_sessions": len(self._sessions),
            "starting_sessions": self._starting,
            "max_sessions": self.max_sessions,
            "session_timeout": self.session_timeout,
            "sessions": [
                {
                    "session_id": context.session_id,
                    "created_at": context.created_at,
                    "last_access": context.last_access,
                    "idle_seconds": now - context.last_access,
                }
                for context in self._sessions.values()
            ],
        }

    async def aclose(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        contexts = list(self._sessions.values())
        self._sessions.clear()
        if contexts:
            await asyncio.gather(
                *(context.env.close() for context in contexts), return_exceptions=True
            )


__all__ = ["BaseGymHandler", "HandlerResult", "SessionContext", "SessionNotFoundError"]
