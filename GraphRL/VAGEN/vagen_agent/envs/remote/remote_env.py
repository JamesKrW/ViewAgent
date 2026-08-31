"""Direct BaseEnv implementation backed by a stateful HTTP service."""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Any, ClassVar

import httpx
from PIL import Image

from vagen_agent.envs._common import (
    BaseCompactEnv,
    BaseConcatEnv,
    BaseNoConcatEnv,
    EnvAction,
    Obs,
    Reward,
)
from vagen_agent.envs.remote.multipart_codec import decode_multipart, encode_multipart
from vagen_agent.envs.remote.observation_codec import to_json_value, unpack_observation

LOGGER = logging.getLogger(__name__)


class RemoteEnv(BaseNoConcatEnv, BaseConcatEnv, BaseCompactEnv):
    """Expose a remote service through VAGEN-SLIME's :class:`BaseEnv` contract.

    The first ``reset`` creates a server-side session and pins this client to the
    selected URL.  Later calls never fail over to a different server because its
    in-memory session would not exist there.  ``retries`` therefore applies only
    while establishing the session; state-changing calls are sent exactly once.
    """

    _CLIENT_KEYS: ClassVar[set[str]] = {
        "base_urls",
        "url_file",
        "timeout",
        "retries",
        "backoff",
        "max_delay",
        "backoff_jitter_min",
        "backoff_jitter_range",
        "token",
        "log_retries",
        "failover_after_failures",
    }

    def __init__(self, env_config: dict[str, Any] | None = None):
        super().__init__(env_config)
        raw_urls = self.env_config.get("base_urls")
        if not raw_urls:
            url_file = self.env_config.get("url_file")
            if not url_file:
                raise ValueError(
                    "RemoteEnv needs config.base_urls or config.url_file with one URL per line"
                )
            raw_urls = Path(str(url_file)).expanduser().read_text(encoding="utf-8")
        self.base_urls = self._parse_urls(raw_urls)
        self.timeout = float(self.env_config.get("timeout", 120.0))
        self.retries = int(self.env_config.get("retries", 6))
        self.backoff = float(self.env_config.get("backoff", 2.0))
        self.max_delay = float(self.env_config.get("max_delay", 64.0))
        self.backoff_jitter_min = float(self.env_config.get("backoff_jitter_min", 0.7))
        self.backoff_jitter_range = float(
            self.env_config.get("backoff_jitter_range", 0.6)
        )
        token = self.env_config.get("token")
        self.token = "" if token is None else str(token)
        self.log_retries = bool(self.env_config.get("log_retries", True))
        self.failover_after_failures = int(
            self.env_config.get("failover_after_failures", 4)
        )
        if (
            self.timeout <= 0
            or self.retries < 0
            or self.backoff < 0
            or self.max_delay < 0
            or self.backoff_jitter_min < 0
            or self.backoff_jitter_range < 0
        ):
            raise ValueError(
                "RemoteEnv timeout must be positive and retry delays non-negative"
            )
        if self.failover_after_failures < 0:
            raise ValueError("RemoteEnv failover_after_failures must be non-negative")

        self._remote_env_config = {
            key: value
            for key, value in self.env_config.items()
            if key not in self._CLIENT_KEYS
        }
        self.remote_env_config = self._remote_env_config
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._base_url: str | None = None
        self._current_url_index = 0

    @staticmethod
    def _parse_urls(raw: Any) -> tuple[str, ...]:
        if isinstance(raw, str):
            values = raw.replace(";", "\n").splitlines()
        elif isinstance(raw, (list, tuple)):
            values = raw
            if not all(isinstance(value, str) for value in values):
                raise TypeError("RemoteEnv base_urls entries must be strings")
        else:
            raise TypeError("RemoteEnv base_urls must be a URL or a list of URLs")
        urls = tuple(
            str(value).strip().rstrip("/")
            for value in values
            if str(value).strip() and not str(value).lstrip().startswith("#")
        )
        if not urls:
            raise ValueError("RemoteEnv base_urls is empty")
        if len(set(urls)) != len(urls):
            raise ValueError("RemoteEnv base_urls contains duplicates")
        return urls

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    def _headers(self, boundary: str) -> dict[str, str]:
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        if self.token:
            headers["X-API-Key"] = self.token
        return headers

    @staticmethod
    def _observation(data: dict[str, Any], images: list[Image.Image]) -> Obs:
        if "observation" in data:
            return unpack_observation(data["observation"], images)
        # Protocol-v1 compatibility: only text plus an implicit ordered image list.
        observation: Obs = {"obs_str": str(data.get("obs", ""))}
        if images:
            observation["multi_modal_input"] = {"<image>": images}
        return observation

    def _pick_url_index(self, attempt: int) -> int:
        if len(self.base_urls) == 1:
            return 0
        if attempt <= self.failover_after_failures:
            return self._current_url_index
        return (self._current_url_index + attempt - self.failover_after_failures) % len(
            self.base_urls
        )

    async def _connect(self, seed: int | None) -> tuple[Obs, dict] | None:
        client = await self._ensure_client()
        request: dict[str, Any] = {"env_config": self._remote_env_config}
        if seed is not None:
            request["seed"] = int(seed)
            self._current_url_index = int(seed) % len(self.base_urls)
        boundary, body = encode_multipart(request)
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            url_index = self._pick_url_index(attempt)
            base_url = self.base_urls[url_index]
            try:
                response = await client.post(
                    f"{base_url}/connect", content=body, headers=self._headers(boundary)
                )
                if response.status_code == 503:
                    raise RuntimeError("remote environment server is busy")
                response.raise_for_status()
                data, images = decode_multipart(
                    response.headers.get("content-type", ""), response.content
                )
                session_id = data.get("session_id")
                if not session_id:
                    raise RuntimeError(
                        "remote environment service returned no session_id"
                    )
                self._session_id = str(session_id)
                self._base_url = base_url
                self._current_url_index = url_index
                LOGGER.info(
                    "connected RemoteEnv session %s to %s", self._session_id, base_url
                )
                if seed is not None and ("observation" in data or "obs" in data):
                    return self._observation(data, images), dict(data.get("info") or {})
                return None
            except Exception as exc:  # noqa: BLE001 - connect retry covers transport/server errors
                last_error = exc
                if attempt >= self.retries:
                    break
                delay = min(
                    self.max_delay,
                    self.backoff
                    * (2**attempt)
                    * (
                        self.backoff_jitter_min
                        + self.backoff_jitter_range * random.random()
                    ),
                )
                if self.log_retries:
                    LOGGER.warning(
                        "RemoteEnv connect retry %d/%d after %.2fs: %s",
                        attempt + 1,
                        self.retries,
                        delay,
                        exc,
                    )
                await asyncio.sleep(delay)
        raise RuntimeError(f"failed to connect RemoteEnv: {last_error}") from last_error

    async def _call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        images: list[Image.Image] | None = None,
    ) -> tuple[dict[str, Any], list[Image.Image]]:
        if self._session_id is None or self._base_url is None:
            raise RuntimeError(f"cannot call RemoteEnv.{method} before reset")
        client = await self._ensure_client()
        boundary, body = encode_multipart(
            {"session_id": self._session_id, "method": method, "params": params or {}},
            images,
        )
        response = await client.post(
            f"{self._base_url}/call", content=body, headers=self._headers(boundary)
        )
        response.raise_for_status()
        return decode_multipart(
            response.headers.get("content-type", ""), response.content
        )

    async def _reset(self, seed: int | None = None) -> tuple[Obs, dict]:
        if self._session_id is None:
            initial = await self._connect(seed)
            if initial is not None:
                return initial
        data, images = await self._call("reset", {"seed": seed})
        return self._observation(data, images), dict(data.get("info") or {})

    async def _system_prompt(self) -> Obs:
        data, images = await self._call("system_prompt")
        return self._observation(data, images)

    async def _step(self, action: Any) -> tuple[Obs, Reward, bool, bool, dict]:
        text = getattr(action, "text", action)
        params: dict[str, Any] = {"action_str": str(text)}
        if isinstance(action, EnvAction):
            params["action"] = to_json_value(action.value, path="action")
        data, images = await self._call("step", params)
        info = dict(data.get("info") or {})
        truncated = bool(data.get("truncated", info.get("truncated", False)))
        terminated = bool(data.get("terminated", data.get("done", False)))
        return (
            self._observation(data, images),
            data.get("reward", 0.0),
            terminated and not truncated,
            truncated,
            info,
        )

    async def close(self) -> None:
        if self._session_id is not None:
            try:
                await self._call("close")
            except Exception as exc:  # noqa: BLE001 - cleanup is deliberately best effort
                LOGGER.warning(
                    "failed to close RemoteEnv session %s: %s", self._session_id, exc
                )
            finally:
                self._session_id = None
                self._base_url = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["RemoteEnv"]
