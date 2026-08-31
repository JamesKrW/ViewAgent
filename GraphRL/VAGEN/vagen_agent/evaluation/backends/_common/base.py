"""Backend response and lifecycle contract for standalone evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BackendReply:
    content: str
    reasoning_content: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    response_id: str | None
    endpoint: str
    request_sampling: dict[str, Any]
    recorded_messages: list[dict[str, Any]]


class EvaluationBackend(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        thinking_token_budget: int | None = None,
    ) -> BackendReply: ...

    async def close(self) -> None: ...


__all__ = ["BackendReply", "EvaluationBackend"]
