"""OpenAI-compatible model backend used by standalone evaluation."""

from __future__ import annotations

import asyncio
import base64
import copy
import time
from typing import TYPE_CHECKING, Any

from vagen_agent.evaluation.backends._common import BackendReply, register_backend
from vagen_agent.evaluation.image_codec import IMAGE_MIME, encode_jpeg

if TYPE_CHECKING:
    from vagen_agent.evaluation.config import ModelSpec


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set an engine-specific OAI extension without baking engines into the runner."""

    parts = dotted_key.split(".")
    current = mapping
    for part in parts[:-1]:
        nested = current.setdefault(part, {})
        if not isinstance(nested, dict):
            raise TypeError(
                f"cannot set {dotted_key!r}: request field {part!r} is already non-object"
            )
        current = nested
    leaf = parts[-1]
    if leaf in current and current[leaf] != value:
        raise ValueError(
            f"thinking budget conflicts with explicit sampling field {dotted_key!r}"
        )
    current[leaf] = value


def _data_url(image: Any) -> str:
    return f"data:{IMAGE_MIME};base64," + base64.b64encode(encode_jpeg(image)).decode(
        "ascii"
    )


def render_message(message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a VAGEN message to OAI multimodal content and a redacted recording."""

    role = str(message.get("role", "user"))
    parts = message.get("content", "")
    parts = parts if isinstance(parts, list) else [{"type": "text", "text": str(parts)}]
    images = iter(message.get("images") or [])
    rendered: list[dict[str, Any]] = []
    recorded: list[dict[str, Any]] = []
    image_index = 0
    for part in parts:
        if not isinstance(part, dict):
            text = str(part)
            rendered.append({"type": "text", "text": text})
            recorded.append({"type": "text", "text": text})
            continue
        kind = part.get("type")
        if kind == "text":
            text = str(part.get("text", ""))
            rendered.append({"type": "text", "text": text})
            recorded.append({"type": "text", "text": text})
        elif kind == "image":
            try:
                image = next(images)
            except StopIteration as exc:
                raise ValueError("message has more image parts than attached images") from exc
            rendered.append({"type": "image_url", "image_url": {"url": _data_url(image)}})
            recorded.append({"type": "image", "image_index": image_index})
            image_index += 1
        else:
            raise ValueError(f"unsupported message content type {kind!r}")
    try:
        next(images)
    except StopIteration:
        pass
    else:
        raise ValueError("message has attached images without matching image parts")
    if not rendered:
        rendered = [{"type": "text", "text": ""}]
        recorded = [{"type": "text", "text": ""}]
    return {"role": role, "content": rendered}, {"role": role, "content": recorded}


@register_backend("openai")
class OpenAIChatBackend:
    """One client for SGLang, vLLM, or a hosted OpenAI-compatible API."""

    def __init__(self, spec: ModelSpec) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency error
            raise ImportError("standalone evaluation requires httpx") from exc
        self._httpx = httpx
        self.spec = spec
        self._client = httpx.AsyncClient(timeout=spec.timeout)
        self._gate = asyncio.Semaphore(spec.max_concurrency)
        self._request_index = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        thinking_token_budget: int | None = None,
    ) -> BackendReply:
        rendered, recorded = [], []
        for message in messages:
            actual, shadow = render_message(message)
            rendered.append(actual)
            recorded.append(shadow)

        # Engine extension objects (for example SGLang's custom_params) are nested.
        # Copy them before injecting a per-call budget so concurrent calls never mutate
        # the frozen model configuration or one another.
        generation = copy.deepcopy({**self.spec.sampling, **sampling})
        generation.pop("max_new_tokens", None)
        if max_new_tokens is not None:
            generation[self.spec.token_limit_field] = int(max_new_tokens)
        if thinking_token_budget is not None:
            if self.spec.thinking_budget_field is None:
                raise ValueError(
                    f"model {self.spec.name!r} has a thinking budget but no "
                    "thinking_budget_field"
                )
            effective_budget = int(thinking_token_budget)
            if max_new_tokens is not None:
                # A reasoning budget is only useful when the model still has room to
                # emit executable final content. This also makes compact's smaller
                # summary generation safe without a harness-specific API.
                effective_budget = min(
                    effective_budget,
                    max(0, int(max_new_tokens) - self.spec.thinking_answer_reserve),
                )
            _set_nested(generation, self.spec.thinking_budget_field, effective_budget)
        payload = {"model": self.spec.served_model, "messages": rendered, **generation}
        headers = {
            "Authorization": f"Bearer {self.spec.api_key}",
            **self.spec.headers,
        }
        request_index = self._request_index
        self._request_index += 1
        started = time.monotonic()
        for attempt in range(self.spec.max_retries + 1):
            base_url = self.spec.base_urls[(request_index + attempt) % len(self.spec.base_urls)]
            endpoint = f"{base_url}/chat/completions"
            try:
                async with self._gate:
                    response = await self._client.post(endpoint, json=payload, headers=headers)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.is_error:
                    raise RuntimeError(
                        f"model server returned HTTP {response.status_code}: {response.text[:8000]}"
                    )
                value = response.json()
                choice = value["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content")
                if not isinstance(content, str):
                    raise TypeError("model response content must be text")
                if reasoning is not None and not isinstance(reasoning, str):
                    reasoning = str(reasoning)
                usage = value.get("usage") or {}
                completion_tokens = int(usage.get("completion_tokens") or 0)
                if not completion_tokens and content:
                    completion_tokens = max(1, len(content) // 4)
                return BackendReply(
                    content=content,
                    reasoning_content=reasoning,
                    finish_reason=(str(choice["finish_reason"])
                                   if choice.get("finish_reason") is not None else None),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=completion_tokens,
                    latency_seconds=time.monotonic() - started,
                    response_id=(str(value["id"]) if value.get("id") is not None else None),
                    endpoint=endpoint,
                    request_sampling=generation,
                    recorded_messages=recorded,
                )
            except (
                self._httpx.TimeoutException,
                self._httpx.NetworkError,
                self._httpx.HTTPStatusError,
            ):
                if attempt >= self.spec.max_retries:
                    raise
                await asyncio.sleep(min(
                    self.spec.max_backoff,
                    self.spec.min_backoff * (2**attempt),
                ))
        raise AssertionError("unreachable")

    async def close(self) -> None:
        await self._client.aclose()


__all__ = ["BackendReply", "OpenAIChatBackend", "render_message"]
