"""Provider backends used by the legacy ViewSuite evaluation configs.

The evaluator, concurrency, harness execution, recording and resume behavior
all come from :mod:`vagen_agent.evaluation`.  Only provider-specific HTTP
translation lives here.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from vagen_agent.evaluation.backends import (
    BackendReply,
    OpenAIChatBackend,
    register_backend,
    render_message,
)
from vagen_agent.evaluation.image_codec import IMAGE_MIME, encode_jpeg


def _text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get("text", ""))
        for part in content or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _fingerprint(messages: list[dict[str, Any]]) -> int:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(str(message.get("role", "")).encode())
        digest.update(_text(message).encode())
        for image in message.get("images") or []:
            digest.update(encode_jpeg(image))
    return int(digest.hexdigest()[:16], 16)


def _reply(
    *,
    content: str,
    started: float,
    endpoint: str,
    sampling: dict[str, Any],
    messages: list[dict[str, Any]],
    reasoning: str | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    response_id: str | None = None,
) -> BackendReply:
    recorded = [render_message(message)[1] for message in messages]
    return BackendReply(
        content=content,
        reasoning_content=reasoning,
        finish_reason=finish_reason,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens or (max(1, len(content) // 4) if content else 0)),
        latency_seconds=time.monotonic() - started,
        response_id=response_id,
        endpoint=endpoint,
        request_sampling=sampling,
        recorded_messages=recorded,
    )


class _HttpBackend:
    def __init__(self, spec: Any) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ViewSuite evaluation requires httpx") from exc
        self.httpx = httpx
        self.spec = spec
        self.client = httpx.AsyncClient(timeout=spec.timeout)
        self.gate = asyncio.Semaphore(spec.max_concurrency)

    async def _post(
        self,
        endpoint: str,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        for attempt in range(self.spec.max_retries + 1):
            try:
                async with self.gate:
                    response = await self.client.post(endpoint, json=payload, headers=headers)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.is_error:
                    raise RuntimeError(
                        f"model server returned HTTP {response.status_code}: "
                        f"{response.text[:8000]}"
                    )
                return response.json()
            except (
                self.httpx.TimeoutException,
                self.httpx.NetworkError,
                self.httpx.HTTPStatusError,
            ):
                if attempt >= self.spec.max_retries:
                    raise
                await asyncio.sleep(
                    min(self.spec.max_backoff, self.spec.min_backoff * (2**attempt))
                )
        raise AssertionError("unreachable")

    async def close(self) -> None:
        await self.client.aclose()


# These providers all implement the same OpenAI chat-completions wire protocol.
for _name in ("sglang", "vllm", "together"):
    register_backend(_name)(OpenAIChatBackend)


@register_backend("azure")
class AzureChatBackend(_HttpBackend):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        thinking_token_budget: int | None = None,
    ) -> BackendReply:
        del thinking_token_budget
        rendered = [render_message(message)[0] for message in messages]
        generation = copy.deepcopy({**self.spec.sampling, **sampling})
        generation.pop("max_new_tokens", None)
        if max_new_tokens is not None:
            generation[self.spec.token_limit_field] = int(max_new_tokens)
        options = self.spec.options
        endpoint = (
            f"{options['azure_endpoint'].rstrip('/')}/openai/deployments/"
            f"{options['deployment']}/chat/completions"
            f"?api-version={options['api_version']}"
        )
        started = time.monotonic()
        value = await self._post(
            endpoint,
            payload={"messages": rendered, **generation},
            headers={"api-key": self.spec.api_key, **self.spec.headers},
        )
        choice = value["choices"][0]
        message = choice.get("message") or {}
        usage = value.get("usage") or {}
        return _reply(
            content=str(message.get("content") or ""),
            reasoning=message.get("reasoning_content"),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            response_id=value.get("id"),
            started=started,
            endpoint=endpoint,
            sampling=generation,
            messages=messages,
        )


def _responses_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role", "user"))
    images = iter(message.get("images") or [])
    raw = message.get("content", "")
    parts = raw if isinstance(raw, list) else [{"type": "text", "text": str(raw)}]
    content = []
    for part in parts:
        if not isinstance(part, dict):
            content.append({"type": "input_text", "text": str(part)})
        elif part.get("type") == "text":
            kind = "output_text" if role == "assistant" else "input_text"
            content.append({"type": kind, "text": str(part.get("text", ""))})
        elif part.get("type") == "image":
            image = next(images)
            data = base64.b64encode(encode_jpeg(image)).decode("ascii")
            content.append(
                {"type": "input_image", "image_url": f"data:{IMAGE_MIME};base64,{data}"}
            )
    return {"role": role, "content": content}


@register_backend("azure_responses", "openai_responses")
class ResponsesBackend(_HttpBackend):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        thinking_token_budget: int | None = None,
    ) -> BackendReply:
        del thinking_token_budget
        generation = copy.deepcopy({**self.spec.sampling, **sampling})
        generation.pop("max_tokens", None)
        generation.pop("max_completion_tokens", None)
        if max_new_tokens is not None:
            generation["max_output_tokens"] = int(max_new_tokens)
        options = self.spec.options
        endpoint = options.get("responses_url")
        if not endpoint:
            endpoint = (
                f"{options['azure_endpoint'].rstrip('/')}/openai/responses"
                f"?api-version={options['api_version']}"
            )
        headers = {**self.spec.headers}
        if self.spec.api_key:
            headers.setdefault("api-key", self.spec.api_key)
            headers.setdefault("Authorization", f"Bearer {self.spec.api_key}")
        started = time.monotonic()
        value = await self._post(
            endpoint,
            payload={
                "model": self.spec.served_model,
                "input": [_responses_message(message) for message in messages],
                **generation,
            },
            headers=headers,
        )
        content = str(value.get("output_text") or "")
        if not content:
            content = "".join(
                str(part.get("text", ""))
                for item in value.get("output") or []
                for part in item.get("content") or []
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}
            )
        usage = value.get("usage") or {}
        return _reply(
            content=content,
            finish_reason=value.get("status"),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            response_id=value.get("id"),
            started=started,
            endpoint=endpoint,
            sampling=generation,
            messages=messages,
        )


def _provider_parts(message: dict[str, Any], *, provider: str) -> list[dict[str, Any]]:
    images = iter(message.get("images") or [])
    raw = message.get("content", "")
    parts = raw if isinstance(raw, list) else [{"type": "text", "text": str(raw)}]
    out: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") == "text":
            text = str(part.get("text", "")) if isinstance(part, dict) else str(part)
            if text:
                out.append({"type": "text", "text": text} if provider == "claude" else {"text": text})
        elif part.get("type") == "image":
            image = next(images)
            data = base64.b64encode(encode_jpeg(image)).decode("ascii")
            if provider == "claude":
                out.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": IMAGE_MIME, "data": data},
                    }
                )
            else:
                out.append({"inline_data": {"mime_type": IMAGE_MIME, "data": data}})
    return out


@register_backend("claude")
class ClaudeBackend(_HttpBackend):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        thinking_token_budget: int | None = None,
    ) -> BackendReply:
        del thinking_token_budget
        system: list[str] = []
        converted = []
        for message in messages:
            if message.get("role") == "system":
                if _text(message).strip():
                    system.append(_text(message).strip())
            else:
                converted.append(
                    {
                        "role": "assistant" if message.get("role") == "assistant" else "user",
                        "content": _provider_parts(message, provider="claude"),
                    }
                )
        generation = copy.deepcopy({**self.spec.sampling, **sampling})
        limit = max_new_tokens or generation.pop("max_output_tokens", None) or generation.pop("max_tokens", 512)
        endpoint = f"{self.spec.base_urls[0].rstrip('/')}/v1/messages"
        payload = {
            "model": self.spec.served_model,
            "messages": converted,
            "max_tokens": int(limit),
            **generation,
        }
        if system:
            payload["system"] = "\n".join(system)
        started = time.monotonic()
        value = await self._post(
            endpoint,
            payload=payload,
            headers={
                "x-api-key": self.spec.api_key,
                "anthropic-version": self.spec.options.get("api_version", "2023-06-01"),
                **self.spec.headers,
            },
        )
        content = "\n".join(
            str(block.get("text", ""))
            for block in value.get("content") or []
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = value.get("usage") or {}
        return _reply(
            content=content,
            finish_reason=value.get("stop_reason"),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            response_id=value.get("id"),
            started=started,
            endpoint=endpoint,
            sampling={"max_tokens": int(limit), **generation},
            messages=messages,
        )


def _camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


@register_backend("gemini")
class GeminiBackend(_HttpBackend):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        thinking_token_budget: int | None = None,
    ) -> BackendReply:
        del thinking_token_budget
        system: list[str] = []
        contents = []
        for message in messages:
            if message.get("role") == "system":
                if _text(message).strip():
                    system.append(_text(message).strip())
            else:
                contents.append(
                    {
                        "role": "model" if message.get("role") == "assistant" else "user",
                        "parts": _provider_parts(message, provider="gemini"),
                    }
                )
        raw = copy.deepcopy({**self.spec.sampling, **sampling})
        generation = dict(raw.pop("generation_config", {}) or {})
        generation.update(raw)
        if max_new_tokens is not None:
            generation["max_output_tokens"] = int(max_new_tokens)
        safety = generation.pop("safety_settings", None)
        generation = {_camel(str(key)): value for key, value in generation.items()}
        key = self.spec.api_key
        endpoint = (
            f"{self.spec.base_urls[0].rstrip('/')}/v1beta/models/"
            f"{self.spec.served_model}:generateContent?key={key}"
        )
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}
        if safety is not None:
            payload["safetySettings"] = safety
        started = time.monotonic()
        value = await self._post(endpoint, payload=payload, headers=self.spec.headers)
        candidate = (value.get("candidates") or [{}])[0]
        content = "\n".join(
            str(part.get("text", ""))
            for part in (candidate.get("content") or {}).get("parts") or []
            if isinstance(part, dict) and part.get("text")
        )
        usage = value.get("usageMetadata") or {}
        return _reply(
            content=content,
            finish_reason=candidate.get("finishReason"),
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            started=started,
            endpoint=endpoint,
            sampling=generation,
            messages=messages,
        )


class _RandomBackend:
    def __init__(self, spec: Any) -> None:
        self.spec = spec

    async def close(self) -> None:
        return None

    def _rng(self, messages: list[dict[str, Any]], sampling: dict[str, Any]) -> random.Random:
        seed = int(sampling.get("random_seed", self.spec.sampling.get("random_seed", 0)))
        return random.Random(seed ^ _fingerprint(messages))


@register_backend("random_response")
class RandomResponseBackend(_RandomBackend):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        thinking_token_budget: int | None = None,
    ) -> BackendReply:
        del max_new_tokens, thinking_token_budget
        started = time.monotonic()
        effective = {**self.spec.sampling, **sampling}
        path = effective.get("file_path")
        if not path:
            raise ValueError("random_response requires chat_config.file_path")
        responses = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(responses, list) or not responses:
            raise ValueError(f"random response file is empty or invalid: {path}")
        content = str(self._rng(messages, effective).choice(responses))
        return _reply(
            content=content,
            started=started,
            endpoint="random://response",
            sampling=effective,
            messages=messages,
        )


_STEP_RE = re.compile(r"(?:Step|next turn:)\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_POSE_RE = re.compile(
    r"\[tx=([-\d.]+),\s*ty=([-\d.]+),\s*tz=([-\d.]+),\s*"
    r"rx=([-\d.]+)°?,\s*ry=([-\d.]+)°?,\s*rz=([-\d.]+)°?\]"
)


@register_backend("random_navigation")
class RandomNavigationBackend(_RandomBackend):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        thinking_token_budget: int | None = None,
    ) -> BackendReply:
        del max_new_tokens, thinking_token_budget
        started = time.monotonic()
        effective = {**self.spec.sampling, **sampling}
        rng = self._rng(messages, effective)
        joined = "\n".join(_text(message) for message in messages)
        last_user = next(
            (_text(message) for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        match = _STEP_RE.search(last_user)
        step, limit = (int(match.group(1)), int(match.group(2))) if match else (1, 1)
        habitat = "submit_pose(" in joined or "Habitat-GS" in joined
        if step >= limit:
            poses = [tuple(float(m.group(i)) for i in range(1, 7)) for m in _POSE_RE.finditer(joined)]
            pose = rng.choice(poses) if poses else (0.0,) * 6
            name = "submit_pose" if habitat else "answer"
            actions = f"{name}({', '.join(f'{value:.2f}' for value in pose)})"
        else:
            pool = (
                ["w", "s", "a", "d", "z", "x", "arrow_left", "arrow_right", "arrow_up", "arrow_down"]
                if habitat
                else [
                    "move_forward", "move_backward", "move_right", "move_left",
                    "move_up", "move_down", "turn_left", "turn_right",
                    "look_up", "look_down", "rotate_ccw", "rotate_cw",
                ]
            )
            actions = "|".join(rng.choice(pool) for _ in range(rng.randint(1, 3)))
            if step == 1 and not habitat and "select_view" in joined:
                actions = f"select_view(init_view)|{actions}"
        content = f"<think>random exploration step {step}/{limit}</think><action>{actions}|</action>"
        return _reply(
            content=content,
            started=started,
            endpoint="random://navigation",
            sampling=effective,
            messages=messages,
        )


__all__ = [
    "AzureChatBackend",
    "ClaudeBackend",
    "GeminiBackend",
    "RandomNavigationBackend",
    "RandomResponseBackend",
    "ResponsesBackend",
]
