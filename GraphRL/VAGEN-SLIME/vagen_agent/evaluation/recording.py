"""Episode recording and the thin client/environment seams used by every harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vagen_agent.evaluation.backends import EvaluationBackend
from vagen_agent.evaluation.image_codec import IMAGE_MIME, IMAGE_SUFFIX, encode_jpeg


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class EvaluationResponse:
    """Structural subset of ``vagen_agent.rollout.client.Response`` used by harnesses/envs.

    Keeping this small type local prevents standalone evaluation from importing Slime's
    token-level rollout client. Environments consume ``text``; harnesses additionally
    use usage and call_id. OAI endpoints do not expose generated token ids or logprobs.
    """

    text: str
    token_ids: list[int]
    logprobs: list[float]
    stop_reason: str | None
    usage: Usage
    call_id: int


def json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 20:
        return "<depth-limit>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, depth + 1) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item(), depth + 1)
        except (TypeError, ValueError):
            pass
    return repr(value)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


@dataclass
class EpisodeRecorder:
    calls: list[dict[str, Any]] = field(default_factory=list)
    reset_info: dict[str, Any] = field(default_factory=dict)
    system_prompt: dict[str, Any] | None = None
    current_observation: dict[str, Any] | None = None
    rewards: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    _next_call_id: int = field(default=1, init=False, repr=False)

    def allocate_call_id(self) -> int:
        """Allocate an episode-global id shared by every model role."""

        value = self._next_call_id
        self._next_call_id += 1
        return value

    def add_call(self, value: dict[str, Any]) -> None:
        self.calls.append(value)

    def attach_step(
        self,
        call_id: int,
        *,
        before: dict[str, Any] | None,
        after: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        call = next((item for item in reversed(self.calls) if item["call_id"] == call_id), None)
        if call is None:
            raise RuntimeError(f"environment step has no matching model call {call_id}")
        call["transition"] = {
            "observation": before,
            "next_observation": after,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": dict(info or {}),
            "parsed_action": (info or {}).get("parsed_action"),
            "native_action": (info or {}).get("native_action"),
        }
        self.current_observation = after
        self.rewards.append(float(reward))
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)


class EvaluationClient:
    """The harness-facing ``create``/``size`` API backed by OAI chat completions."""

    def __init__(
        self,
        backend: EvaluationBackend,
        recorder: EpisodeRecorder,
        *,
        role: str = "default",
        sampling: dict[str, Any],
        thinking_token_budget: int | None,
        max_calls: int,
    ) -> None:
        self.backend = backend
        self.recorder = recorder
        self.role = str(role)
        self.sampling = dict(sampling)
        self.thinking_token_budget = thinking_token_budget
        self.max_calls = int(max_calls)
        self.n_calls = 0

    async def create(self, messages: list[dict[str, Any]], **sampling: Any) -> EvaluationResponse:
        if self.n_calls >= self.max_calls:
            raise RuntimeError(
                f"harness exceeded {self.max_calls} model calls in one episode"
            )
        self.n_calls += 1
        reply = await self.backend.complete(
            messages,
            self.sampling,
            max_new_tokens=sampling.pop("max_new_tokens", None),
            thinking_token_budget=self.thinking_token_budget,
        )
        if sampling:
            raise ValueError(f"unsupported harness sampling overrides: {sorted(sampling)}")
        response = EvaluationResponse(
            text=reply.content,
            token_ids=[],
            logprobs=[],
            stop_reason=reply.finish_reason,
            usage=Usage(reply.prompt_tokens, reply.completion_tokens),
            call_id=self.recorder.allocate_call_id(),
        )
        self.recorder.add_call({
            "call_id": response.call_id,
            "model_role": self.role,
            # Keep the VAGEN messages until persistence so their PIL frames can be
            # content-addressed beside the trajectory. The HTTP backend's redacted
            # shadow is useful for tests, but deliberately omits those bytes.
            "_messages": list(messages),
            # Record the effective request, including a harness-specific token ceiling.
            # This matters for compact summaries, whose generation limit deliberately
            # differs from an action turn.
            "sampling": dict(reply.request_sampling),
            "response": {
                "content": reply.content,
                "reasoning_content": reply.reasoning_content,
                "finish_reason": reply.finish_reason,
                "usage": asdict(response.usage),
                "latency_seconds": reply.latency_seconds,
                "response_id": reply.response_id,
                "endpoint": reply.endpoint,
            },
            "transition": None,
        })
        return response

    def size(self, messages: list[dict[str, Any]]) -> int:
        """Conservative fallback used only when a harness sets ``context_window``.

        OAI APIs do not expose a tokenizer endpoint. Compact's normal budget accounting
        uses the server-reported prompt/completion counts after each call; this estimate
        only prevents a request from knowingly exceeding a configured context window.
        """
        characters = 0
        images = 0
        for message in messages:
            content = message.get("content", "")
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "image":
                    images += 1
                elif isinstance(part, dict):
                    characters += len(str(part.get("text", "")))
                else:
                    characters += len(str(part))
        return max(1, (characters + 2) // 3 + images * 1024 + len(messages) * 8)


class RecordingEnv:
    """Record the direct BaseEnv contract without changing a Harness."""

    def __init__(self, env: Any, recorder: EpisodeRecorder, *, seed: int) -> None:
        self.env = env
        self.recorder = recorder
        self.seed = int(seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    async def reset(self, seed: int | None = None):
        # Harnesses intentionally know nothing about dataset rows and call reset()
        # without arguments. Bind the job identity here so a resumed episode replays
        # the exact same environment rather than silently using an unseeded reset.
        effective_seed = self.seed if seed is None else int(seed)
        observation, info = await self.env.reset(seed=effective_seed)
        self.recorder.current_observation = observation
        self.recorder.reset_info = dict(info or {})
        return observation, info

    async def system_prompt(self):
        prompt = await self.env.system_prompt()
        self.recorder.system_prompt = prompt
        return prompt

    async def step(self, response: EvaluationResponse):
        before = self.recorder.current_observation
        after, reward, terminated, truncated, info = await self.env.step(response)
        self.recorder.attach_step(
            response.call_id,
            before=before,
            after=after,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )
        return after, reward, terminated, truncated, info

    async def close(self) -> None:
        await self.env.close()


class EpisodeStore:
    """Stable per-seed completion markers and human-auditable trajectories."""

    def __init__(self, root: Path, *, record_images: bool) -> None:
        self.root = root
        self.record_images = record_images

    def episode_root(self, model: str, tag: str, seed: int) -> Path:
        return self.root / model / f"tag_{tag}" / f"seed_{seed}"

    def result_path(self, model: str, tag: str, seed: int) -> Path:
        return self.episode_root(model, tag, seed) / "result.json"

    def is_completed(self, model: str, tag: str, seed: int, fingerprint: str) -> bool:
        path = self.result_path(model, tag, seed)
        if not path.is_file():
            return False
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return result.get("status") == "completed" and result.get("config_sha256") == fingerprint

    def _image(
        self,
        root: Path,
        image: Any,
        cache: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        cache_key = id(image)
        if cache_key in cache:
            return dict(cache[cache_key])
        payload = encode_jpeg(image)
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path("images") / f"{digest}{IMAGE_SUFFIX}"
        if self.record_images:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                atomic_bytes(target, payload)
        descriptor = {
            "sha256": digest,
            "path": str(relative) if self.record_images else None,
            "width": image.width,
            "height": image.height,
            "media_type": IMAGE_MIME,
        }
        cache[cache_key] = descriptor
        return dict(descriptor)

    def _message(
        self,
        root: Path,
        message: dict[str, Any] | None,
        image_cache: dict[int, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if message is None:
            return None
        value = {
            "role": message.get("role"),
            "content": json_safe(message.get("content")),
            "images": [
                self._image(root, image, image_cache)
                for image in message.get("images") or []
            ],
        }
        reserved = {"role", "content", "images"}
        value.update(
            {
                key: json_safe(item)
                for key, item in message.items()
                if key not in reserved
            }
        )
        return value

    def write(self, result: dict[str, Any], recorder: EpisodeRecorder) -> Path:
        root = self.episode_root(result["model"], result["tag"], int(result["seed"]))
        root.mkdir(parents=True, exist_ok=True)
        image_cache: dict[int, dict[str, Any]] = {}
        calls = []
        for call in recorder.calls:
            value = dict(call)
            transition = value.get("transition")
            value["messages"] = [
                self._message(root, message, image_cache)
                for message in value.pop("_messages")
            ]
            if transition is not None:
                transition = dict(transition)
                transition["observation"] = self._message(
                    root, transition["observation"], image_cache
                )
                transition["next_observation"] = self._message(
                    root, transition["next_observation"], image_cache
                )
                value["transition"] = transition
            calls.append(value)
        public = {
            **result,
            "system_prompt": self._message(root, recorder.system_prompt, image_cache),
            "reset_info": json_safe(recorder.reset_info),
            "trajectory": calls,
        }
        trajectory = "".join(
            json.dumps(json_safe({"sequence": sequence, **call}), ensure_ascii=False) + "\n"
            for sequence, call in enumerate(calls)
        )
        atomic_text(root / "trajectory.jsonl", trajectory)
        transcript = []
        for call in calls:
            role = call.get("model_role", "default")
            transcript.append(f"CALL {call['call_id']} [{role}]")
            for message in call["messages"]:
                transcript.append(f"{message['role'].upper()}: {message['content']}")
            response = call["response"]
            if response.get("reasoning_content"):
                transcript.append(f"REASONING: {response['reasoning_content']}")
            transcript.append(f"ASSISTANT: {response.get('content', '')}")
            if call.get("transition"):
                transcript.append(
                    f"REWARD: {call['transition']['reward']:+.6f} "
                    f"INFO: {json.dumps(json_safe(call['transition']['info']), ensure_ascii=False)}"
                )
            transcript.append("")
        atomic_text(root / "transcript.txt", "\n".join(transcript))
        # This is the completion marker used by resume, so publish it only after every
        # auxiliary trajectory artifact is safely in place.
        atomic_json(root / "result.json", public)
        return root


__all__ = [
    "EpisodeRecorder",
    "EpisodeStore",
    "EvaluationClient",
    "EvaluationResponse",
    "RecordingEnv",
    "Usage",
    "atomic_bytes",
    "atomic_json",
    "atomic_text",
    "json_safe",
]
