"""The layer that knows about tokens, and the only one.

The harness sees two methods -- ``create`` and ``size`` -- and gets back text plus a usage
count. Everything token-level happens here: routing a message list onto the tree, rendering
the new edges, splicing the sampled ids of history back verbatim, POSTing, recording.

There is deliberately no per-model branching. If you find yourself writing
``if self.model_name ==``, the thing you are branching on belongs in a
:class:`~vagen_agent.models._common.common.ModelAdapter`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vagen_agent.rollout.frames import FrameTable
from vagen_agent.rollout.rendering import EdgeRenderer
from vagen_agent.rollout.trajectory import Generated, Record, Span, TreeInvariant

logger = logging.getLogger(__name__)

#: Per-episode ceilings. These replace `MAX_CALLS_PER_TURN` and the runner's
#: `for _ in range(max_turns)`, and they are strictly stronger: a harness that calls
#: `create()` in a loop without ever stepping the environment -- the failure the old
#: constant was added for -- is bounded by neither of those but is bounded by these.
MAX_CALLS_PER_EPISODE = 200
MAX_TOKENS_PER_EPISODE = 1_000_000


class EpisodeBudgetExceeded(RuntimeError):
    """A harness kept generating without finishing. Names both counters."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Response:
    """What the harness gets back. Text plus what it needs to budget and to act."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    #: Which call this was. Client-internal: the scoring seam files the reward under it.
    #: The env contract declares this off-limits (see ``envs/_common/common.py``).
    call_id: int = 0


class RolloutClient:
    """Conversation routing, edge rendering, and slime's sglang router."""

    def __init__(self, model, record: Record, *, args=None, sampling_params: dict | None = None,
                 frames: FrameTable | None = None,
                 response_decoder: Callable[[list[int], str], str] | None = None,
                 max_calls: int = MAX_CALLS_PER_EPISODE,
                 max_tokens: int = MAX_TOKENS_PER_EPISODE):
        self.model = model
        self.record = record
        self.args = args
        self.sampling_params = dict(sampling_params or {})
        self.frames = frames if frames is not None else FrameTable()
        self.renderer = EdgeRenderer(model, self.frames)
        self.response_decoder = response_decoder
        self.url = (f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
                    if args is not None else None)
        self.n_calls = 0
        self.n_tokens = 0
        self._max_calls, self._max_tokens = max_calls, max_tokens

    @property
    def tokenizer(self):
        return self.model.tokenizer

    # -------------------------------------------------------------------- create
    async def create(self, messages: list[dict], **sampling) -> Response:
        self._charge()
        node, new_messages = self.record.mount(messages)
        held_ids, held_engine, held_frames = self.record.tokens_at(node)
        spans = self.renderer.render(new_messages, opening=node.is_root,
                                     after_generated=node.is_generated)

        parent = node
        for message, span in zip(new_messages, spans, strict=True):
            parent = self.record.add_context_node(parent, message, span)

        prompt_ids = held_ids + [t for s in spans for t in s.ids]
        engine_ids = held_engine + [t for s in spans for t in s.engine_ids]
        frame_keys = held_frames + [k for s in spans for k in s.frames]

        # Held tokens must be an exact prefix of what we are about to send. If they are
        # not, an edge rendered differently this time than last -- which is the one class
        # of bug that both the rollout and the training pass would see identically, so
        # nothing downstream could report it. slime's agent stack heals this at token
        # level (CLEAN/REALIGN/FORK) because it re-renders history; we splice history
        # verbatim, so here it can only be our renderer being wrong.
        if prompt_ids[: len(held_ids)] != held_ids:
            raise TreeInvariant(
                f"edge rendering is not prefix-stable: the {len(held_ids)} tokens held at "
                f"the mount point are not a prefix of this call's {len(prompt_ids)}-token "
                f"prompt. Check the model adapter's block()/render_group() against "
                f"self_test.")

        out = await self._post(engine_ids, self.frames.b64_of(frame_keys),
                               {**self.sampling_params, **sampling})

        self.n_calls += 1
        self.n_tokens += len(prompt_ids) + len(out["token_ids"])
        call_id = self.n_calls
        response_text = out["text"] if isinstance(out["text"], str) else ""
        if self.response_decoder is not None:
            response_text = self.response_decoder(list(out["token_ids"]), response_text)
        gen = Generated(call_id=call_id, output_ids=list(out["token_ids"]),
                        logprobs=list(out["logprobs"] or [0.0] * len(out["token_ids"])),
                        finish_reason=out.get("stop_reason"))
        # The assistant node carries the sampled ids on both streams: a response has no
        # multimodal placeholders, so there is nothing to expand.
        self.record.add_generated(
            parent, {"role": "assistant", "content": response_text}, gen,
            Span(ids=list(out["token_ids"]), engine_ids=list(out["token_ids"])))
        if out.get("stop_reason") == "abort":
            self.record.aborted = True

        usage = out.get("usage") or {}
        return Response(text=response_text,
                        token_ids=list(out["token_ids"]),
                        logprobs=list(out["logprobs"] or []),
                        stop_reason=out.get("stop_reason"),
                        usage=Usage(int(usage.get("prompt_tokens", len(prompt_ids))),
                                    int(usage.get("completion_tokens", len(out["token_ids"])))),
                        call_id=call_id)

    def _charge(self) -> None:
        if self.n_calls >= self._max_calls:
            raise EpisodeBudgetExceeded(
                f"{self.n_calls} model calls in one episode (limit {self._max_calls}). "
                f"A harness that generates without ever stepping the environment looks "
                f"exactly like this.")
        if self.n_tokens >= self._max_tokens:
            raise EpisodeBudgetExceeded(
                f"{self.n_tokens} tokens in one episode (limit {self._max_tokens}).")

    # --------------------------------------------------------------------- size
    def size(self, messages: list[dict]) -> int:
        """What this message list would cost, without recording anything.

        Side-effect free by construction: rendering is separable from mounting, and the
        frame table is content-addressed, so measuring cannot ship a picture twice.
        """
        node, new_messages = self.record.mount(messages)
        held_ids, _, _ = self.record.tokens_at(node)
        spans = self.renderer.render(new_messages, opening=node.is_root,
                                     after_generated=node.is_generated)
        return len(held_ids) + sum(len(s.ids) for s in spans)

    # --------------------------------------------------------------------- post
    async def _post(self, engine_ids: list[int], image_data: list[str],
                    sampling: dict) -> dict[str, Any]:
        """One POST to slime's sglang router.

        No retry on an empty generation. Under slime an empty generation is what a
        **weight update** looks like from inside a call -- the rollout manager posts
        ``abort_request`` and waits for the server to drain -- so retrying either fails
        again against a server that is deliberately refusing, or succeeds against *new
        weights*, splicing two policies into one episode with nothing recording it. slime
        already has the right answer: surface ``ABORTED`` and let the worker re-queue.

        The unexpanded stream goes over the wire; sglang expands the placeholders itself
        and swaps in its content-hash pad values, which is what makes its prefix cache
        distinguish two different pictures. See ``Rendered.engine_ids``.
        """
        from slime.utils.http_utils import post

        payload = {"input_ids": list(engine_ids), "sampling_params": sampling,
                   "return_logprob": True}
        if image_data:
            payload["image_data"] = image_data
        raw = await post(self.url, payload)
        meta = raw.get("meta_info") or {}
        triples = meta.get("output_token_logprobs") or []
        return {
            "text": raw.get("text") or "",
            # Ids from the server's own record, not by re-tokenizing the text: BPE is not
            # compositional, so a re-encoded response can split differently from how it
            # was sampled, and the mask, the reward placement and the log-probs would all
            # align to a sequence that was never generated.
            "token_ids": [int(t[1]) for t in triples],
            "logprobs": [float(t[0]) for t in triples] or None,
            "stop_reason": (meta.get("finish_reason") or {}).get("type"),
            "usage": {"prompt_tokens": len(engine_ids), "completion_tokens": len(triples)},
        }


__all__ = ["EpisodeBudgetExceeded", "Response", "RolloutClient", "Usage"]
