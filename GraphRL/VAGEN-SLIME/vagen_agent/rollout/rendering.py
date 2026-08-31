"""Rendering one edge at a time.

A conversation is a path through a tree, so every node has to own its own token span:
a span that depends on what came after it cannot be shared between two branches, and
``no_concat`` shares the system prompt across every turn of the episode.

Two cases, and the difference is only at the head:

**Opening** (mounting at the root). Render the whole group canonically, then peel each
message's block off the right; whatever is left is the first message's span. Peeling
rather than deriving is not fussiness -- ``block()`` gets the head wrong on at least one
shipped family. Qwen3-VL's template drops a *second* system message, so deriving a system
block from a placeholder exchange that already contains one yields zero tokens, and the
whole prompt then starts at the first user turn. Peeling cannot make that mistake, and it
self-checks: the tail either matches or we raise.

**Continuation**. Each new message is ``block(m)``, plus the message separator when the
parent is a generated node. The model stops *at* ``<|im_end|>`` and the template writes
the ``\\n`` after it, so that newline is template output the next edge has to supply.
Without it every turn boundary in every episode is one token short of the canonical
render -- and since the rollout and the training pass see the same seam, nothing
downstream can report it.

The generation prompt goes on the last new message only, which is always the node a
generated node attaches under, so it is never stale on a path anyone extends.
"""

from __future__ import annotations

from vagen_agent.models._common.common import ModelAdapterError
from vagen_agent.rollout.trajectory import Span


class EdgeRenderer:
    """Turns messages into per-node spans, with a cache keyed by (context, message)."""

    def __init__(self, model, frames):
        self.model = model
        self.frames = frames
        self._cache: dict[tuple, Span] = {}

    # ------------------------------------------------------------------ public
    def render(self, messages: list[dict], *, opening: bool, after_generated: bool) -> list[Span]:
        """One ``Span`` per message, in order."""
        if not messages:
            return []
        spans = self._group(messages) if opening else self._blocks(messages, after_generated)
        if len(spans) != len(messages):
            raise ModelAdapterError(
                f"renderer produced {len(spans)} spans for {len(messages)} messages")
        return spans

    # ------------------------------------------------------------------ opening
    def _group(self, messages: list[dict]) -> list[Span]:
        whole = self.model.render_group(messages, generation_prompt=True)
        if len(messages) == 1:
            return [self._span(whole, messages[0])]

        # Peel from the right. Each tail block is self-contained, so what remains after
        # peeling every message but the first is exactly the first message's span.
        tails: list[Span] = []
        cut_ids = len(whole.ids)
        cut_engine = len(whole.engine_ids) if whole.engine_ids is not None else None
        for i in range(len(messages) - 1, 0, -1):
            last = i == len(messages) - 1
            block = self.model.block(messages[i], generation_prompt=last)
            n = len(block.ids)
            if whole.ids[cut_ids - n: cut_ids] != block.ids:
                raise ModelAdapterError(
                    f"the opening render does not end with message {i}'s own block "
                    f"({messages[i].get('role')}), so the group cannot be split into "
                    f"per-message spans. This family needs its own adapter.")
            cut_ids -= n
            if cut_engine is not None and block.engine_ids is not None:
                cut_engine -= len(block.engine_ids)
            elif cut_engine is not None:
                cut_engine -= n
            tails.append(self._span(block, messages[i]))
        tails.reverse()

        head = Span(
            ids=list(whole.ids[:cut_ids]),
            engine_ids=list((whole.engine_ids or whole.ids)[:cut_engine if cut_engine is not None else cut_ids]),
            frames=self.frames.intern_all(messages[0].get("images")),
            mm_train=None,
        )
        # The head's own frames were expanded inside the group render, so its processor
        # tensors are not separable; a head that carries pictures is refused rather than
        # guessed at. No shipped environment puts one in the system prompt.
        if head.frames:
            raise ModelAdapterError(
                "the first message of a conversation carries images. The opening is split "
                "by peeling blocks off the right, which cannot separate the head's "
                "processor tensors from the group's. Put the frames in the observation "
                "rather than the system prompt.")
        return [head, *tails]

    # ------------------------------------------------------------- continuation
    def _blocks(self, messages: list[dict], after_generated: bool) -> list[Span]:
        spans = []
        for i, message in enumerate(messages):
            last = i == len(messages) - 1
            key = (self._content_key(message), last, after_generated and i == 0)
            span = self._cache.get(key)
            if span is None:
                block = self.model.block(message, generation_prompt=last)
                span = self._span(block, message)
                if after_generated and i == 0:
                    sep = self.model.message_separator()
                    span = Span(ids=list(sep) + span.ids,
                                engine_ids=list(sep) + span.engine_ids,
                                frames=span.frames, mm_train=span.mm_train)
                self._cache[key] = span
            spans.append(span)
        return spans

    # ------------------------------------------------------------------ helpers
    def _span(self, rendered, message: dict) -> Span:
        return Span(
            ids=list(rendered.ids),
            engine_ids=list(rendered.engine_ids if rendered.engine_ids is not None
                            else rendered.ids),
            frames=self.frames.intern_all(getattr(rendered, "images", None)),
            mm_train=rendered.mm_train_inputs,
        )

    def _content_key(self, message: dict) -> tuple:
        content = message.get("content", "")
        if isinstance(content, list):
            content = tuple(
                (p.get("type"), p.get("text", "")) if isinstance(p, dict) else str(p)
                for p in content)
        keys = tuple(self.frames.key_of(i) for i in (message.get("images") or ()))
        return (message.get("role"), content, keys)


__all__ = ["EdgeRenderer"]
