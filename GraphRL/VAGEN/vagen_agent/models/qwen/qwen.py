"""Qwen-family rendering: Qwen2.5-VL, Qwen3-VL, Qwen3.5.

Ported from VAGEN's ``agent_loop/verl_client.py``, which had this logic tangled with the
verl server call. The tangle is why it lives here now: the rendering is a property of the
model, the server call is a property of the backend, and supporting a new family should
not be a diff against a file that also knows what a ``request_id`` is.

The whole difficulty is **continuations**. A chat template emits a preamble before the
first message, so rendering a message that continues a conversation as if it began one
splices that preamble into the middle of the sequence. The fix is to render it behind a
placeholder exchange and strip exactly what the placeholder cost -- and the two
subtleties below are what make "exactly" hard.
"""

from __future__ import annotations

import logging
from typing import Any

from vagen_agent.models._common.common import (
    ModelAdapter,
    ModelAdapterError,
    Rendered,
    register_model,
)
from vagen_agent.models._common.image_tokens import image_token_ids, vision_sentinel_ids

logger = logging.getLogger(__name__)


#: Prepended to a continuation span so the template emits no preamble of its own, then
#: stripped. Every use must render the same turns or the strip is the wrong length.
#:
#: **The system turn** suppresses the template's default system block ("You are a helpful
#: assistant."), which would otherwise appear in the continuation render but not in the
#: prefix, leaving its tail spliced ahead of every observation.
#:
#: **The user turn** is for Qwen3.5, whose template scans for a non-``<tool_response>``
#: user message and calls ``raise_exception('No user query found in messages.')`` when
#: there is none -- so a system turn alone cannot even be rendered to compute the prefix,
#: let alone prepended to an assistant-only span. Verified on Qwen2.5-VL, Qwen3-VL and
#: Qwen3.5: with both turns the prefix is 12 tokens on all three, no default system block
#: is injected anywhere, and ``render([*placeholder, *delta])`` begins with
#: ``render(placeholder)`` for assistant-only, user-only and mixed spans.
_PLACEHOLDER_TURNS = [
    {"role": "system", "content": "placeholder"},
    {"role": "user", "content": "placeholder"},
]


#: Processor outputs that describe the *token sequence* rather than the frames, and so
#: must not travel as multimodal training inputs.
#:
#: `input_ids` / `attention_mask` are obvious. `mm_token_type_ids` is the one that is not:
#: it is shaped [1, n_tokens] and marks which positions are image tokens. Carried along, it
#: survives as far as slime's `get_batch`, which concatenates every multimodal tensor along
#: dim 0 to pack a micro-batch -- and two rows with different prompt lengths then fail with
#: "Sizes of tensors must match except in dimension 0. Expected size 114 but got size 98".
#: Even if the shapes happened to agree it would be wrong, because after packing it
#: describes an unpacked single sample. The model derives it from the packed `input_ids`.
_TOKEN_ALIGNED = ("input_ids", "attention_mask", "mm_token_type_ids")


def _parts(message: dict) -> list[dict]:
    content = message.get("content", "")
    return content if isinstance(content, list) else [{"type": "text", "text": str(content)}]


def _text_only(message: dict) -> str:
    return "".join(part.get("text", "") for part in _parts(message))


def _images_of(message: dict) -> list[Any]:
    """Frames carried alongside a message, in the order their placeholders appear."""
    return list(message.get("images") or [])


@register_model("qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen")
class QwenAdapter(ModelAdapter):
    """Rendering for the Qwen chat template, with or without a vision processor."""

    #: Decided at construction: a processor means the model can be handed pictures. Read
    #: rather than declared per-family, because the same adapter serves the text-only and
    #: multimodal checkpoints of the same family and getting it from the class would make
    #: the answer depend on which name the config happened to use.
    @property
    def supports_images(self) -> bool:  # type: ignore[override]
        return self.processor is not None

    # ------------------------------------------------------------------- edges
    def _render_flat(self, flat: list[dict], images: list, generation_prompt: bool) -> Rendered:
        """One template application over already-flattened messages."""
        if self.processor is not None:
            text = self.processor.apply_chat_template(
                flat, add_generation_prompt=generation_prompt, tokenize=False,
                **self.apply_chat_template_kwargs)
            inputs = self.processor(text=[text], images=images or None, return_tensors="pt",
                                    **self.mm_processor_kwargs)
            ids = inputs["input_ids"].squeeze(0).tolist()
            engine_ids = list(self.tokenizer.encode(text, add_special_tokens=False)) if images else None
            mm_train = ({k: v for k, v in inputs.items()
                         if k not in _TOKEN_ALIGNED} or None) if images else None
            return Rendered(ids, list(images), mm_train, engine_ids)
        ids = list(self.tokenizer.apply_chat_template(
            [{"role": m["role"], "content": _text_only(m)} for m in flat],
            add_generation_prompt=generation_prompt, tokenize=True, return_dict=False,
            **self.apply_chat_template_kwargs))
        return Rendered(ids, [], None, None)

    def _check_images(self, images) -> None:
        if images and self.processor is None:
            raise ModelAdapterError(
                "the environment produced images but this model was built without a "
                "processor, so there is nothing to expand the placeholders. Either point "
                "the run at a multimodal checkpoint or switch the environment to a text "
                "render mode.")

    def block(self, message: dict, *, generation_prompt: bool = False) -> Rendered:
        images = _images_of(message)
        self._check_images(images)
        flat = [*self._placeholder_flat(), {"role": message["role"], "content": _parts(message)}]
        whole = self._render_flat(flat, images, generation_prompt)
        prefix = self._template_prefix()
        if whole.ids[: len(prefix)] != prefix:
            raise ModelAdapterError(
                "the chat template did not begin this message's render with the "
                "placeholder exchange, so stripping a fixed length would corrupt the "
                "block. This family needs its own adapter.")
        # The placeholder is text, so it carries no frames: the same prefix length trims
        # both streams, and `mm_train_inputs` already describes only `images`.
        engine = whole.engine_ids[len(prefix):] if whole.engine_ids is not None else None
        return Rendered(whole.ids[len(prefix):], images, whole.mm_train_inputs, engine)

    def render_group(self, messages: list[dict], *, generation_prompt: bool = True) -> Rendered:
        images = [img for m in messages for img in _images_of(m)]
        self._check_images(images)
        flat = [{"role": m["role"], "content": _parts(m)} for m in messages]
        return self._render_flat(flat, images, generation_prompt)

    # ------------------------------------------------------------------ vision
    def image_placeholder_ids(self) -> set[int]:
        return self._vision_ids()[0]

    def vision_sentinel_ids(self) -> set[int]:
        return self._vision_ids()[1]

    def _vision_ids(self):
        """Cached: a property of the model, and this is asked once per row."""
        if getattr(self, "_vision_cache", None) is None:
            source = self.processor or self.tokenizer
            self._vision_cache = ((image_token_ids(source), vision_sentinel_ids(source))
                                  if source is not None else (set(), set()))
        return self._vision_cache

    # ---------------------------------------------------------------- template
    def can_render_isolated_user_turn(self) -> bool:
        """True: the placeholder exchange already carries a user turn, which is exactly
        what Qwen3.5's template demands. A summary request therefore renders on every
        family this adapter serves."""
        return True

    def _placeholder_flat(self) -> list[dict]:
        """The placeholder in the same parts-list shape as every other rendered message.

        The shape matters: a template can tokenize a plain string differently from a
        one-element parts list, and then the strip is measured against a render that never
        happened.
        """
        return [{"role": t["role"], "content": _parts(t)} for t in _PLACEHOLDER_TURNS]

    def _continuation_prefix(self) -> list[int]:
        """Exactly what the placeholder exchange contributes ahead of a continuation.

        The template's own preamble, *minus the separator*. The model stops at
        ``<|im_end|>`` and the template adds the ``\\n`` after it; stripping the placeholder
        whole would take that newline away, leaving every continuation one token short of
        the canonical render. Rollout and training see the same seam, so nothing
        downstream can tell -- it is off-distribution input, not a mismatch.
        """
        if getattr(self, "_cont_prefix_cache", None) is None:
            separator = self.message_separator()
            prefix = self._template_prefix()
            self._cont_prefix_cache = prefix[: -len(separator)] if separator else prefix
        return self._cont_prefix_cache

    def _template_prefix(self) -> list[int]:
        """Tokens the template emits for the placeholder exchange, generation prompt off.

        Cached: rendering it costs a template application and it never changes.
        """
        if getattr(self, "_prefix_cache", None) is None:
            placeholder = self._placeholder_flat()
            if self.processor is not None:
                text = self.processor.apply_chat_template(
                    placeholder, add_generation_prompt=False, tokenize=False,
                    **self.apply_chat_template_kwargs
                )
                self._prefix_cache = self.processor(
                    text=[text], return_tensors="pt")["input_ids"].squeeze(0).tolist()
            else:
                # `_text_only`, as on every other tokenizer-path render. A bare tokenizer's
                # template concatenates `message['content']` as a string, so handing it the
                # parts list raises TypeError -- which is what a text-only model did here,
                # on the first continuation, while both other call sites converted.
                self._prefix_cache = list(self.tokenizer.apply_chat_template(
                    [{"role": m["role"], "content": _text_only(m)} for m in placeholder],
                    add_generation_prompt=False, tokenize=True, return_dict=False,
                    **self.apply_chat_template_kwargs,
                ))
        return self._prefix_cache

    # message_separator / probing lives on ModelAdapter: it is derived from the chat
    # template, not from anything Qwen-specific, and the eos-based derivation this
    # class used to carry silently returned [] on Qwen3.5. See the base class.


__all__ = ["QwenAdapter"]
