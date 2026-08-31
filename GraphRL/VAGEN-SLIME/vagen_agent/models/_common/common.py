"""The model-family axis.

What a model family costs us is not its weights -- the trainer handles those -- but the
handful of places where *rendering a conversation into tokens* is family-specific. Those
places were previously scattered across the verl client and the agent loop, which meant
supporting a new family was a diff against two files that also contained the backend and
the trainer glue. Here they are one class with one registry, so a family is added without
touching anything else (see ``vagen_agent/README.md`` for the axes).

An adapter answers four questions:

    how does one message tokenize -- on its own, and at the head of a conversation?
    which ids stand in for a picture, and which bracket one?
    can this template render an isolated user turn?
    do the per-edge pieces compose into what the template would have produced?

The last one is the contract's teeth. Everything here is *derived* from the tokenizer
rather than tabulated, because a spelling table grows with every model and its misses are
silent -- but a derivation can be wrong too, and wrongly by a token or two at every turn
boundary, which rollout and training both see and therefore neither can detect. So an
adapter must be able to prove itself against a real render before a run starts. See
:meth:`ModelAdapter.self_test`.
"""

from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Callable
from typing import Any, NamedTuple

from slime.utils.misc import load_function

logger = logging.getLogger(__name__)

Msg = dict


class Rendered(NamedTuple):
    """What one render produced.

    A ``NamedTuple`` rather than a dataclass on purpose: ``Client.measure`` takes
    ``rendered[0] if isinstance(rendered, tuple) else rendered``, so a tuple subclass keeps
    the vendored client working untouched while still giving the fields names.
    """

    #: The tokens, with multimodal placeholders already expanded.
    ids: list[int]
    #: The raw frames this span carried, in placeholder order. The inference server needs
    #: these; they are re-sent on every call because it re-processes the whole prompt.
    images: list[Any]
    #: Everything else the processor produced -- ``pixel_values``, ``image_grid_thw``, ... .
    #:
    #: Accumulated per render rather than recomputed at the end. Running the processor a
    #: second time over the assembled conversation would be both expensive and a second
    #: source of truth: if the two passes tiled an image differently, the tensors handed to
    #: the trainer would not describe the sequence that was generated, and nothing checks.
    mm_train_inputs: dict[str, Any] | None = None
    #: The same span with multimodal placeholders left *unexpanded* -- one placeholder
    #: token per frame. ``None`` when there is nothing to expand, and then ``ids`` serves.
    #:
    #: Two forms are needed because the two consumers disagree, and only one of them can be
    #: satisfied by a single stream. **Training** wants the expanded ids, because that is
    #: the sequence ``mm_train_inputs`` describes and the sequence the model runs on. The
    #: **engine** wants the unexpanded ones: sglang decodes ``input_ids`` back to text and
    #: splits it on a single-token regex -- ``((?:<\|image_pad\|>))``, not a run -- then
    #: pulls one frame per match. Handing it an already-expanded span makes one 64-token
    #: image read as 64 images, and it dies indexing ``image_grid_thw[1]`` with one grid.
    #: slime's own rollout sidesteps this by posting ``text`` instead of ``input_ids``
    #: whenever images are attached (``sglang_rollout.py``); posting unexpanded ids is the
    #: same idea without giving up the exact prompt length the mask is measured against.
    engine_ids: list[int] | None = None


class ModelAdapterError(RuntimeError):
    """The adapter cannot render this model's conversations correctly.

    Fatal by design. Every failure this is raised for is a systematic one -- it applies to
    every turn of every episode -- so continuing produces a whole run of subtly
    off-distribution sequences rather than a few bad rows.
    """


def _flatten_token_value(value: Any) -> tuple[Any, ...]:
    """Normalise tokenizer token/id fields, some of which may be sequences."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


class ModelAdapter(ABC):
    """How one model family turns messages into tokens.

    Constructed with whatever the backend has: a tokenizer always, a processor when the
    model is multimodal. Holds no conversation state -- that is the message tree in
    ``vagen_agent.rollout.trajectory`` -- and never talks to an inference server -- that is
    ``vagen_agent.rollout.client``.
    """

    #: Whether this family can be handed pictures at all. Enforced by the renderer, which
    #: refuses images when there is no processor to expand them -- the alternative is an
    #: IndexError deep inside ``get_rope_index`` that names nothing.
    supports_images: bool = False

    #: Tokenizer roles that are protocol scaffolding rather than model output.  This is
    #: deliberately a role allow-list, not ``tokenizer.all_special_ids``: task vocabularies
    #: such as HID actions normally live in ``additional_special_tokens`` and must survive
    #: decoding.  A model-family adapter may extend or replace this tuple when its
    #: tokenizer exposes another named control role.
    generation_control_token_fields: tuple[str, ...] = (
        "pad_token",
        "eos_token",
        "bos_token",
        "mask_token",
        "sep_token",
        "cls_token",
        "image_token",
        "video_token",
        "audio_token",
        "vision_bos_token",
        "vision_eos_token",
        "audio_bos_token",
        "audio_eos_token",
        "audio_pad_token",
    )

    def __init__(self, tokenizer, processor=None, *,
                 apply_chat_template_kwargs: dict | None = None,
                 mm_processor_kwargs: dict | None = None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})

    # --------------------------------------------------------------- decoding
    def generation_control_token_ids(self) -> set[int]:
        """Ids to remove when reconstructing text generated by this model.

        Hugging Face's ``skip_special_tokens=True`` cannot express this policy: it also
        removes every task-specific token registered under ``additional_special_tokens``.
        Resolve only named structural roles, so a checkpoint can carry its own action
        vocabulary without requiring the environment to know its numeric ids.
        """
        ids: set[int] = set()
        token_map = getattr(self.tokenizer, "special_tokens_map", {}) or {}
        for field in self.generation_control_token_fields:
            value = token_map.get(field, getattr(self.tokenizer, field, None))
            for token in _flatten_token_value(value):
                token_id = self.tokenizer.convert_tokens_to_ids(str(token))
                if isinstance(token_id, int) and token_id >= 0:
                    ids.add(token_id)

            # Prefer the tokenizer's canonical id property when available.  This also
            # covers lightweight/custom tokenizers whose special_tokens_map is partial.
            value = getattr(self.tokenizer, f"{field}_id", None)
            for token_id in _flatten_token_value(value):
                if isinstance(token_id, int) and token_id >= 0:
                    ids.add(token_id)
        return ids

    def decode_generated(
        self,
        token_ids: list[int],
        *,
        preserve_special_tokens: tuple[str, ...] | list[str] = (),
    ) -> str:
        """Decode sampled ids, dropping model controls but preserving task tokens.

        ``preserve_special_tokens`` is an escape hatch for a task protocol whose token is
        assigned to a structural role by a particular tokenizer.  Normally it is empty:
        task tokens in ``additional_special_tokens`` are preserved by default.
        """
        preserve_ids = {
            token_id
            for token in preserve_special_tokens
            if isinstance(
                token_id := self.tokenizer.convert_tokens_to_ids(str(token)), int
            )
            and token_id >= 0
        }
        skip_ids = self.generation_control_token_ids() - preserve_ids
        values = [int(token_id) for token_id in token_ids if int(token_id) not in skip_ids]
        return self.tokenizer.decode(
            values,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    # ------------------------------------------------------------------ edges
    # A conversation is a path through a tree, so the renderer has to work an edge at a
    # time: each node owns its own token span, and a span is only reusable across
    # branches if it does not depend on what came after it. `render` above renders a
    # *span of new messages*; these two render the pieces that span is assembled from.
    #
    # The assembly rule (verified per family by `self_test`):
    #
    #     opening (mount at the root)   render_group(msgs)  then peel blocks off the right
    #     continuation                  [separator if the parent is a spliced assistant]
    #                                   + block(m)  for each new message
    #
    # An assistant node is never rendered: its tokens are the ones the model actually
    # sampled, spliced back verbatim. That is the whole reason this cannot be a single
    # whole-list render -- see `self_test` for the reasoning-model case that breaks it.

    def block(self, message: Msg, *, generation_prompt: bool = False) -> Rendered:
        """One message's self-contained block: no preamble, trailing separator included.

        Must satisfy, for any messages that can legally follow one another:
        ``block(a) + block(b) == render_group([a, b])`` -- except at the head, where a
        template may treat the first message specially (Qwen3-VL drops a *second* system
        message entirely, so a head system block cannot be derived this way). The head is
        therefore obtained by subtraction inside ``render_group``, not from here.
        """
        raise NotImplementedError

    def render_group(self, messages: list[Msg], *, generation_prompt: bool = True) -> Rendered:
        """The canonical whole-list render, used only for the call that opens a path."""
        raise NotImplementedError

    # ----------------------------------------------------------------- vision
    def image_placeholder_ids(self) -> set[int]:
        """Ids a picture sits behind. Empty for a text-only family."""
        return set()

    def vision_sentinel_ids(self) -> set[int]:
        """Ids bracketing a picture, if the family declares them.

        Not decoration: a cut that orphans a run from its opening sentinel makes the run
        stop counting as an image, and every position after it shifts -- silently, since
        the placeholder and feature counts still agree.
        """
        return set()

    # --------------------------------------------------------------- template
    def can_render_isolated_user_turn(self) -> bool:
        """Whether a lone user message can be rendered without the template refusing.

        ``CompactHarness`` needs this: its summary request goes out as a single user turn
        into an open conversation. Qwen3.5's template scans for a non-``<tool_response>``
        user message and calls ``raise_exception('No user query found in messages.')`` when
        it finds none, so on that family the answer depends on how the span is assembled.
        Declared rather than discovered, so ``harness=compact`` on a family that cannot do
        it is refused at startup instead of at the first compaction.
        """
        return True

    # ------------------------------------------------------------------ proof
    def self_test(self) -> None:
        """Prove that the pieces this adapter emits compose into what the template emits.

        One property, checked over several paths::

            concatenating the per-edge deltas of a path == the template's one-shot render
            of that path

        That is the contract the tree depends on. Every node owns its own span, spans are
        reused across branches, and an assistant's span is the tokens the model actually
        sampled -- so if the pieces do not compose, every turn boundary in every episode
        carries the difference. Because the rollout and the training pass see the same
        seam, nothing downstream can report it: it is off-distribution input, not a
        mismatch, which is why it needs a proof rather than a monitor.

        The paths are chosen for what they are each the only witness to:

        * ``[user]`` -- a single-message opening, where the group render is the whole span.
        * ``[system, user]`` -- the opening is split by peeling the tail off the right.
          Deriving the head instead does not work: Qwen3-VL's template drops a *second*
          system message, so a head system block derived from a placeholder exchange that
          already has one comes out empty and the prompt silently starts at the user turn.
        * ``[system, user2]`` mounted on the system node -- ``no_concat``'s shape, and the
          only path where a user block follows a system block directly. The old two-case
          contract (opening / after-assistant) had no witness for it at all.
        * two and three turns deep -- the separator after a spliced assistant response.
          The model stops *at* ``<|im_end|>`` and the template writes the ``\n`` after it,
          so the next edge has to supply it.

        Deliberately not checked: that the template's rendering of a *historical* assistant
        turn matches what the model emitted. On a reasoning model it does not, by design --
        Qwen3.5 strips the thinking block from history -- and at runtime we splice the
        sampled tokens rather than re-rendering, which is the point.
        """
        system = {"role": "system", "content": "SYS"}
        u1 = {"role": "user", "content": "U1"}
        a1 = {"role": "assistant", "content": "A1"}
        u2 = {"role": "user", "content": "U2"}
        u3 = {"role": "user", "content": "U3"}

        try:
            separator = self.message_separator()
            head = self._peeled_head([system, u1])
        except ModelAdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 - report it as the contract violation it is
            raise ModelAdapterError(
                f"{type(self).__name__} could not render the self-test conversation: "
                f"{exc!r}. The adapter cannot be trusted to tokenize an episode.") from exc

        # --- assistant-free paths: the assembly must equal the template exactly --------
        exact = {
            "[user]": ([u1], self.render_group([u1]).ids),
            "[system, user]": ([system, u1], head + self.block(u1, generation_prompt=True).ids),
            "[system, user2]  (no_concat: mounted on the system node)":
                ([system, u2], head + self.block(u2, generation_prompt=True).ids),
            "[system, user, user]":
                ([system, u1, u2],
                 head + self.block(u1).ids + self.block(u2, generation_prompt=True).ids),
            "[system, user, user, user]":
                ([system, u1, u2, u3],
                 head + self.block(u1).ids + self.block(u2).ids
                 + self.block(u3, generation_prompt=True).ids),
        }
        for label, (messages, assembled) in exact.items():
            self._require_equal(label, assembled, self._render_canonical(messages))

        # --- paths through an assistant: bracket it, do not reproduce it ---------------
        # What sits where the assistant goes is the tokens the model *sampled*, and on a
        # reasoning model that is deliberately not what the template writes for a
        # historical turn: Qwen3.5 strips the thinking block from history while the model
        # really did emit one. There is therefore no canonical render to compare the whole
        # sequence against -- and re-rendering history is precisely what this design
        # exists to avoid. What is still checkable, on every family, is the scaffolding:
        # everything we emit before the assistant, and everything we emit after it.
        canonical = self._render_canonical([system, u1, a1, u2])
        # Up to and *excluding* the generation prompt. On a reasoning model the generation
        # prompt carries content -- Qwen3.5's ends `<|im_start|>assistant\\n<think>\\n` --
        # and in a rendered history that tail is where the response text goes instead.
        # Requiring the generation prompt to be a prefix of the full render would reject
        # every reasoning model, which is the same trap the old two-case contract
        # documented and avoided.
        before = head + self.block(u1).ids
        after = list(separator) + self.block(u2, generation_prompt=True).ids
        if canonical[: len(before)] != before:
            self._require_equal("[system, user, asst, user] (prefix up to the response)",
                                before, canonical[: len(before)])
        if after and canonical[-len(after):] != after:
            self._require_equal("[system, user, asst, user] (suffix after the response)",
                                after, canonical[-len(after):])

        if separator and self.block(u2).ids[: len(separator)] == separator:
            raise ModelAdapterError(
                f"{type(self).__name__}: block() already begins with the message "
                f"separator, so the edge after a spliced assistant would emit it twice.")

    def _require_equal(self, label: str, assembled: list[int], canonical: list[int]) -> None:
        if assembled == canonical:
            return
        where = next((i for i in range(min(len(assembled), len(canonical)))
                      if assembled[i] != canonical[i]),
                     min(len(assembled), len(canonical)))
        raise ModelAdapterError(
            f"{type(self).__name__}: per-edge rendering does not compose for {label} -- "
            f"{len(assembled)} tokens assembled against {len(canonical)} from the "
            f"template, first differing at {where}.\n"
            f"  assembled [{max(0, where - 4)}:{where + 8}] = "
            f"{assembled[max(0, where - 4):where + 8]}\n"
            f"  template  [{max(0, where - 4)}:{where + 8}] = "
            f"{canonical[max(0, where - 4):where + 8]}\n"
            f"Every turn boundary in every episode would carry this, and since the "
            f"rollout and the training pass see the same seam, nothing downstream would "
            f"report it.")

    def _peeled_head(self, messages: list[Msg]) -> list[int]:
        """The first message's span, obtained the way the renderer obtains it."""
        whole = self.render_group(messages, generation_prompt=True)
        cut = len(whole.ids)
        for i in range(len(messages) - 1, 0, -1):
            block = self.block(messages[i], generation_prompt=(i == len(messages) - 1)).ids
            if whole.ids[cut - len(block):cut] != block:
                raise ModelAdapterError(
                    f"{type(self).__name__}: the opening render does not end with message "
                    f"{i}'s own block, so it cannot be split into per-message spans.")
            cut -= len(block)
        return list(whole.ids[:cut])

    def _render_canonical(self, messages: list[Msg]) -> list[int]:
        """What the template produces for the whole exchange in one go."""
        return list(self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False,
            **self.apply_chat_template_kwargs))

    #: Two message contents that share no tokens, used to find where content ends.
    _PROBE_A = "aaaaaaaa"
    _PROBE_B = "bbbbbbbb"

    def message_separator(self) -> list[int]:
        """What the template appends after a message that the model never generates.

        Qwen closes every message with ``<|im_end|>\\n``. The model stops *at*
        ``<|im_end|>``, so the newline is template output -- and a continuation that fails
        to supply it leaves the assembled sequence one token short of the canonical render
        at every single turn boundary. The rollout and the training pass see the same seam,
        so nothing downstream can report it: it is off-distribution input, not a mismatch.

        **Probed from the template, not read off ``eos_token_id``.** Render the same role
        twice with contents sharing no tokens; what the two renders have in common at the
        end is exactly what follows the content, i.e. ``[close, *separator]``. The close is
        the token the model emits to stop, so the separator is the rest.

        The obvious alternative -- take what follows ``eos_token_id``, then verify by
        diffing two template renders -- breaks on a **reasoning** model, and breaks
        *closed*. Qwen3.5's ``add_generation_prompt`` emits
        ``<|im_start|>assistant\\n<think>\\n`` while its render of a *historical* assistant
        turn carries no think block, so the diff mis-measures what the model emitted, the
        verification rejects a correct separator, and the method returns ``[]``. Measured on
        Qwen3.5-4B: the newline was dropped from every observation after the first, on the
        one model this port is being built for. Probing has no such dependency and yields
        ``[198]`` on Qwen2.5-VL, Qwen3-VL, Qwen3-Instruct-2507 and Qwen3.5 alike.

        One assumption remains: that the close is a **single** token at the head of the
        common suffix. It is checked as far as it can be -- a separator is template
        whitespace, never content -- and the result is logged, so a family where it does not
        hold is visible rather than silent.
        """
        if getattr(self, "_sep_cache", None) is not None:
            return self._sep_cache
        self._sep_cache: list[int] = []
        try:
            a = self._render_probe(self._PROBE_A)
            b = self._render_probe(self._PROBE_B)
            k = 0
            while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
                k += 1
            tail = a[len(a) - k:] if k else []
            if len(tail) < 2:
                # Nothing after the close. A template that runs messages together needs no
                # separator, and inventing one would splice tokens that do not belong.
                return self._sep_cache
            candidate = tail[1:]
            decoded = self.tokenizer.decode(candidate)
            if decoded.strip() != "":
                logger.warning(
                    "message-separator probe produced %r (%r), which is not whitespace -- "
                    "the 'close is one token' assumption does not hold for this template. "
                    "Falling back to no separator; continuations will be %d token(s) short.",
                    candidate, decoded, len(candidate))
                return self._sep_cache
            self._sep_cache = candidate
            logger.info("message separator for %s: %r (%r)",
                        type(self).__name__, candidate, decoded)
        except Exception as exc:  # noqa: BLE001 - a failed derivation must not stop a run
            logger.warning("could not derive the message separator (%s); continuations will "
                           "be one token short of the canonical render", exc)
        return self._sep_cache

    def _render_probe(self, content: str) -> list[int]:
        """One user message through the chat template, no generation prompt."""
        return list(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=False,
            tokenize=True, return_dict=False, **self.apply_chat_template_kwargs))


# --------------------------------------------------------------------- registry
MODEL_ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_model(*names: str) -> Callable[[type], type]:
    """Register an adapter under one or more model-family names.

    Refuses to rebind a name to a *different* class: a silent rebinding means a run reports
    the family it was configured with and tokenizes as another one. Re-registering the same
    class is fine, since a module is legitimately imported more than once.
    """
    def decorator(cls: type) -> type:
        _require_adapter(cls, "/".join(names))
        for name in names:
            existing = MODEL_ADAPTERS.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"model adapter {name!r} is already registered to "
                    f"{existing.__qualname__}; pick another name rather than shadowing it."
                )
            MODEL_ADAPTERS[name] = cls
        return cls
    return decorator


def resolve_model_adapter(name: str) -> type[ModelAdapter]:
    """The class for ``name``: a registered name, or a ``module:Class`` import path.

    The import path exists for the same reason it does for harnesses: a new family is
    often tried from a config before anyone wants a decorator in this package.
    """
    if name in MODEL_ADAPTERS:
        return MODEL_ADAPTERS[name]
    if "." not in name and ":" not in name:
        raise ValueError(
            f"unknown model adapter {name!r}; choose from {sorted(MODEL_ADAPTERS)}, or give "
            f"an import path like 'mypkg.adapters:MyAdapter'."
        )
    # slime resolves every one of its own plugin points (--custom-*-path) with
    # load_function, so an import path here behaves the same way and fails the same
    # way as one passed to slime directly.
    try:
        cls = load_function(name.replace(":", "."))
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f"could not import model adapter {name!r}: {exc}") from exc
    _require_adapter(cls, name)
    return cls


def build_model_adapter(name: str, tokenizer, processor=None, **kwargs) -> ModelAdapter:
    """Instantiate by name or import path. Does **not** run ``self_test``.

    Separated because the proof needs a tokenizer that is fully loaded and is worth paying
    for exactly once per process, at startup, where a failure can still stop the run --
    see ``ModelAdapter.self_test``.
    """
    return resolve_model_adapter(name)(tokenizer, processor, **kwargs)


def _require_adapter(cls, name: str) -> None:
    if not (isinstance(cls, type) and issubclass(cls, ModelAdapter)):
        raise TypeError(
            f"{name} resolved to {cls!r}, which does not subclass ModelAdapter -- so "
            f"render / image_placeholder_ids / self_test are not guaranteed."
        )


__all__ = [
    "MODEL_ADAPTERS",
    "ModelAdapter",
    "ModelAdapterError",
    "build_model_adapter",
    "register_model",
    "resolve_model_adapter",
]
