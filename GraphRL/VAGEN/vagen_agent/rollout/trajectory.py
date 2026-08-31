"""The message tree, and turning it into slime ``Sample``s.

Vendored from ``slime/slime/agent/trajectory.py`` and cut down. slime built this because
its agent is a black box behind an HTTP endpoint: it sees a stream of ``messages`` lists
and has to *infer* the conversation structure from how they overlap. We adopt the same
inference deliberately -- it is what lets a harness be ordinary agent code instead of
something that has to declare its context policy -- so we inherit the same shape.

What was dropped, and why: slime carries a token-level drift classifier
(``CLEAN / REALIGN / FORK``) because its adapter re-renders history through the served
chat template, and a replayed assistant turn does not re-render to the tokens that were
sampled. We render one edge at a time and splice the sampled ids back verbatim, so held
tokens are always an exact prefix. The classifier would return CLEAN every time; instead
of carrying it, ``RolloutClient.create`` asserts the prefix property. Drift is then a loud bug
in our renderer rather than something silently healed.

What was added: a per-node reward vector, multimodal frames that follow the token stream,
and leaf filtering.

Vocabulary. A **node** is one message. A **path** from the root to a leaf is one real
conversation -- every message on it was in the context of the next -- and therefore one
training row. Two paths that share a prefix share those tokens, but a *generated* response
on a shared prefix is trained only once (see ``response_trained``); the rest re-emit it as
context. Gradient is deduplicated; context is not.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any


class TreeInvariant(RuntimeError):
    """The tree stopped describing a sequence the model could have seen."""


# --------------------------------------------------------------------------- nodes
@dataclasses.dataclass
class Span:
    """What one node contributes to the token stream.

    Two streams, because the two consumers disagree -- see ``Rendered.engine_ids``.
    ``ids`` is what the trainer sees (multimodal placeholders expanded);
    ``engine_ids`` is what gets POSTed (unexpanded, sglang expands it itself).
    """

    ids: list[int] = dataclasses.field(default_factory=list)
    engine_ids: list[int] = dataclasses.field(default_factory=list)
    frames: list[str] = dataclasses.field(default_factory=list)      # frame-table hashes
    mm_train: dict[str, Any] | None = None


@dataclasses.dataclass
class Generated:
    """The payload of a node the model actually produced."""

    call_id: int
    output_ids: list[int]
    logprobs: list[float]
    finish_reason: str | None = None
    #: Filled by the scoring seam, keyed by call_id in ``Record.rewards``; mirrored here
    #: only for the dump.
    info: dict[str, Any] | None = None


class MessageNode:
    def __init__(self, role: str | None = None, message: dict | None = None,
                 parent: "MessageNode | None" = None):
        self.role = role
        self.message = message
        self.parent = parent
        self.children: list[MessageNode] = []
        self.span = Span()
        #: Set only on nodes the model generated. A node with ``gen is None`` came from a
        #: prompt and exists to route.
        self.gen: Generated | None = None
        #: Shared by sibling paths; the first to reach it trains it, the rest re-emit it
        #: as loss_mask=0 context -- so each response is trained exactly once.
        self.response_trained = False

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_generated(self) -> bool:
        return self.gen is not None

    def add_child(self, child: "MessageNode") -> "MessageNode":
        child.parent = self
        self.children.append(child)
        return child

    def path(self) -> list["MessageNode"]:
        """Root-exclusive, ordered from the first real message down to self."""
        chain, node = [], self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator["MessageNode"]:
        if not self.children:
            yield self
            return
        for child in self.children:
            yield from child.leaves()


# -------------------------------------------------------------------------- record
def _key(message: dict) -> tuple:
    """Identity of a message for routing.

    Frames are compared by their content hash, not by object identity: an environment that
    re-renders the same board must not look like a different message, or the episode forks
    every turn and quietly becomes N one-turn rows. Text compares by value.
    """
    content = message.get("content", "")
    if isinstance(content, list):
        content = tuple(
            (part.get("type"), part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content)
    frames = tuple(message.get("frame_keys") or ())
    return (message.get("role"), content, frames)


@dataclasses.dataclass
class Record:
    """Everything one episode produced. Held by the client, consumed by ``assemble``."""

    episode_id: str = "episode"
    root: MessageNode = dataclasses.field(default_factory=MessageNode)
    #: call_id -> reward vector, length == len(output_ids). Written by the scoring seam.
    #: **Absence means this call was never cashed in** -- no env step happened on it --
    #: which is the leaf filter's criterion.
    rewards: dict[int, list[float]] = dataclasses.field(default_factory=dict)
    #: Episode-level facts, filled by the scoring seam.
    turns: int = 0
    terminated: bool = False
    truncated: bool = False
    aborted: bool = False
    _rewards_order: list[int] = dataclasses.field(default_factory=list)

    @property
    def total_reward(self) -> float:
        return float(sum(sum(v) for v in self.rewards.values()))

    # -------------------------------------------------------------------- routing
    def mount(self, messages: list[dict]) -> tuple[MessageNode, list[dict]]:
        """Walk down matching messages; return the deepest match and the unmatched tail.

        Assistant nodes are skipped over rather than matched: the harness replays the
        assistant text it was handed, and whether that text round-trips is not something
        the routing should depend on. What identifies a path is the *prompt* messages on
        it plus which generated node it descends from.
        """
        node, i = self.root, 0
        while i < len(messages):
            message = messages[i]
            if message.get("role") == "assistant":
                # Match it against the generated child we already hold, if any.
                nxt = next((c for c in node.children if c.is_generated), None)
                if nxt is None:
                    break
                node, i = nxt, i + 1
                continue
            nxt = next((c for c in node.children
                        if not c.is_generated and _key(c.message or {}) == _key(message)), None)
            if nxt is None:
                break
            node, i = nxt, i + 1
        return node, messages[i:]

    def add_context_node(self, parent: MessageNode, message: dict, span: Span) -> MessageNode:
        node = parent.add_child(MessageNode(message.get("role"), message))
        node.span = span
        return node

    def add_generated(self, parent: MessageNode, message: dict, gen: Generated,
                      span: Span) -> MessageNode:
        node = parent.add_child(MessageNode("assistant", message))
        node.gen = gen
        node.span = span
        return node

    def tokens_at(self, node: MessageNode) -> tuple[list[int], list[int], list[str]]:
        ids: list[int] = []
        engine: list[int] = []
        frames: list[str] = []
        for step in node.path():
            ids += step.span.ids
            engine += step.span.engine_ids
            frames += step.span.frames
        return ids, engine, frames


# ------------------------------------------------------------------------ builder
class SampleBuilder:
    """Accumulates one path's nodes into the token stream of one Sample."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.scores: list[float] = []
        self.frames: list[str] = []
        self.mm_train: list[dict] = []
        #: Tokens before the first generated response. Stripped from the mask on the way
        #: out, so mask/logprobs/scores cover only the response region -- slime's contract.
        self.leading_prompt_len: int | None = None

    def add_context(self, span: Span) -> None:
        self.tokens += span.ids
        self.frames += span.frames
        if span.mm_train:
            self.mm_train.append(span.mm_train)
        if self.leading_prompt_len is None:
            return                      # still the opening prompt; no mask yet
        n = len(span.ids)
        self.loss_mask += [0] * n
        self.logprobs += [0.0] * n
        self.scores += [0.0] * n

    def add_response(self, ids: list[int], logprobs: list[float], scores: list[float],
                     *, trained: bool) -> None:
        if self.leading_prompt_len is None:
            self.leading_prompt_len = len(self.tokens)
        n = len(ids)
        self.tokens += list(ids)
        self.loss_mask += [1 if trained else 0] * n
        self.logprobs += list(logprobs) if trained else [0.0] * n
        self.scores += list(scores) if trained else [0.0] * n

    def has_trained_response(self) -> bool:
        return any(self.loss_mask)

    @property
    def response_scores(self) -> list[float]:
        return self.scores


# ----------------------------------------------------------------------- assemble
def assemble(record: Record, source: Any, *, algorithm: str = "default_gae",
             frames: Any = None, status: Any = None, metadata_extra: dict | None = None,
             fold=None) -> list:
    """The tree, as the training rows it produced.

    Order matters twice:

    1. **Leaves are filtered before any of them claims a shared prefix.** ``leaves()``
       walks children in insertion order, so under a retry the abandoned response is
       visited first and would claim the prefix it shares with the response that replaced
       it. Dropping that row afterwards would take the shared prefix's gradient with it --
       and that part is real data.
    2. ``Sample.reward`` is computed **before** the cross-row fold. Under
       ``token_level_gae`` a folded row sums to more than the episode total (every row has
       to see the whole future independently), so ``sum(folded)`` is not the episode's
       worth.
    """
    from vagen_agent.algorithms import episode_scalar, fold_across_rows

    kept = [leaf for leaf in record.root.leaves()
            if not leaf.is_root
            and any(node.gen.call_id in record.rewards
                    for node in leaf.path() if node.is_generated)]

    builders: list[SampleBuilder] = []
    for leaf in kept:
        builder = SampleBuilder()
        for node in leaf.path():
            if not node.is_generated:
                builder.add_context(node.span)
                continue
            trained = not node.response_trained
            node.response_trained = True
            gen = node.gen
            zeros = [0.0] * len(gen.output_ids)
            builder.add_response(gen.output_ids, gen.logprobs,
                                 record.rewards.get(gen.call_id, zeros), trained=trained)
        if builder.has_trained_response():
            builders.append(builder)

    episode_total = episode_scalar(record)
    folded = (fold or fold_across_rows)([b.response_scores for b in builders], algorithm)

    return [_to_sample(b, v, source, record, episode_total, algorithm, frames, status,
                       metadata_extra, ordinal)
            for ordinal, (b, v) in enumerate(zip(builders, folded, strict=True))]


def _to_sample(builder: SampleBuilder, scores: list[float], source: Any, record: Record,
               episode_total: float, algorithm: str, frames: Any, status: Any,
               metadata_extra: dict | None, ordinal: int):
    from slime.utils.types import Sample

    start = builder.leading_prompt_len or 0
    if len(builder.loss_mask) != len(builder.tokens) - start:
        raise TreeInvariant(
            f"row {ordinal}: {len(builder.tokens) - start} response tokens but "
            f"{len(builder.loss_mask)} mask entries")

    index = getattr(source, "index", None)
    sample = Sample(
        group_index=getattr(source, "group_index", None),
        index=index,
        # Shared by every row of this episode. Without it slime's loss reducer weights an
        # episode by how many rows it happened to produce, and its by-rollout step
        # splitter may put an episode's rows in different training steps.
        rollout_id=index,
        prompt=getattr(source, "prompt", ""),
        label=getattr(source, "label", None),
        tokens=list(builder.tokens),
        response_length=len(builder.tokens) - start,
        loss_mask=list(builder.loss_mask),
        rollout_log_probs=list(builder.logprobs),
        reward=episode_total,
        metadata={"round_number": ordinal,
                  "episode_turns": record.turns,
                  "episode_reward": episode_total,
                  "terminated": record.terminated,
                  "truncated": record.truncated,
                  **(metadata_extra or {})},
    )
    if status is not None:
        sample.status = status
    if frames is not None and builder.frames:
        sample.multimodal_inputs = {"images": frames.images(builder.frames)}
        sample.multimodal_train_inputs = frames.merge_train_inputs(builder.mm_train)
    # The per-token vector only travels when an estimator reads it; `algorithms.post_process`
    # swaps it in for `rewards` on the rollout side. Attaching it unconditionally would
    # make every run look like a per-token run to anything that checks.
    if algorithm == "token_level_gae":
        sample.metadata["per_token_reward"] = list(scores)
    # Not slime's, but every reader of a Sample in this repo may look.
    #
    # Both vectors, because they answer different questions and the fold is destructive.
    # `vagen_scores` is what the estimator will see; `vagen_raw_scores` is what the
    # environment actually paid, turn by turn. A transcript that showed the folded values
    # would report the same number on every row under default_gae and a different set of
    # numbers under token_level_gae -- for one episode in which the environment did
    # exactly the same thing.
    sample.vagen_scores = list(scores)
    sample.vagen_raw_scores = list(builder.response_scores)
    return sample


__all__ = ["Generated", "MessageNode", "Record", "SampleBuilder", "Span", "TreeInvariant",
           "assemble", "dropped_sample", "episode_status"]


# ------------------------------------------------------------------ episode status
def episode_status(*, aborted: bool, truncated: bool):
    """The status slime should see for every row of this episode.

    ``ABORTED`` is the load-bearing one, and only under fully-async: slime's worker checks
    ``any(s.status == ABORTED)`` and re-queues the whole group rather than shipping it to
    training. An abort is what a *weight update* looks like from inside a generation -- the
    rollout manager posts ``abort_request`` and waits for the server to drain -- so
    mislabelling one as COMPLETED trains on an episode that was cut off mid-way and
    reports it as a finished one.
    """
    from slime.utils.types import Sample

    if aborted:
        return Sample.Status.ABORTED
    return Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED


def dropped_sample(source: Any, reason: str):
    """A placeholder standing in for an episode that produced nothing trainable.

    Returning an empty list instead is the obvious move and it is wrong. slime's
    ``_convert_samples_to_train_data`` indexes ``samples[0]`` in several places and
    ``build_dp_schedule`` asserts each step has at least ``dp_size`` samples, so an episode
    that vanishes shrinks the batch and can take the training step with it -- which turns
    "one environment sample was unlucky" into "the step died".

    slime already has the right mechanism: ``remove_sample`` zeroes the loss mask, so the
    row occupies its slot and contributes no gradient.
    """
    import logging as _logging

    from slime.utils.types import Sample

    _logging.getLogger(__name__).warning(
        "dropping episode (index=%s): %s", getattr(source, "index", None), reason)
    sample = Sample(
        group_index=getattr(source, "group_index", None),
        index=getattr(source, "index", None),
        rollout_id=getattr(source, "index", None),
        prompt=getattr(source, "prompt", ""),
        tokens=list(getattr(source, "tokens", None) or [0]),
        response_length=1,
        loss_mask=[0],
        rollout_log_probs=None,
        reward=0.0,
        metadata={"vagen_dropped": reason},
    )
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    return sample
