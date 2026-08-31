"""``--custom-generate-function-path``: one episode, whichever environment and harness.

This is the whole of the verl replacement. A harness runs the episode as ordinary agent
code against an OpenAI-shaped client; the rollout client records what the model was actually
shown and generated; ``assemble`` turns that record into slime ``Sample``s. This module
only wires the axes together and owns the two things that must not be left to a harness:
the environment's lifetime, and the episode's identity.

It contains no environment-specific and no harness-specific code, which is the property
worth protecting: adding an environment is a registry entry plus a dataset column, and
switching context policy is one word in a yaml.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

from vagen_agent.config import EnvSpec, RunConfig
from vagen_agent.envs import build_env, get_env_cls
from vagen_agent.harness import build_harness
from vagen_agent.models import build_model_adapter
from vagen_agent.rollout.client import RolloutClient
from vagen_agent.rollout.frames import FrameTable
from vagen_agent.rollout.scoring import ScoringSeam
from vagen_agent.rollout.trajectory import (
    Record,
    assemble,
    dropped_sample,
    episode_status,
)

logger = logging.getLogger(__name__)


@dataclass
class _Context:
    """Everything that is the same for every episode in this process.

    Built once: loading a tokenizer and a processor per episode would dominate the
    rollout. The frame table is here for the same reason and one more -- it is
    content-addressed, so a board that recurs across every episode this worker runs is
    encoded and processed once.
    """

    run: RunConfig
    model: Any
    frames: FrameTable


_context: _Context | None = None
_context_lock = threading.Lock()
_episode_counter = itertools.count()


def _get_context(args) -> _Context:
    global _context
    with _context_lock:
        if _context is not None:
            return _context

        from slime.rollout.sglang_rollout import GenerateState

        state = GenerateState(args)
        run = RunConfig.from_args(args)

        name = run.model_adapter
        if name == "auto":
            name = _detect_family(state.tokenizer, state.processor)
            logger.info("model_adapter=auto resolved to %r", name)
        model = build_model_adapter(
            name, state.tokenizer, state.processor,
            apply_chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None) or {},
            mm_processor_kwargs=getattr(args, "mm_processor_kwargs", None) or {})

        # Proved once per process, at startup, where a failure can still stop the run.
        # A wrong adapter does not fail: it renders every turn boundary slightly off, and
        # both the rollout and the training pass see the same seam, so nothing downstream
        # can report it.
        model.self_test()
        _check_axes_agree(run, model)

        _context = _Context(run=run, model=model, frames=FrameTable())
        return _context


def _detect_family(tokenizer, processor) -> str:
    """Guess the adapter from the tokenizer/processor class name.

    Deliberately narrow, and it *raises* rather than falling back to a default -- see
    ``model.self_test`` for why guessing wrong is invisible.
    """
    names = " ".join(type(x).__name__.lower() for x in (processor, tokenizer) if x is not None)
    if "qwen" in names:
        return "qwen"
    raise ValueError(
        f"could not tell which model family {names!r} belongs to. Set `model_adapter:` in "
        f"the custom config explicitly -- guessing wrong would not fail, it would render "
        f"every turn boundary slightly off and nothing would report it.")


def _check_axes_agree(run: RunConfig, model) -> None:
    """Cross-axis claims checked where a bad combination can still stop the run.

    Both were declared and never read. A compact run on a family that cannot render an
    isolated user turn used to fail at the first compaction -- a long way into a rollout,
    on some episodes and not others.
    """
    from vagen_agent.harness import resolve_harness
    from vagen_agent.harness.compact import CompactHarness

    harness_cls = resolve_harness(run.harness)
    if not harness_cls.supports_training:
        raise ValueError(
            f"harness={run.harness} requires model roles {list(harness_cls.model_roles)}, "
            "but Slime rollouts currently construct one trainable client and token tree. "
            "Use standalone evaluation for this harness; multi-policy RL needs an "
            "explicit role-to-optimizer and reward-attribution design."
        )
    if issubclass(harness_cls, CompactHarness) \
            and not model.can_render_isolated_user_turn():
        raise ValueError(
            f"harness={run.harness} asks the model to summarise with a single user turn, "
            f"which {type(model).__name__} says this family's chat template cannot "
            f"render. Pick concat or no_concat, or a different model.")


def _build_episode(args, ctx: _Context, sample):
    spec = EnvSpec.from_metadata(getattr(sample, "metadata", None), ctx.run)
    record = Record(episode_id=f"{getattr(sample, 'index', None)}-{os.getpid()}-"
                              f"{next(_episode_counter)}")
    client = RolloutClient(
        ctx.model,
        record,
        args=args,
        sampling_params={"temperature": getattr(args, "rollout_temperature", 1.0)},
        frames=ctx.frames,
        response_decoder=_response_decoder(spec, ctx.model),
    )
    window = (getattr(args, "rollout_max_context_len", None)
              or getattr(args, "rollout_max_response_len", None))
    response_limit = _response_limit(window, spec, args)
    harness = build_harness(run_harness_name(ctx.run), window=window,
                            response_limit=response_limit, **ctx.run.harness_kwargs())
    base_env = build_env(
        get_env_cls(spec.env_name),
        spec.config,
        max_turns=spec.max_turns,
        required_type=harness.environment_type,
    )
    env = ScoringSeam(base_env, record, seed=spec.seed)
    return spec, record, client, env, harness


def run_harness_name(run: RunConfig) -> str:
    return run.harness


def _response_limit(window, spec: EnvSpec, args) -> int | None:
    """The hard upper bound for one environment action.

    A dataset may tighten slime's global response cap for one environment, but may not
    enlarge it.  The context window is included as a final defensive ceiling; the
    per-call remaining room is applied later by ``BaseHarness.room``.
    """
    candidates = (spec.response_length_per_turn,
                  getattr(args, "rollout_max_response_len", None), window)
    limits = [int(value) for value in candidates if value is not None]
    if not limits:
        return None
    if any(value < 1 for value in limits):
        raise ValueError(f"response limits must be positive, got {limits}")
    return min(limits)


def _response_decoder(spec: EnvSpec, model):
    """Build the model-side text decoder required by this action dialect.

    Decoding belongs to the model adapter because special-token roles and ids belong to
    the checkpoint's tokenizer.  The environment only declares which task vocabulary
    must survive that decoding.
    """
    dialect = str(spec.config.get("dialect", "")).lower()
    if dialect not in {"hid", "hid_v1"}:
        return None
    from hidagent.datasets._common.hid_vocab import ALL_SPECIAL_TOKENS

    vocab = model.tokenizer.get_vocab()
    missing = [token for token in ALL_SPECIAL_TOKENS if token not in vocab]
    if missing:
        preview = ", ".join(repr(token) for token in missing[:5])
        raise ValueError(
            f"the selected tokenizer is missing {len(missing)} HID action token(s), "
            f"including {preview}. Evaluate the tokenizer saved with the HID-trained "
            "checkpoint; adding tokens only at rollout would assign untrained ids."
        )

    def decode_generated(token_ids: list[int], _server_text: str) -> str:
        return model.decode_generated(
            token_ids,
            preserve_special_tokens=ALL_SPECIAL_TOKENS,
        )

    return decode_generated


async def generate(args, sample, sampling_params) -> list:
    """One episode -> the training rows it produced.

    Returns a list because an episode is one row only under ``concat``; ``no_concat``
    gives each turn its own and ``compact`` starts a new one at every compaction. slime
    supports that shape directly, provided the siblings share a ``rollout_id``.
    """
    ctx = _get_context(args)
    spec, record, client, env, harness = _build_episode(args, ctx, sample)
    client.sampling_params.update(sampling_params or {})

    try:
        await harness.run_episode(client, env)
    finally:
        # Every guard on this path raises mid-episode, and an environment left open is
        # held for the rest of the batch. A rollout that dies on the first episode used to
        # take its simulator down with it.
        await env.close()

    if ctx.run.transcript_dir:
        from vagen_agent.rollout.dump import dump_episode

        dump_episode(record, ctx.run, ctx.frames, spec)

    metrics = dict(getattr(env, "last_metrics", {}) or {})
    metrics.setdefault("success", float(getattr(env, "success", False)))
    samples = assemble(
        record, sample,
        algorithm=ctx.run.algorithm,
        frames=ctx.frames,
        status=episode_status(aborted=record.aborted, truncated=record.truncated),
        metadata_extra={"source_name": spec.source_name,
                        "env_name": spec.env_name,
                        "metrics": metrics})
    if not samples:
        return [dropped_sample(sample, "the model never spoke in any conversation")]
    return samples


__all__ = ["generate"]
