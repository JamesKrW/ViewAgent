"""Run configuration: which point on each axis this run occupies.

Two sources, deliberately split by *what varies*:

**Per run** -- the harness, the model family, and the algorithm. These come from the yaml
that slime's ``--custom-config-path`` points at, because one run has one of each.

**Per row** -- which environment, its config, its seed, its turn limit. These ride in
``sample.metadata``, put there by ``make_dataset.py``, because a batch may legitimately mix
environments and each needs its own settings. slime's ``Dataset`` already carries an
arbitrary ``metadata`` dict per row and passes it through to the training side as
``source_names``, so nothing had to be added to slime for this.

That split is what makes "add an environment" a dataset change rather than a code change.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from vagen_agent.algorithms import AlgorithmSpec, resolve_algorithm

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------- run config
@dataclass
class RunConfig:
    """One point on each axis, plus the knobs a harness needs.

    Everything has a default that is either safe or absent; nothing is silently invented.
    ``harness`` in particular has no default -- a run that does not say which context
    policy it uses is a run whose results cannot be compared with anything.
    """

    harness: str
    algorithm: str = "default_gae"
    #: ``"auto"`` picks the adapter from the tokenizer/processor class name. Explicit is
    #: better when it matters; auto exists so a config does not have to name a family it
    #: does not care about.
    model_adapter: str = "auto"

    # -- harness knobs, only read by the ones that want them --------------------
    compact_budget: int | None = None
    compact_summary_budget: int | None = None
    #: Constructor arguments shared with standalone evaluation's ``harness_config``.
    harness_config: dict[str, Any] = field(default_factory=dict)

    # -- defaults for env-spec fields a dataset row may omit --------------------
    default_max_turns: int = 5
    default_response_length_per_turn: int | None = None
    default_max_env_response_per_turn: int | None = None

    #: Where the environment registry lives. Absent means VAGEN's own.
    env_registry: str | None = None

    # -- conversation transcripts (vagen/transcript) ----------------------------
    #: Root for one JSONL transcript per episode. Absent switches recording off
    #: entirely, which is the default: a rollout should not start writing files
    #: because someone imported the module.
    transcript_dir: str | None = None
    #: Also keep the frames, content-addressed under ``<transcript_dir>/frames``.
    #: Off by default -- a run producing millions of pictures should say so before it
    #: starts storing them. Off still records that a frame was there and how large it
    #: was, so the conversation reads correctly either way.
    transcript_images: bool = False
    #: Subdirectory under ``transcript_dir``. Episode ids are unique within a process,
    #: not across a run, so set this per training step (or per run) when you want a
    #: listing that is not the whole history at once.
    transcript_group: str = ""

    #: Anything else the yaml carried, kept rather than dropped so a custom extension can
    #: read its own settings without this dataclass having to know about them.
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN: ClassVar[set[str]] = {
        "harness",
        "algorithm",
        "model_adapter",
        "compact_budget",
        "compact_summary_budget",
        "default_max_turns",
        "default_response_length_per_turn",
        "default_max_env_response_per_turn",
        "env_registry",
        "harness_config",
        "transcript_dir",
        "transcript_images",
        "transcript_group",
    }

    @classmethod
    def from_args(cls, args) -> RunConfig:
        """Build from slime's args.

        slime merges ``--custom-config-path``'s yaml onto the argument namespace
        (``arguments.py``: ``if args.custom_config_path: ... setattr``), so the keys arrive
        as plain attributes. Read from there rather than re-parsing the yaml, so a value
        overridden on the command line wins -- which is how every other slime knob behaves.
        """
        if not getattr(args, "harness", None):
            raise ValueError(
                "no `harness` in the custom config. A run has to say which context policy "
                f"it uses -- one of {sorted(_harness_names())} -- because it decides what "
                "the model sees and how many training rows an episode becomes. There is no "
                "safe default: picking one silently makes two runs incomparable while both "
                "look fine."
            )
        known = {name: getattr(args, name) for name in cls._KNOWN if hasattr(args, name)}
        if "harness_config" in known:
            raw_harness_config = known["harness_config"]
            if raw_harness_config is not None and not isinstance(raw_harness_config, Mapping):
                raise TypeError("harness_config must be a mapping")
            known["harness_config"] = dict(raw_harness_config or {})
        extra = dict(getattr(args, "vagen_extra", {}) or {})
        config = cls(**known, extra=extra)
        config._check_against(args)
        return config

    def _check_against(self, args) -> None:
        """The two ways the yaml and slime's flags can disagree without failing.

        Both produce a finished run with plausible curves that trained a different
        algorithm than the one configured, which is why they are worth a hard stop.
        """
        spec = self.spec
        estimator = getattr(args, "advantage_estimator", None)
        if estimator and estimator != spec.slime_estimator:
            raise ValueError(
                f"algorithm: {self.algorithm} needs --advantage-estimator "
                f"{spec.slime_estimator}, but the run passes {estimator!r}. slime also "
                f"decides whether to build a critic from that literal string, so the two "
                f"have to agree."
            )
        if spec.requires_undiscounted:
            gamma, lambd = getattr(args, "gamma", 1.0), getattr(args, "lambd", 1.0)
            if (gamma, lambd) != (1.0, 1.0):
                raise ValueError(
                    f"algorithm: {self.algorithm} is only defined at gamma = lambd = 1, "
                    f"and this run has gamma={gamma} lambd={lambd}. The port reproduces "
                    f"VAGEN's episode-global estimator by giving every row of an episode "
                    f"the episode total and letting slime's per-row GAE run; that identity "
                    f"is a telescoping property of gamma = lambd = 1 and holds nowhere "
                    f"else. Discounted, each row would be credited as if it began the "
                    f"episode."
                )

    # ------------------------------------------------------------------ derived
    @property
    def spec(self) -> AlgorithmSpec:
        return resolve_algorithm(self.algorithm)

    def harness_kwargs(self) -> dict:
        """The subset of the config a harness constructor wants.

        Passed by name rather than by handing the whole run config over, so a harness
        cannot quietly start depending on the optimizer or advantage estimator.
        """
        kwargs = dict(self.harness_config)
        legacy = {
            "budget": (self.compact_budget, "compact_budget"),
            "summary_budget": (
                self.compact_summary_budget,
                "compact_summary_budget",
            ),
        }
        for key, (value, source) in legacy.items():
            if value is None:
                continue
            if key in kwargs and kwargs[key] != value:
                raise ValueError(
                    f"harness_config.{key} conflicts with legacy {source}"
                )
            kwargs[key] = value
        return kwargs

    def describe(self) -> str:
        s = self.spec
        return (
            f"harness={self.harness}  model={self.model_adapter}\n"
            f"algorithm={self.algorithm} -> --advantage-estimator {s.slime_estimator}"
            + (f" + {s.custom_advantage_path}" if s.custom_advantage_path else " (slime native)")
            + (f"\n  {s.note}" if s.note else "")
        )


# --------------------------------------------------------------------- env spec
@dataclass
class EnvSpec:
    """One dataset row's environment: what to build, and the budgets it runs under.

    Read from ``sample.metadata``. Defaults come from :class:`RunConfig` rather than from
    constants here, so a run can move them without editing the dataset.
    """

    env_name: str
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    max_turns: int = 5
    response_length_per_turn: int | None = None
    max_env_response_per_turn: int | None = None
    source_name: str = "unknown"

    @classmethod
    def from_metadata(cls, metadata: dict | None, run: RunConfig) -> EnvSpec:
        metadata = metadata or {}
        if not metadata.get("env_name"):
            raise ValueError(
                "the dataset row carries no `env_name` in its metadata, so there is nothing "
                f"to build. Available keys: {sorted(metadata)}. Rows are produced by "
                "vagen_agent/make_dataset.py -- if this row came from somewhere else, it "
                "needs at least {'env_name': ..., 'config': {...}, 'seed': ...}."
            )
        # No silent default for max_turns. Falling back to one turn looks like a working
        # run whose episodes all stop after a single step -- nearly invisible, since every
        # row is well-formed and merely short.
        max_turns = int(metadata.get("max_turns") or run.default_max_turns)
        if max_turns < 1:
            raise ValueError(f"max_turns={max_turns} for env {metadata['env_name']!r}")
        return cls(
            env_name=str(metadata["env_name"]),
            config=dict(metadata.get("config") or {}),
            seed=metadata.get("seed"),
            max_turns=max_turns,
            response_length_per_turn=(metadata.get("response_length_per_turn")
                                      or run.default_response_length_per_turn),
            max_env_response_per_turn=(metadata.get("max_env_response_per_turn")
                                       or run.default_max_env_response_per_turn),
            source_name=str(metadata.get("source_name") or metadata["env_name"]),
        )


def _harness_names():
    from vagen_agent.harness import HARNESSES

    return HARNESSES.keys()


__all__ = ["EnvSpec", "RunConfig"]
