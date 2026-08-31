"""Episode-level metrics, logged through slime's rollout-logging hook.

slime aggregates what reaches ``train_data``, and ``train_data`` is tokens, masks and one
reward. Everything an *environment* knows -- did the agent solve it, how many turns did it
take, was the response well-formed -- lives in ``sample.metadata`` and would otherwise
never be plotted. This module puts it on the same axes as the reward.

**Every episode counts once.** Under ``no_concat`` an episode becomes one Sample per
turn, all carrying the same metadata, so averaging over samples weights an episode by how
many turns it happened to take. A policy that learns to stall would then *raise* the
apparent success rate without solving anything more. Deduplication by ``rollout_id`` is the
whole reason this is not three lines.

Wired with::

    --custom-rollout-log-function-path      vagen_agent.metrics.log_rollout_data
    --custom-eval-rollout-log-function-path vagen_agent.metrics.log_eval_rollout_data

Both hooks are "return True to replace the default logging", so each function computes
slime's own metrics first and adds to them -- returning True without doing that would
silently delete `rollout/rewards`, the perf block, and everything else.
"""

from __future__ import annotations

import logging

from slime.utils.misc import group_by

logger = logging.getLogger(__name__)

#: Framework-owned episode facts. Environment-owned facts live in metadata["metrics"].
_EPISODE_KEYS = ("episode_turns", "episode_reward")
#: Episode facts that are booleans and read better as rates.
_RATE_KEYS = ("terminated", "truncated")


def episode_metrics(samples) -> dict[str, float]:
    """Environment-level metrics, one vote per episode.

    Also reports ``samples_per_episode`` -- how many ``Sample`` rows one episode became,
    which is the layout made visible: 1 under ``concat``, the turn count under
    ``no_concat``, the compaction count under ``compact``. Worth a curve of its own: it
    is the quantity that silently reweights everything else, because a training step is
    a fixed number of *episodes* and a long one therefore contributes more rows to it.
    Under ``compact`` it also moves during training as the policy's verbosity changes.
    """
    def episode_of(sample):
        rid = getattr(sample, "rollout_id", None)
        return rid if rid is not None else getattr(sample, "index", id(sample))

    grouped = group_by([s for s in samples if getattr(s, "metadata", None)], episode_of)
    if not grouped:
        return {}
    by_episode = {k: v[0].metadata for k, v in grouped.items()}
    samples_per = {k: len(v) for k, v in grouped.items()}

    out: dict[str, float] = {}
    episodes = list(by_episode.values())
    n = len(episodes)

    for key in _EPISODE_KEYS:
        values = [float(m[key]) for m in episodes if isinstance(m.get(key), (int, float, bool))]
        if values:
            out[key] = sum(values) / len(values)
    for key in _RATE_KEYS:
        values = [bool(m[key]) for m in episodes if key in m]
        if values:
            out[f"{key}_rate"] = sum(values) / len(values)

    out.update(_mean_environment_metrics(episodes))

    out["episodes"] = float(n)
    out["samples_per_episode"] = sum(samples_per.values()) / n

    # Per-environment metrics, so a mixed batch stays separable. Only when there is more
    # than one source -- a single-environment run does not need every metric twice.
    #
    # `<metric>-<source>`, not `<metric>/<source>`: this is prefixed again on the way out
    # (`env/`, `eval/<dataset>-`), and a key three segments deep loses its step metric in
    # wandb's UI and gets drawn against the global log counter instead.
    sources = {m.get("source_name") for m in episodes if m.get("source_name")}
    if len(sources) > 1:
        for source in sorted(sources):
            source_metrics = _mean_environment_metrics(
                [m for m in episodes if m.get("source_name") == source]
            )
            for key, value in source_metrics.items():
                out[f"{key}-{source}"] = value
    return out


def _mean_environment_metrics(episodes: list[dict]) -> dict[str, float]:
    """Average each environment-published final metric over reporting episodes."""
    values_by_key: dict[str, list[float]] = {}
    for meta in episodes:
        raw_metrics = meta.get("metrics") or {}
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        for key, value in metrics.items():
            if isinstance(key, str) and isinstance(value, (int, float, bool)):
                values_by_key.setdefault(key, []).append(float(value))

        # Read old rows and checkpoints without keeping the old name in new output.
        if "success" not in metrics and isinstance(
            meta.get("traj_success"), (int, float, bool)
        ):
            values_by_key.setdefault("success", []).append(float(meta["traj_success"]))

    return {
        key: sum(values) / len(values)
        for key, values in values_by_key.items()
        if values
    }


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Train-rollout logging: slime's metrics plus the environment's."""
    from slime.ray.rollout import (
        compute_metrics_from_samples,
        compute_perf_metrics_from_samples,
    )
    from slime.utils import logging_utils

    # compute_rollout_step / dict_add_prefix live in metric_utils, not misc -- rollout
    # re-exports neither. Getting this wrong is an ImportError raised inside the Ray actor
    # on the first rollout, i.e. after the engines are up and the model is loaded.
    from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix

    flat = _flatten(samples)
    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, flat), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, flat, rollout_time), "perf/")
    log_dict |= dict_add_prefix(episode_metrics(flat), "env/")

    logger.info("perf %s: %s", rollout_id, log_dict)
    log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    return True


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    """Eval logging: mean reward per dataset, plus the environment's metrics.

    ``data`` is ``{dataset_name: {"samples": [...], "rewards": [...]}}``. Mirrors slime's
    own eval logging rather than calling it, because that one is a private function that
    also consumes the dict.
    """
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step

    log_dict = {**(extra_metrics or {})}
    for name, info in (data or {}).items():
        rewards = info.get("rewards") or []
        if rewards:
            log_dict[f"eval/{name}"] = sum(rewards) / len(rewards)
        for key, value in episode_metrics(_flatten(info.get("samples") or [])).items():
            # `eval/<dataset>-<key>`, not `eval/<dataset>/<key>`: wandb's UI resolves a
            # metric's declared step_metric only two segments deep, so a three-segment key
            # is drawn against the global `_step` however the definition reads. Confirmed
            # on `rollout/response_len/max` as well. slime already writes its own
            # `eval/<key>-truncated_ratio` this way.
            log_dict[f"eval/{name}-{key}"] = value

    logger.info("eval %s: %s", rollout_id, log_dict)
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")

    # A few whole episodes, as HTML, beside the numbers. An eval curve says the policy got
    # worse; it does not say that it started emitting two actions a turn or that the frame
    # stopped arriving. This runs in the rollout actor, which slime gives a wandb run of
    # its own (`init_tracking(..., primary=False)`), so it can publish directly.
    from vagen_agent.utils.wandb_util import log_eval_episodes

    log_eval_episodes(args, data, step)
    return True


def _flatten(node) -> list:
    """slime hands samples as a Sample, a list, or a list of lists -- and under a
    row-splitting harness a third level as well."""
    if node is None:
        return []
    if isinstance(node, (list, tuple)):
        out = []
        for item in node:
            out.extend(_flatten(item))
        return out
    return [node]


__all__ = ["episode_metrics", "log_eval_rollout_data", "log_rollout_data"]
