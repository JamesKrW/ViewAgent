"""SLIME logging hooks with GraphRL metric aliases and rollout durability."""

from __future__ import annotations

import logging
import os
from typing import Any

from vagen_agent.metrics import episode_metrics

from graphrl.slime.rollout import finalize_rollout_step

logger = logging.getLogger(__name__)


def _flatten(node: Any) -> list[Any]:
    if node is None:
        return []
    if isinstance(node, (list, tuple)):
        output = []
        for item in node:
            output.extend(_flatten(item))
        return output
    return [node]


def _step(rollout_id: int) -> int:
    # The GraphRL convention is eval-before-train at step 0, followed by
    # one-based completed RL steps. The custom drivers call baseline eval with -1.
    return 0 if int(rollout_id) < 0 else int(rollout_id) + 1


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    from slime.ray.rollout import compute_metrics_from_samples, compute_perf_metrics_from_samples
    from slime.utils import logging_utils
    from slime.utils.metric_utils import dict_add_prefix

    flat = _flatten(samples)
    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, flat), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, flat, rollout_time), "perf/")
    log_dict |= dict_add_prefix(episode_metrics(flat), "env/")
    step = _step(rollout_id)
    log_dict["rollout/step"] = step
    logger.info("rollout %s: %s", step, log_dict)
    logging_utils.log(args, log_dict, step_key="rollout/step")

    # The marker is published after the JSONL and every selected image are on disk.
    finalize_rollout_step(args, int(rollout_id), samples)
    return True


def _adaptive_observe(metrics: dict[str, Any], step: int) -> None:
    experiment = os.environ.get("GRAPHRL_ADAPTIVE_EXPERIMENT_DIR")
    round_value = os.environ.get("GRAPHRL_ADAPTIVE_ROUND")
    if not experiment or round_value is None:
        return
    from graphrl.slime.schedule import SlimeAdaptiveSchedule

    SlimeAdaptiveSchedule(experiment, int(round_value)).observe(metrics, step)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    from slime.utils import logging_utils

    log_dict = {**(extra_metrics or {})}
    for name, info in (data or {}).items():
        rewards = info.get("rewards") or []
        if rewards:
            log_dict[f"eval/{name}"] = sum(rewards) / len(rewards)
        metrics = episode_metrics(_flatten(info.get("samples") or []))
        # Existing GraphRL schedules and dashboards use ``traj_success`` while
        # VAGEN-SLIME standardizes the environment contract on ``success``.
        # Publish both names without forcing every environment adapter to carry
        # a duplicate metric.
        if "success" in metrics and "traj_success" not in metrics:
            metrics["traj_success"] = metrics["success"]
        for key, value in metrics.items():
            log_dict[f"eval/{name}-{key}"] = value
            # Preserve the metric namespace used by the existing adaptive configs.
            log_dict[f"val-aux/{name}/{key}/mean@1"] = value

    step = _step(rollout_id)
    log_dict["rollout/step"] = step
    logger.info("eval %s: %s", step, log_dict)
    logging_utils.log(args, log_dict, step_key="rollout/step")
    _adaptive_observe(log_dict, step)

    from vagen_agent.utils.wandb_util import log_eval_episodes

    log_eval_episodes(args, data, step)
    return True


__all__ = ["log_eval_rollout_data", "log_rollout_data"]
