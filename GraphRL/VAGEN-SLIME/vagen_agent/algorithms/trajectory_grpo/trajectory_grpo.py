"""Episode-grouped GRPO registration."""

from vagen_agent.algorithms._common import AlgorithmSpec, register_algorithm

SPEC = register_algorithm(
    "trajectory_grpo",
    AlgorithmSpec(
        slime_estimator="grpo",
        spans_rows=True,
        reward_folding="episode_total",
        note="one advantage per episode, normalised within the prompt group; the grouping "
        "is done in the reward post-process, not by Slime's flat reshape",
    ),
)

__all__ = ["SPEC"]
