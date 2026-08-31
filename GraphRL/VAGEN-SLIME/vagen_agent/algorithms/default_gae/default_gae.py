"""Default episode-level GAE mapped onto Slime's row-local PPO estimator."""

from vagen_agent.algorithms._common import AlgorithmSpec, register_algorithm

SPEC = register_algorithm(
    "default_gae",
    AlgorithmSpec(
        slime_estimator="ppo",
        needs_critic=True,
        spans_rows=True,
        requires_undiscounted=True,
        reward_folding="episode_total",
        note="episode-total reward on every row; Slime's per-row GAE at gamma=lam=1 "
        "reproduces VAGEN's episode-global default_gae exactly",
    ),
)

__all__ = ["SPEC"]
