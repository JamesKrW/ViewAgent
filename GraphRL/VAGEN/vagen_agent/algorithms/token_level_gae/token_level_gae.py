"""Token-level episode GAE registration."""

from vagen_agent.algorithms._common import AlgorithmSpec, register_algorithm

SPEC = register_algorithm(
    "token_level_gae",
    AlgorithmSpec(
        slime_estimator="ppo",
        needs_critic=True,
        spans_rows=True,
        requires_undiscounted=True,
        per_token_reward=True,
        reward_folding="token_suffix",
        note="each turn's reward stays on the token that earned it; Slime's own GAE "
        "takes the per-token vector directly",
    ),
)

__all__ = ["SPEC"]
