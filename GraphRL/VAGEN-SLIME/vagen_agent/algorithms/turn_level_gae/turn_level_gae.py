"""Turn-level GAE declaration pending value-mask support."""

from vagen_agent.algorithms._common import AlgorithmSpec, register_algorithm

SPEC = register_algorithm(
    "turn_level_gae",
    AlgorithmSpec(
        slime_estimator="ppo",
        custom_advantage_path="vagen_agent.advantage.turn_level_gae.compute",
        needs_critic=True,
        spans_rows=True,
        requires_undiscounted=True,
        needs_value_mask=True,
        reward_folding="identity",
        note="NOT YET PORTED -- also needs a value_mask channel Slime's critic loss lacks",
    ),
)

__all__ = ["SPEC"]
