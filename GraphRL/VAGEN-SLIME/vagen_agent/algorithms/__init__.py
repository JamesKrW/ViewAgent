"""Algorithm registry and Slime-facing reward transformations."""

from vagen_agent.algorithms._common import (
    ALGORITHMS,
    AlgorithmSpec,
    register_algorithm,
    registered_algorithms,
    resolve_algorithm,
)
from vagen_agent.algorithms._common.rewards import (
    episode_reward_per_row,
    episode_reward_vectors,
    episode_scalar,
    fold_across_rows,
    post_process,
)
from vagen_agent.algorithms.default_gae import SPEC as DEFAULT_GAE
from vagen_agent.algorithms.token_level_gae import SPEC as TOKEN_LEVEL_GAE
from vagen_agent.algorithms.trajectory_grpo import SPEC as TRAJECTORY_GRPO
from vagen_agent.algorithms.turn_level_gae import SPEC as TURN_LEVEL_GAE

__all__ = [
    "ALGORITHMS",
    "DEFAULT_GAE",
    "TOKEN_LEVEL_GAE",
    "TRAJECTORY_GRPO",
    "TURN_LEVEL_GAE",
    "AlgorithmSpec",
    "episode_reward_per_row",
    "episode_reward_vectors",
    "episode_scalar",
    "fold_across_rows",
    "post_process",
    "register_algorithm",
    "registered_algorithms",
    "resolve_algorithm",
]
