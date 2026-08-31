"""Environment capability required by ``CompactHarness``."""

from vagen_agent.envs._common.base_vagen_env import BaseVagenEnv


class BaseCompactEnv(BaseVagenEnv):
    """Declare compatibility with context compaction during an episode."""


__all__ = ["BaseCompactEnv"]
