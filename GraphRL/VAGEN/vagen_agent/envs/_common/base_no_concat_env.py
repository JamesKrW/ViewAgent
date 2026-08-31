"""Environment capability required by ``NoConcatHarness``."""

from vagen_agent.envs._common.base_vagen_env import BaseVagenEnv


class BaseNoConcatEnv(BaseVagenEnv):
    """Declare compatibility with independent per-turn model contexts."""


__all__ = ["BaseNoConcatEnv"]
