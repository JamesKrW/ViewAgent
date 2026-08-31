"""Environment capability required by ``ConcatHarness``."""

from vagen_agent.envs._common.base_vagen_env import BaseVagenEnv


class BaseConcatEnv(BaseVagenEnv):
    """Declare compatibility with one growing episode conversation."""


__all__ = ["BaseConcatEnv"]
