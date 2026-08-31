"""Reusable client/server framework for process- or host-isolated environments."""

from vagen_agent.envs.remote.handler import (
    BaseGymHandler,
    HandlerResult,
    SessionNotFoundError,
)
from vagen_agent.envs.remote.remote_env import RemoteEnv
from vagen_agent.envs.remote.observation_codec import (
    pack_observation,
    unpack_observation,
)


def __getattr__(name: str):
    # Keep rollout-only installs free from a mandatory FastAPI import.  The client
    # needs only httpx and Pillow; server processes ask for GymService explicitly.
    if name == "GymService":
        from vagen_agent.envs.remote.service import GymService

        return GymService
    raise AttributeError(name)


__all__ = [
    "BaseGymHandler",
    "GymService",
    "HandlerResult",
    "SessionNotFoundError",
    "RemoteEnv",
    "pack_observation",
    "unpack_observation",
]
