"""Register ViewSuite environments before delegating rollout generation to VAGEN-SLIME."""

from __future__ import annotations

from vagen_agent.envs import register_env
from vagen_agent.rollout import generate as _generate
from view_suite.envs.habitat_gs_proxy_task.slime_adapter import (
    HabitatGSInteractiveViewPlanning,
)


register_env("HabitatGSInteractiveViewPlanning", HabitatGSInteractiveViewPlanning)


async def generate(args, sample, sampling_params):
    return await _generate(args, sample, sampling_params)


__all__ = ["generate"]
