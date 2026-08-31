from __future__ import annotations

import asyncio
import inspect

import pytest
from PIL import Image

from vagen_agent.envs import (
    BaseCompactEnv,
    BaseConcatEnv,
    BaseEnv,
    BaseNoConcatEnv,
    BaseVagenEnv,
    build_env,
)
from vagen_agent.envs._common.utils import observation_to_message


class _NoConcatOnly(BaseNoConcatEnv):
    async def _reset(self, seed=None):
        return {"obs_str": f"state {seed}"}, {"seed": seed}

    async def _system_prompt(self):
        return {"obs_str": "act"}

    async def _step(self, action):
        return {"obs_str": str(action)}, 0.0, False, False, {}


class _AllHarnesses(BaseNoConcatEnv, BaseConcatEnv, BaseCompactEnv):
    async def _reset(self, seed=None):
        return {"obs_str": f"state {seed}"}, {"seed": seed}

    async def _system_prompt(self):
        return {"obs_str": "act"}

    async def _step(self, action):
        return (
            {"obs_str": str(action)},
            1.0,
            False,
            False,
            {"metrics": {"success": 1, "progress": 0.5}},
        )


def test_base_env_is_only_the_abstract_harness_contract() -> None:
    assert inspect.isabstract(BaseEnv)
    assert BaseEnv.__abstractmethods__ == {
        "close",
        "configure",
        "reset",
        "step",
        "system_prompt",
    }
    assert BaseVagenEnv.__abstractmethods__ == {
        "_reset",
        "_step",
        "_system_prompt",
    }


def test_harness_environment_bases_are_independent_capabilities() -> None:
    assert issubclass(BaseNoConcatEnv, BaseVagenEnv)
    assert issubclass(BaseConcatEnv, BaseVagenEnv)
    assert issubclass(BaseCompactEnv, BaseVagenEnv)
    assert not issubclass(BaseCompactEnv, BaseConcatEnv)
    assert not issubclass(BaseConcatEnv, BaseNoConcatEnv)
    assert isinstance(_AllHarnesses({}), BaseNoConcatEnv)
    assert isinstance(_AllHarnesses({}), BaseConcatEnv)
    assert isinstance(_AllHarnesses({}), BaseCompactEnv)


def test_build_env_rejects_an_unsupported_harness_contract() -> None:
    with pytest.raises(TypeError, match="does not support BaseConcatEnv"):
        build_env(_NoConcatOnly, {}, required_type=BaseConcatEnv)


def test_base_env_enforces_turn_limit_and_tracks_metrics() -> None:
    async def scenario() -> None:
        env = build_env(_AllHarnesses, {}, max_turns=1, required_type=BaseCompactEnv)
        observation, info = await env.reset(seed=7)
        assert observation == {"obs_str": "state 7"}
        assert info == {"seed": 7}
        _, reward, terminated, truncated, step_info = await env.step("left")
        assert reward == 1.0
        assert terminated is False
        assert truncated is True
        assert step_info["truncated"] is True
        assert env.last_metrics == {"success": 1.0, "progress": 0.5}
        assert env.success is True

    asyncio.run(scenario())


def test_observation_to_message_preserves_structured_state() -> None:
    message = observation_to_message(
        {
            "obs_str": "before <image> after",
            "multi_modal_input": {"<image>": [Image.new("RGB", (2, 2))]},
            "screen_size": [1280, 720],
        }
    )
    assert message["role"] == "user"
    assert [part["type"] for part in message["content"]] == [
        "text",
        "image",
        "text",
    ]
    assert len(message["images"]) == 1
    assert message["screen_size"] == [1280, 720]
