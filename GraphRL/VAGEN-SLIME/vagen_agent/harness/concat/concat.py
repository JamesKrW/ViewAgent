"""One conversation for the whole episode: one training row."""

from __future__ import annotations

from vagen_agent.envs._common.base_concat_env import BaseConcatEnv
from vagen_agent.harness._common.base_harness import (
    BaseHarness,
    assistant,
    obs_to_message,
)


class ConcatHarness(BaseHarness):
    """Append everything. The message list only ever grows, so the tree is one chain and
    the episode is one row."""

    environment_type = BaseConcatEnv

    def __init__(self, window: int | None = None, response_limit: int | None = None):
        self.window, self.response_limit = window, response_limit

    async def run_episode(self, client, env):
        initial, _reset_info = await env.reset()
        messages = [
            obs_to_message(await env.system_prompt(), role="system"),
            obs_to_message(initial),
        ]
        while True:
            response = await client.create(
                messages, **self.room(client, messages, self.window, self.response_limit))
            messages.append(assistant(response.text))
            obs, _reward, terminated, truncated, _info = await env.step(response)
            if terminated or truncated:
                return
            messages.append(obs_to_message(obs))
