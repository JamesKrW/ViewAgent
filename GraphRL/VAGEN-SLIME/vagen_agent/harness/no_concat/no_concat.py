"""A conversation per turn: one training row per turn."""

from __future__ import annotations

from vagen_agent.envs._common.base_no_concat_env import BaseNoConcatEnv
from vagen_agent.harness._common.base_harness import BaseHarness, obs_to_message


class NoConcatHarness(BaseHarness):
    """Rebuild the list every turn, so the model sees the system prompt and the latest
    observation and never the history.

    Nothing here opens or closes a conversation. A fresh list that shares only the system
    prompt diverges from the previous turn right after it, and the tree records that as a
    sibling branch -- so the row split falls out of what this code does to a list, rather
    than being declared anywhere.
    """

    environment_type = BaseNoConcatEnv

    def __init__(self, window: int | None = None, response_limit: int | None = None):
        self.window, self.response_limit = window, response_limit

    async def run_episode(self, client, env):
        initial, _reset_info = await env.reset()
        obs = obs_to_message(initial)
        system = obs_to_message(await env.system_prompt(), role="system")
        while True:
            messages = [system, obs]
            response = await client.create(
                messages, **self.room(client, messages, self.window, self.response_limit))
            step_obs, _reward, terminated, truncated, _info = await env.step(response)
            if terminated or truncated:
                return
            obs = obs_to_message(step_obs)
