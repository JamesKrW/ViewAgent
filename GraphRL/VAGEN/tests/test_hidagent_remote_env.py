"""Contract checks for the HIDAgent RemoteEnv bridge.

Run directly with ``python tests/test_hidagent_remote_env.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from vagen_agent.harness.no_concat import NoConcatHarness
from vagen_agent.rollout import RolloutClient
from vagen_agent.rollout.trajectory import Record, Span


class FakeModelClient:
    def size(self, _messages):
        return 2

    async def create(self, messages, **_sampling):
        assert [message["role"] for message in messages] == ["system", "user"]
        return SimpleNamespace(text="<click>500500")


def test_harness_resets_before_requesting_session_prompt() -> None:
    events = []

    class OrderedEnv:
        async def reset(self):
            events.append("reset")
            return {"role": "user", "content": "observation"}, {}

        async def system_prompt(self):
            events.append("system_prompt")
            return {"role": "system", "content": "system"}

        async def step(self, _response):
            return {"role": "user", "content": "done"}, 1.0, True, False, {}

    asyncio.run(
        NoConcatHarness(window=32, response_limit=8).run_episode(
            FakeModelClient(), OrderedEnv()
        )
    )
    assert events == ["reset", "system_prompt"]


def test_adapter_reconstructs_response_text_from_generated_ids() -> None:
    class Renderer:
        def render(self, messages, **_kwargs):
            return [Span(ids=[10], engine_ids=[10])] * len(messages)

    adapter = RolloutClient(
        SimpleNamespace(),
        Record(),
        response_decoder=lambda ids, _text: f"decoded:{','.join(map(str, ids))}",
    )
    adapter.renderer = Renderer()

    async def post(_engine_ids, _images, _sampling):
        return {
            "text": "server text with special tokens stripped",
            "token_ids": [7, 8, 9],
            "logprobs": [-0.1, -0.2, -0.3],
            "stop_reason": "stop",
            "usage": {},
        }

    adapter._post = post
    response = asyncio.run(adapter.create([{"role": "user", "content": "go"}]))
    assert response.text == "decoded:7,8,9"
    generated = next(child for child in adapter.record.root.children[0].children if child.gen)
    assert generated.message["content"] == "decoded:7,8,9"


def main() -> None:
    test_harness_resets_before_requesting_session_prompt()
    test_adapter_reconstructs_response_text_from_generated_ids()
    print("PASS HIDAgent RemoteEnv bridge")


if __name__ == "__main__":
    main()
