from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

import pytest

from vagen_agent.config import RunConfig
from vagen_agent.envs import (
    BaseCompactEnv,
    BaseConcatEnv,
    BaseNoConcatEnv,
    EnvAction,
    register_env,
)
from vagen_agent.evaluation.backends import (
    BackendReply,
    OpenAIChatBackend,
    registered_backends,
    render_message,
)
from vagen_agent.evaluation.config import ModelSpec, load_config
from vagen_agent.evaluation.recording import EpisodeStore
from vagen_agent.evaluation.runner import EvaluationRunner
from vagen_agent.evaluation.viewer import Index
from vagen_agent.harness import (
    BaseHarness,
    client_for_role,
    obs_to_message,
    register_harness,
)


class _CounterEnv(BaseNoConcatEnv, BaseConcatEnv, BaseCompactEnv):
    async def _reset(self, seed=None):
        self.turn = 0
        self.formatted_turns = 0
        return {
            "obs_str": f"state {seed}",
            "screen_size": [1280, 720],
            "proprio": {"velocity": [0.0, 1.0]},
        }, {"seed": seed}

    async def _system_prompt(self):
        return {"obs_str": "Return <answer>Right</answer>."}

    async def _step(self, response):
        self.turn += 1
        done = self.turn >= int(self.env_config.get("finish_after", 3))
        self.formatted_turns += int("<answer>" in response.text)
        return (
            {"obs_str": f"state {self.turn}"},
            1.0 if done else 0.0,
            done,
            False,
            {
                "success": done,
                "format_correct": "<answer>" in response.text,
                "metrics": {
                    "success": float(done),
                    "progress": self.turn / int(self.env_config.get("finish_after", 3)),
                    "format_compliance": self.formatted_turns / self.turn,
                },
            },
        )


@register_harness("_test_two_role")
class _TwoRoleHarness(BaseHarness):
    """Exercise VAGEN's generic external multi-role Harness support."""

    model_roles = ("planner", "actor")
    action_model_role = "actor"
    supports_training = False

    def __init__(self, window=None, response_limit=None) -> None:
        del window, response_limit

    async def run_episode(self, client, env) -> None:
        observation, _reset_info = await env.reset()
        system = obs_to_message(await env.system_prompt(), role="system")
        planner = client_for_role(client, "planner")
        actor = client_for_role(client, "actor")
        plan = await planner.create([system, obs_to_message(observation)])
        actor_observation = obs_to_message(observation)
        actor_observation = dict(actor_observation)
        actor_observation["content"] = (
            f"{actor_observation.get('content', '')}\nSubgoal: {plan.text}"
        )
        action = await actor.create([system, actor_observation])
        await env.step(action)


class _FakeBackend:
    calls = 0

    def __init__(self, spec):
        self.spec = spec

    async def complete(
        self, messages, sampling, *, max_new_tokens=None, thinking_token_budget=None
    ):
        type(self).calls += 1
        summary = "Summarise the conversation" in str(messages[-1].get("content"))
        return BackendReply(
            content="summary" if summary else "<answer>Right</answer>",
            reasoning_content="short reasoning",
            finish_reason="stop",
            prompt_tokens=100 * len(messages),
            completion_tokens=20,
            latency_seconds=0.01,
            response_id=f"reply-{self.calls}",
            endpoint="http://fake/v1/chat/completions",
            request_sampling={
                **sampling,
                "max_tokens": max_new_tokens,
                "custom_params": {"thinking_budget": thinking_token_budget},
            },
            recorded_messages=[],
        )

    async def close(self):
        return None


class _RoleBackend(_FakeBackend):
    async def complete(
        self, messages, sampling, *, max_new_tokens=None, thinking_token_budget=None
    ):
        reply = await super().complete(
            messages,
            sampling,
            max_new_tokens=max_new_tokens,
            thinking_token_budget=thinking_token_budget,
        )
        content = (
            "inspect the target"
            if self.spec.name.endswith("-planner")
            else "<click>1"
        )
        return BackendReply(
            content=content,
            reasoning_content=reply.reasoning_content,
            finish_reason=reply.finish_reason,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            latency_seconds=reply.latency_seconds,
            response_id=reply.response_id,
            endpoint=reply.endpoint,
            request_sampling=reply.request_sampling,
            recorded_messages=reply.recorded_messages,
        )


def _write_config(path: Path, output: Path) -> None:
    path.write_text(
        f"""
experiment:
  id: test-eval
  output_dir: {output}
run:
  base_seed: 0
  max_concurrent_episodes: 2
  resume: skip_completed
  record_images: true
defaults:
  max_turns: 3
  response_length_per_turn: 64
  sampling:
    temperature: 0
models:
  - name: fake-model
    served_model: fake/model
    base_url: http://fake/v1
    api_key: super-secret
    max_concurrency: 2
    thinking_token_budget: 32
    thinking_budget_field: custom_params.thinking_budget
    thinking_answer_reserve: 8
envs:
  - name: EvaluationCounter
    tag: concat
    n_envs: 1
    seed_list: [10]
    harness: concat
    config: &env
      finish_after: 3
  - name: EvaluationCounter
    tag: no-concat
    n_envs: 1
    seed_list: [11]
    harness: no_concat
    config: *env
  - name: EvaluationCounter
    tag: compact
    n_envs: 1
    seed_list: [12]
    harness: compact
    harness_config:
      budget: 500
      summary_budget: 32
    config: *env
""",
        encoding="utf-8",
    )


def test_render_message_keeps_image_order():
    from PIL import Image

    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    actual, shadow = render_message({
        "role": "user",
        "content": [
            {"type": "text", "text": "before"},
            {"type": "image"},
            {"type": "text", "text": "between"},
            {"type": "image"},
        ],
        "images": [first, second],
    })
    assert [part["type"] for part in actual["content"]] == [
        "text", "image_url", "text", "image_url"
    ]
    assert shadow["content"][1] == {"type": "image", "image_index": 0}
    assert shadow["content"][3] == {"type": "image", "image_index": 1}
    first_url = actual["content"][1]["image_url"]["url"]
    assert first_url.startswith("data:image/jpeg;base64,")
    encoded = first_url.partition(",")[2]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.size == (2, 2)


def test_episode_store_records_jpeg_images(tmp_path):
    from PIL import Image

    store = EpisodeStore(tmp_path, record_images=True)
    root = tmp_path / "episode"
    image = Image.new("RGBA", (3, 2), (255, 0, 0, 128))
    descriptor = store._image(root, image, {})

    assert descriptor["media_type"] == "image/jpeg"
    assert descriptor["path"].endswith(".jpg")
    target = root / descriptor["path"]
    assert target.read_bytes().startswith(b"\xff\xd8")
    with Image.open(target) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.mode == "RGB"
        assert decoded.size == (3, 2)


def test_env_action_preserves_model_response_accounting() -> None:
    response = type(
        "Response",
        (),
        {"text": "raw", "call_id": 7, "token_ids": [11, 12]},
    )()
    action = EnvAction({"kind": "km", "keys": ["W"]}, response)
    assert action.value == {"kind": "km", "keys": ["W"]}
    assert action.text == "raw"
    assert action.call_id == 7
    assert action.token_ids == [11, 12]


def test_thinking_budget_maps_to_engine_field_and_reserves_final_tokens():
    import httpx

    async def exercise(field: str):
        requests = []

        def reply(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "reply",
                "choices": [{
                    "message": {"content": "<answer>Right</answer>"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            })

        backend = OpenAIChatBackend(ModelSpec(
            name="engine",
            served_model="model",
            base_urls=("http://engine/v1",),
            thinking_token_budget=900,
            thinking_budget_field=field,
            thinking_answer_reserve=200,
        ))
        await backend._client.aclose()
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        result = await backend.complete(
            [{"role": "user", "content": "go"}],
            {},
            max_new_tokens=1000,
            thinking_token_budget=900,
        )
        await backend.close()
        return requests[0], result

    sglang_request, sglang_reply = asyncio.run(exercise("custom_params.thinking_budget"))
    assert sglang_request["custom_params"] == {"thinking_budget": 800}
    assert sglang_reply.request_sampling["custom_params"] == {"thinking_budget": 800}

    vllm_request, _ = asyncio.run(exercise("thinking_token_budget"))
    assert vllm_request["thinking_token_budget"] == 800


def test_runner_composes_harnesses_records_and_resumes(tmp_path):
    register_env("EvaluationCounter", _CounterEnv)
    config_path = tmp_path / "eval.yaml"
    output = tmp_path / "runs"
    _write_config(config_path, output)
    config = load_config(config_path)
    assert registered_backends() == ("openai",)
    assert config.models[0].backend == "openai"
    assert config.public_config["models"][0]["api_key"] == "<redacted>"
    filtered = EvaluationRunner(
        config, backend_factory=_FakeBackend, tags={"concat"}, seeds={10}
    )
    assert [(job.environment.tag, job.seed) for job in filtered.jobs()] == [("concat", 10)]

    _FakeBackend.calls = 0
    summary = asyncio.run(EvaluationRunner(config, backend_factory=_FakeBackend).run())
    assert summary["recorded"] == 3
    assert {value["harness"] for value in summary["tasks"].values()} == {
        "concat", "no_concat", "compact"
    }
    first_call_count = _FakeBackend.calls
    assert first_call_count > 9  # compact includes at least one summary call

    run_root = output / "test-eval"
    results = sorted(run_root.glob("*/tag_*/seed_*/result.json"))
    assert len(results) == 3
    for path in results:
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["status"] == "completed"
        assert result["steps"] == 3
        assert result["format_compliance"] == 1.0
        assert result["metrics"] == {
            "success": 1.0,
            "progress": 1.0,
            "format_compliance": 1.0,
        }
        assert result["reset_info"]["seed"] == result["seed"]
        assert result["length_limited_calls"] == 0
        assert result["empty_final_content_calls"] == 0
        assert result["trajectory"][0]["sampling"]["max_tokens"] == 64
        assert result["trajectory"][0]["sampling"]["custom_params"] == {
            "thinking_budget": 32
        }
        first_observation = result["trajectory"][0]["transition"]["observation"]
        assert first_observation["screen_size"] == [1280, 720]
        assert first_observation["proprio"] == {"velocity": [0.0, 1.0]}
        assert (path.parent / "trajectory.jsonl").is_file()
        assert (path.parent / "transcript.txt").is_file()

    for task in summary["tasks"].values():
        assert task["metrics"] == {
            "success": 1.0,
            "progress": 1.0,
            "format_compliance": 1.0,
        }

    # The second invocation constructs a backend but must make no model call: all three
    # stable model/tag/seed completion markers match the same config hash.
    second = asyncio.run(EvaluationRunner(config, backend_factory=_FakeBackend).run())
    assert second["recorded"] == 3
    assert _FakeBackend.calls == first_call_count

    index = Index(output)
    assert index.runs() == [{"id": "test-eval", "recorded": 3}]
    assert len(index.run("test-eval")["jobs"]) == 3


def test_external_multi_role_harness_uses_role_bundle_and_unique_call_ids(tmp_path):
    register_env("EvaluationCounter", _CounterEnv)
    config_path = tmp_path / "multi-role.yaml"
    output = tmp_path / "runs"
    config_path.write_text(
        f"""
experiment:
  id: multi-role-eval
  output_dir: {output}
run:
  record_images: false
models:
  - name: planner-actor
    sampling:
      temperature: 0
    roles:
      planner:
        served_model: fake/planner
        base_url: http://planner/v1
      actor:
        served_model: fake/actor
        base_url: http://actor/v1
envs:
  - name: EvaluationCounter
    tag: bi-level
    seed_list: [4]
    max_turns: 2
    harness: _test_two_role
    config:
      finish_after: 1
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    model = config.models[0]
    assert model.available_roles == {"planner", "actor"}
    assert model.for_role("planner").sampling == {"temperature": 0}
    summary = asyncio.run(EvaluationRunner(config, backend_factory=_RoleBackend).run())
    assert summary["tasks"]["planner-actor/bi-level"]["mean_return"] == 1.0

    result_path = (
        output / "multi-role-eval/planner-actor/tag_bi-level/seed_4/result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["served_model"] == {
        "planner": "fake/planner",
        "actor": "fake/actor",
    }
    assert result["model_roles"] == ["actor", "planner"]
    calls = result["trajectory"]
    assert [call["call_id"] for call in calls] == [1, 2]
    assert [call["model_role"] for call in calls] == ["planner", "actor"]
    assert calls[0]["transition"] is None
    assert calls[1]["transition"]["reward"] == 1.0


def test_external_multi_role_harness_is_rejected_by_training() -> None:
    from vagen_agent.rollout import _check_axes_agree

    with pytest.raises(ValueError, match="Use standalone evaluation"):
        _check_axes_agree(RunConfig(harness="_test_two_role"), object())
