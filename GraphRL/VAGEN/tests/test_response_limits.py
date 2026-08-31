"""Regression checks for per-call response and context limits.

Run directly with::

    python tests/test_response_limits.py
"""

from __future__ import annotations

import asyncio
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

try:
    from hidagent.datasets._common.hid_vocab import ALL_SPECIAL_TOKENS
except ModuleNotFoundError:
    ALL_SPECIAL_TOKENS = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vagen_agent.config import EnvSpec  # noqa: E402
from vagen_agent.harness import BaseHarness  # noqa: E402
from vagen_agent.harness.compact import CompactHarness  # noqa: E402
from vagen_agent.rollout import (  # noqa: E402
    Response,
    Usage,
    _response_decoder,
    _response_limit,
)

SOKOBAN_LAUNCHER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/train/sokoban/run_sokoban.py"),
    run_name="sokoban_launcher_test",
)
SokobanConfig = SOKOBAN_LAUNCHER["Config"]


class SizedClient:
    def __init__(self, prompt_tokens: int):
        self.prompt_tokens = prompt_tokens

    def size(self, _messages) -> int:
        return self.prompt_tokens


def test_room_uses_per_turn_cap_when_context_is_large() -> None:
    messages = [{"role": "user", "content": "x"}]
    assert BaseHarness.room(SizedClient(1000), messages, 8192, 512) == {
        "max_new_tokens": 512
    }


def test_room_uses_remaining_context_when_it_is_smaller() -> None:
    messages = [{"role": "user", "content": "x"}]
    assert BaseHarness.room(SizedClient(8000), messages, 8192, 512) == {
        "max_new_tokens": 192
    }


def test_room_accepts_a_separate_summary_cap() -> None:
    messages = [{"role": "user", "content": "x"}]
    assert BaseHarness.room(SizedClient(1000), messages, 8192, 256) == {
        "max_new_tokens": 256
    }


def test_dataset_limit_can_tighten_but_not_enlarge_cli_limit() -> None:
    spec = EnvSpec(env_name="Sokoban", response_length_per_turn=512)
    # VAGEN gives a whole multi-turn row 4000 response tokens while bounding each
    # environment action at 512.  The global row-sized value must not leak into one call.
    args = SimpleNamespace(rollout_max_response_len=4000)
    assert _response_limit(8192, spec, args) == 512

    spec.response_length_per_turn = 1024
    assert _response_limit(8192, spec, args) == 1024

    spec.response_length_per_turn = 5000
    assert _response_limit(8192, spec, args) == 4000


def test_sokoban_launcher_keeps_row_and_turn_budgets_separate() -> None:
    cfg = SokobanConfig()
    assert cfg.turn_tokens == 512
    assert cfg.rollout_response_tokens == 4000
    assert cfg.rollout_context_tokens == 8192

    compact = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "examples/train/sokoban/compact_ppo.yaml").read_text()
    )
    assert compact["compact_budget"] == 4000
    assert compact["compact_summary_budget"] == 512


def test_sokoban_launcher_defaults_match_vision_environment_and_repo_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = SokobanConfig()
    assert SOKOBAN_LAUNCHER["REPO"] == root
    assert cfg.model_name == "Qwen3-VL-4B-Instruct"
    assert Path(cfg.data_dir) == root / "data"
    assert Path(cfg.runs_dir) == root / "runs"
    assert SOKOBAN_LAUNCHER["_harness_name"](cfg.harness_config) == "no_concat"
    assert SOKOBAN_LAUNCHER["_default_experiment_name"](
        cfg.model_name, cfg.harness_config
    ) == "sokoban_no_concat_Qwen3-VL-4B-Instruct"


def test_launcher_propagates_pytorch_wheel_cudnn_to_ray_workers(
    monkeypatch, tmp_path: Path
) -> None:
    env_prefix = tmp_path / "slime"
    inherited = f"{env_prefix}/lib:/system/lib"
    monkeypatch.setenv("CONDA_PREFIX", str(env_prefix))
    monkeypatch.setenv("LD_LIBRARY_PATH", inherited)

    paths = SOKOBAN_LAUNCHER["runtime_ld_library_path"]().split(":")

    python_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    assert paths[:3] == [
        str(env_prefix / "lib" / python_lib / "site-packages/nvidia/cudnn/lib"),
        str(env_prefix / "lib"),
        "/system/lib",
    ]
    assert len(paths) == len(set(paths))


def test_qwen25_vl_uses_the_existing_qwen25_language_tower_spec() -> None:
    assert SOKOBAN_LAUNCHER["megatron_model_type_of"](
        "Qwen2.5-VL-7B-Instruct", "/unused"
    ) == "qwen2.5-7B"


def test_full_context_refuses_another_generation() -> None:
    messages = [{"role": "user", "content": "x"}]
    try:
        BaseHarness.room(SizedClient(8192), messages, 8192, 512)
    except RuntimeError as exc:
        assert "leaving no room" in str(exc)
    else:
        raise AssertionError("a full context window incorrectly allowed another generation")


def test_compact_uses_action_and_summary_limits_separately() -> None:
    class Client(SizedClient):
        def __init__(self):
            super().__init__(prompt_tokens=1000)
            self.limits = []

        async def create(self, _messages, **sampling):
            self.limits.append(sampling["max_new_tokens"])
            return Response(text="state", token_ids=[1], logprobs=[-0.1],
                            usage=Usage(prompt_tokens=1000, completion_tokens=1))

    class Env:
        def __init__(self):
            self.turn = 0

        async def system_prompt(self):
            return {"role": "system", "content": "system"}

        async def reset(self):
            return {"role": "user", "content": "observation 0"}, {}

        async def step(self, _response):
            self.turn += 1
            return ({"role": "user", "content": f"observation {self.turn}"},
                    0.0, self.turn == 2, False, {})

    client = Client()
    harness = CompactHarness(budget=1, summary_budget=256,
                             window=8192, response_limit=512)
    asyncio.run(harness.run_episode(client, Env()))
    assert client.limits == [512, 256, 512]


@pytest.mark.skipif(ALL_SPECIAL_TOKENS is None, reason="optional hidagent package is not installed")
def test_hid_rollout_selects_a_token_id_decoder() -> None:
    class Tokenizer:
        def get_vocab(self):
            return {token: index for index, token in enumerate(ALL_SPECIAL_TOKENS)}

    class Model:
        tokenizer = Tokenizer()

        def decode_generated(self, ids, *, preserve_special_tokens):
            assert preserve_special_tokens == ALL_SPECIAL_TOKENS
            return "".join({2: "<click>", 3: "5"}.get(value, "") for value in ids)

    spec = EnvSpec(
        env_name="RemoteEnv",
        config={"dialect": "hid_v1"},
    )
    decoder = _response_decoder(spec, Model())
    assert decoder is not None
    assert decoder([1, 2, 3], "server stripped this") == "<click>5"


@pytest.mark.skipif(ALL_SPECIAL_TOKENS is None, reason="optional hidagent package is not installed")
def test_hid_rollout_rejects_a_base_tokenizer_without_the_action_vocab() -> None:
    class Tokenizer:
        def get_vocab(self):
            return {"ordinary": 0}

    spec = EnvSpec(env_name="RemoteEnv", config={"dialect": "hid_v1"})
    try:
        _response_decoder(spec, type("Model", (), {"tokenizer": Tokenizer()})())
    except ValueError as exc:
        assert "tokenizer is missing" in str(exc)
        assert "adding tokens only at rollout" in str(exc)
    else:
        raise AssertionError("a tokenizer without HID tokens must be rejected")


def main() -> None:
    test_room_uses_per_turn_cap_when_context_is_large()
    test_room_uses_remaining_context_when_it_is_smaller()
    test_room_accepts_a_separate_summary_cap()
    test_dataset_limit_can_tighten_but_not_enlarge_cli_limit()
    test_sokoban_launcher_keeps_row_and_turn_budgets_separate()
    test_full_context_refuses_another_generation()
    test_compact_uses_action_and_summary_limits_separately()
    if ALL_SPECIAL_TOKENS is not None:
        test_hid_rollout_selects_a_token_id_decoder()
        test_hid_rollout_rejects_a_base_tokenizer_without_the_action_vocab()
    print("PASS response limits")


if __name__ == "__main__":
    main()
