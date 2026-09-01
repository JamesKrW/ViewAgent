from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import ClassVar

VIEWAGENT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VIEWAGENT_ROOT))
sys.path.insert(0, str(VIEWAGENT_ROOT / "GraphRL/VAGEN-SLIME"))

from vagen_agent.envs import BaseNoConcatEnv, build_env
from vagen_agent.envs import get_env_cls
from graphrl.slime import rollout as rollout_module
from view_suite.envs.habitat_gs_proxy_task import slime_adapter as adapter_module
from view_suite.envs.slime_adapter import resolve_legacy_data_paths


class _FakeLegacyEnv:
    instances: ClassVar[list[_FakeLegacyEnv]] = []

    def __init__(self, config):
        self.config = dict(config)
        self.closed = False
        self.instances.append(self)

    async def reset(self, seed):
        return {"obs_str": f"seed={seed}"}, {"seed": seed}

    async def system_prompt(self):
        return {"obs_str": "navigate"}

    async def step(self, action):
        if action == "timeout":
            return {"obs_str": "late"}, 0.0, True, {"truncated": True}
        return (
            {"obs_str": "done"},
            1.0,
            True,
            {"success": True, "metrics": {"success": 1}},
        )

    async def close(self):
        self.closed = True


def test_graphrl_rollout_registers_view_suite_environment() -> None:
    assert rollout_module.generate is not None
    assert get_env_cls("HabitatGSInteractiveViewPlanning") is (
        adapter_module.HabitatGSInteractiveViewPlanning
    )


def test_adapter_preserves_limits_metrics_and_done_semantics(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            adapter_module.HabitatGSInteractiveViewPlanning,
            "delegate_class",
            _FakeLegacyEnv,
        )
        env = build_env(
            adapter_module.HabitatGSInteractiveViewPlanning,
            {"max_turns": 99},
            max_turns=10,
            required_type=BaseNoConcatEnv,
        )
        observation, info = await env.reset(seed=7)
        assert observation == {"obs_str": "seed=7"}
        assert info == {"seed": 7}
        assert _FakeLegacyEnv.instances[-1].config["max_turns"] == 10

        _, reward, terminated, truncated, step_info = await env.step("submit")
        assert reward == 1.0
        assert terminated is True
        assert truncated is False
        assert step_info["success"] is True
        assert env.last_metrics == {"success": 1.0}
        delegate = _FakeLegacyEnv.instances[-1]
        await env.close()
        assert delegate.closed is True

        timeout_env = build_env(
            adapter_module.HabitatGSInteractiveViewPlanning,
            {},
            max_turns=10,
            required_type=BaseNoConcatEnv,
        )
        await timeout_env.reset(seed=8)
        _, _, terminated, truncated, _ = await timeout_env.step("timeout")
        assert terminated is False
        assert truncated is True
        await timeout_env.close()

    asyncio.run(scenario())


def test_adapter_resolves_historical_filtered_dataset_name(tmp_path) -> None:
    canonical = tmp_path / "path_to_view_test.jsonl"
    canonical.write_text("{}\n", encoding="utf-8")
    resolved = resolve_legacy_data_paths(
        {"jsonl_path": str(tmp_path / "path_to_view_test_filter.jsonl")}
    )
    assert resolved["jsonl_path"] == str(canonical)


def test_adapter_does_not_hide_missing_dataset(tmp_path) -> None:
    missing = tmp_path / "path_to_view_test_filter.jsonl"
    resolved = resolve_legacy_data_paths({"jsonl_path": str(missing)})
    assert resolved["jsonl_path"] == str(missing)
