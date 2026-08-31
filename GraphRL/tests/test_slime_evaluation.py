from __future__ import annotations

import asyncio
import json
from pathlib import Path

from vagen_agent.envs import get_env_cls, register_env
from vagen_agent.evaluation.runner import run_evaluation

from view_suite.envs.slime_adapter import adapt_legacy_env, register_viewsuite_envs
from view_suite.evaluation.config import load_legacy_config
from view_suite.evaluation.run_eval import _export_legacy_layout


class _LegacyOneStepEnv:
    def __init__(self, config):
        self.config = dict(config)

    async def reset(self, seed):
        return {"obs_str": f"question {seed}"}, {"seed": seed}

    async def system_prompt(self):
        return {"obs_str": "Answer with A or B."}

    async def step(self, action):
        return (
            {"obs_str": "finished"},
            1.0,
            True,
            {"success": True, "metrics": {"accuracy": 1.0}},
        )

    async def close(self):
        return None


def test_all_viewsuite_environment_names_are_registered():
    register_viewsuite_envs()
    names = (
        "Path2View",
        "View2Path",
        "InteractiveViewPlanning",
        "Ai2ThorPath2View",
        "Ai2ThorView2Path",
        "Ai2ThorInteractiveViewPlanning",
        "HabitatGSPath2View",
        "HabitatGSView2Path",
        "HabitatGSInteractiveViewPlanning",
    )
    assert all(get_env_cls(name).__name__ == name for name in names)


def test_legacy_eval_config_runs_through_slime_and_exports_old_layout(tmp_path):
    responses = tmp_path / "responses.json"
    responses.write_text('["A"]\n', encoding="utf-8")
    config_path = tmp_path / "eval.yaml"
    output = tmp_path / "rollouts" / "compat_eval"
    config_path.write_text(
        f"""
envs:
  - name: CompatOneStep
    n_envs: 1
    tag_id: preserved_tag_name
    seed_list: [5]
    max_turns: 1
    chat_config:
      file_path: {responses}
experiment:
  dump_dir: {output}
  default_max_turns: 1
run:
  backend: random_response
  max_concurrent_jobs: 1
  resume: skip_completed
backends:
  random_response:
    model: random
""",
        encoding="utf-8",
    )
    register_env("CompatOneStep", adapt_legacy_env("CompatOneStep", _LegacyOneStepEnv))
    config = load_legacy_config(config_path)
    summary = asyncio.run(run_evaluation(config))
    _export_legacy_layout(config)

    assert summary["recorded"] == 1
    task = summary["tasks"]["random/preserved_tag_name"]
    assert task["errors"] == 0
    assert task["success_rate"] == 1.0
    native = output / "random/tag_preserved_tag_name/seed_5/result.json"
    legacy = output / "tag_preserved_tag_name/5/metrics.json"
    assert native.is_file()
    assert legacy.is_file()
    assert json.loads(legacy.read_text(encoding="utf-8"))["success"] is True


def test_legacy_eval_cli_can_override_list_elements(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        """
envs:
  - name: Path2View
    n_envs: 8
    seed: [1, 8, 1]
    max_turns: 1
experiment:
  dump_dir: /tmp/list-override
run:
  backend: random_response
backends:
  random_response:
    model: random
""",
        encoding="utf-8",
    )

    config = load_legacy_config(
        config_path,
        ["envs.0.n_envs=1", "envs.0.seed_list=[7]"],
    )

    assert config.environments[0].seeds == (7,)
