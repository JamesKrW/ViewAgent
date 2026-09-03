"""Checks on the VAGEN/verl command line GraphRL builds.

These guard the seams that fail silently. A wrong flag here does not crash: the
job starts, reports healthy, and trains something other than what the config
says -- which is the failure mode the agent-loop flags exist to prevent.

Run directly with::

    python GraphRL/tests/test_vagen_command_builder.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from graphrl.vagen.utils.command_builder import (
    _DEFAULT_HYDRA_CONFIG_DIR,
    build_vagen_command,
    build_vagen_env,
    resolve_vagen_dir,
    resolve_verl_dir,
)

VAGEN_DIR = resolve_vagen_dir({})
_HAVE_VAGEN = (VAGEN_DIR / "vagen" / "configs" / "baseline_vllm.flags").is_file()
needs_vagen = pytest.mark.skipif(
    not _HAVE_VAGEN, reason="VAGEN submodule is not initialised"
)


def _cmd(**overrides):
    config = {"vagen_dir": str(VAGEN_DIR), **overrides}
    return build_vagen_command(config, "/models/base", Path("/tmp/run/rl"))


def _value_of(cmd, key):
    """Last value Hydra would take for ``key``, ignoring any ``+`` prefix."""
    seen = [a.split("=", 1)[1] for a in cmd if a.lstrip("+").split("=", 1)[0] == key]
    return seen[-1] if seen else None


@needs_vagen
def test_entrypoint_is_the_current_one() -> None:
    # vagen.main_ppo is gone; a stale entrypoint fails as a bare ModuleNotFoundError.
    assert _cmd()[:3] == ["python3", "-m", "vagen.training.main"]


@needs_vagen
def test_agent_loop_flags_are_carried_from_vagen_baseline() -> None:
    # Without these verl runs its own agent loop and none of VAGEN's rollout code
    # executes, while the job still looks healthy.
    cmd = _cmd()
    assert _value_of(cmd, "actor_rollout_ref.rollout.agent.agent_loop_config_path") == (
        f"{VAGEN_DIR}/vagen/configs/agent_v2.yaml"
    )
    assert _value_of(
        cmd, "actor_rollout_ref.rollout.agent.agent_loop_manager_class"
    ) == "vagen.training.agent_loop.multi_output.MultiOutputAgentLoopManager"


@needs_vagen
def test_experiment_override_replaces_the_baseline_flag_rather_than_repeating_it() -> None:
    key = "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"  # baseline sets 1
    cmd = _cmd(
        hydra_overrides={"actor_rollout_ref": {"actor": {"ppo_micro_batch_size_per_gpu": 4}}}
    )
    emitted = [a for a in cmd if a.lstrip("+").split("=", 1)[0] == key]
    assert emitted == [f"{key}=4"], "the shadowed baseline flag must not also be emitted"


@needs_vagen
def test_viewsuite_environments_are_appended_not_redefined() -> None:
    cmd = _cmd()
    appended = {
        a.split("=", 1)[0][len("+env_registry.") :]
        for a in cmd
        if a.startswith("+env_registry.")
    }
    assert "HabitatGSInteractiveViewPlanning" in appended
    assert len(appended) == 9

    # A `+` override on a key VAGEN already defines is an error, so the two
    # registries must not overlap.
    builtin = set(
        (yaml.safe_load((VAGEN_DIR / "vagen" / "configs" / "env_registry.yaml").read_text())
         or {}).get("env_registry", {})
    )
    assert appended.isdisjoint(builtin)


@needs_vagen
def test_controller_paths_win_over_the_experiment_config() -> None:
    cmd = _cmd(hydra_overrides={"trainer": {"default_local_dir": "/wrong"}})
    assert _value_of(cmd, "trainer.default_local_dir") == "/tmp/run/rl/verl_checkpoints"
    assert _value_of(cmd, "actor_rollout_ref.model.path") == "/models/base"
    assert _value_of(cmd, "critic.model.path") == "/models/base"


@needs_vagen
def test_absolute_paths_are_used_for_config_seams_that_default_to_relative() -> None:
    # VAGEN's own defaults for both only resolve when the CWD is its repo root.
    cmd = _cmd()
    assert _value_of(cmd, "data.custom_cls.path") == (
        f"{VAGEN_DIR}/vagen/training/dataset.py"
    )
    verl_dir = resolve_verl_dir(VAGEN_DIR)
    assert f"hydra.searchpath=[file://{verl_dir}/verl/trainer/config]" in cmd


@needs_vagen
def test_verl_precedes_the_installed_copy_on_pythonpath() -> None:
    # The training env ships verl 0.6.1 as a package; the checkout must win.
    parts = build_vagen_env({"vagen_dir": str(VAGEN_DIR)})["PYTHONPATH"].split(":")
    assert parts[0] == str(resolve_verl_dir(VAGEN_DIR))
    # view_suite lives in the repo root, and the registry names classes from it.
    assert str(_DEFAULT_HYDRA_CONFIG_DIR.parents[3]) in parts


def test_relative_vagen_dir_anchors_to_the_repo_not_the_cwd() -> None:
    assert resolve_vagen_dir({"vagen_dir": "GraphRL/VAGEN"}).is_absolute()
    assert resolve_vagen_dir({"vagen_dir": "GraphRL/VAGEN"}) == VAGEN_DIR


def test_missing_verl_checkout_is_reported_as_such(tmp_path: Path) -> None:
    # An uninitialised submodule leaves the directory present but empty, and the
    # resulting Hydra error never mentions verl.
    (tmp_path / "verl").mkdir()
    with pytest.raises(FileNotFoundError, match="submodule update --init --recursive"):
        resolve_verl_dir(tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
