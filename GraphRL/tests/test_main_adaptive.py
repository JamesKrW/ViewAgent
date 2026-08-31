from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from graphrl.adaptive.model import write_checkpoint_manifest, write_snapshot_manifest
from graphrl.adaptive.rollouts import durable_rollout_prefix
from graphrl.adaptive.state import AdaptiveStateStore
from graphrl.adaptive.viewsuite_traj_to_sft import (
    StrictAdaptiveInteractiveViewPlanningGraphBuilder,
)
from graphrl.llama_factory.adaptive_lf_wrapper import AdaptiveLFWrapper
from graphrl.main_adaptive import AdaptiveGraphRLController
from graphrl.slime.adaptive_wrapper import AdaptiveSlimeWrapper


def _controller_config(tmp_path: Path) -> dict:
    return {
        "experiment_dir": str(tmp_path / "experiment"),
        "initial_model_path": str(tmp_path / "base_model"),
        "iterations": 4,
        "general_overrides": {
            "rl": {"training_steps": 65, "slime": {}},
            "traj_to_sft": {"module": "unused.module"},
            "sft": {},
        },
        "iteration_overrides": {"iter1": {"rl": {"training_steps": 1}}},
        "adaptive_round_overrides": {},
        "adaptive_schedule": {
            "metric": "val-aux/ae/traj_success/mean@1",
            "direction": "maximize",
            "eval_every_steps": 20,
            "min_delta": 0.03,
            "min_relative_delta": 0.10,
            "sft_patience": 3,
            "latch_threshold": 0.1,
            "finish_patience": 8,
            "total_rl_steps": 801,
        },
    }


def _write_complete_rollouts(rollout_dir: Path, through: int) -> None:
    for step in range(1, through + 1):
        image_dir = rollout_dir / f"image_{step}"
        image_dir.mkdir(parents=True)
        (rollout_dir / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")
        (image_dir / "0.png").write_bytes(b"png")


def test_backend_horizon_is_remaining_global_budget_not_per_round_limit(tmp_path):
    controller = AdaptiveGraphRLController(_controller_config(tmp_path))
    state = {
        "round": 1,
        "phase": "rl",
        "steps_by_round": {"0": 60, "1": 40},
        "total_rl_steps": 100,
    }
    result = controller._adaptive_iter_config(1, state)

    assert result["rl"]["training_steps"] == 741
    # The trainer resumes at local step 40 and is allowed 701 new steps:
    # 40 + (801 global budget - 100 already consumed) == 741.
    assert result["rl"]["training_steps"] - state["steps_by_round"]["1"] == 701
    slime = result["rl"]["slime"]
    assert slime["eval_interval"] == 20
    assert slime["save_interval"] == 20
    assert slime["train_script"] == "graphrl/slime/train_adaptive.py"


def test_adaptive_launch_spec_uses_separate_trainer_entrypoint(tmp_path):
    train = tmp_path / "train.yaml"
    val = tmp_path / "val.yaml"
    train.write_text("envs: []\n", encoding="utf-8")
    val.write_text("envs: []\n", encoding="utf-8")
    wrapper = AdaptiveSlimeWrapper(
        config={
            "training_steps": 741,
            "_iter_num": 1,
            "slime": {"train_envs": str(train), "eval_envs": str(val)},
        },
        input_paths={"model": "/model"},
        output_paths={
            "base_dir": str(tmp_path / "iter_001" / "rl"),
            "model": str(tmp_path / "iter_001" / "rl" / "rl_model"),
        },
    )
    spec = wrapper.launch_spec()
    assert spec.train_script == "graphrl/slime/train_adaptive.py"
    assert spec.num_rollout == 741


def test_missing_root_state_recovers_from_regular_checkpoint(tmp_path):
    config = _controller_config(tmp_path)
    controller = AdaptiveGraphRLController(config)
    state = controller.store.initialize()
    state["steps_by_round"] = {"0": 20}
    state["total_rl_steps"] = 20
    state["decision_step"] = 20
    controller.store.save(state)
    checkpoint = (
        controller.experiment_dir
        / "iter_000"
        / "rl"
        / "slime_checkpoints"
        / "iter_0000019"
    )
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    _write_complete_rollouts(checkpoint.parent.parent / "rollout_data", 20)
    controller.store.snapshot_to(checkpoint)
    write_checkpoint_manifest(checkpoint)
    controller.store.path.unlink()

    controller._recover_or_guard_state()
    restored = AdaptiveStateStore(
        controller.experiment_dir, controller.adaptive_config
    ).load()
    assert restored["total_rl_steps"] == 20


def test_post_rl_resume_does_not_require_deleted_previous_sft(tmp_path):
    config = _controller_config(tmp_path)
    controller = AdaptiveGraphRLController(config)
    current_rl = controller.experiment_dir / "iter_001" / "rl" / "rl_model"
    current_rl.mkdir(parents=True)
    (current_rl / "config.json").write_text("{}", encoding="utf-8")
    (current_rl / "model.safetensors").write_bytes(b"weights")
    state = {
        "decision": "switch_to_sft",
        "decision_round": 1,
        "decision_ready": True,
    }

    assert controller._phase_input_model(1, "rl", state) == str(current_rl)


def test_existing_root_reconciles_terminal_commit_from_checkpoint(tmp_path):
    config = _controller_config(tmp_path)
    controller = AdaptiveGraphRLController(config)
    root_state = controller.store.initialize()
    root_state["decision"] = "stop_run"
    root_state["decision_step"] = 20
    root_state["decision_ready"] = False
    root_state["steps_by_round"] = {"0": 20}
    root_state["total_rl_steps"] = 20
    root_state["rounds"] = {
        "0": {"last_observed_step": 20, "decision": "stop_run"}
    }
    controller.store.save(root_state)

    committed = copy.deepcopy(root_state)
    committed["decision_ready"] = True
    committed["decision_checkpoint_step"] = 20
    checkpoint = (
        controller.experiment_dir
        / "iter_000"
        / "rl"
        / "slime_checkpoints"
        / "iter_0000019"
    )
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    _write_complete_rollouts(checkpoint.parent.parent / "rollout_data", 20)
    controller.store.snapshot_to(checkpoint, state=committed)
    write_checkpoint_manifest(checkpoint)

    controller._recover_or_guard_state()
    assert controller.store.load()["decision_ready"] is True


def test_adaptive_sft_resume_prunes_partial_newer_checkpoint(tmp_path):
    model_dir = tmp_path / "sft_model"
    complete = model_dir / "checkpoint-10"
    partial = model_dir / "checkpoint-20"

    for checkpoint, step in ((complete, 10), (partial, 20)):
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
        for rank in range(2):
            (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"rng")

    (complete / "latest").write_text("global_step10\n", encoding="utf-8")
    deepspeed_state = complete / "global_step10"
    deepspeed_state.mkdir()
    (deepspeed_state / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(2):
        (deepspeed_state / f"zero_rank_{rank}_optim_states.pt").write_bytes(b"optim")

    wrapper = object.__new__(AdaptiveLFWrapper)
    wrapper.config = {
        "n_gpus": 2,
        "hydra_overrides": {"finetuning_type": "full", "deepspeed": "z2.json"},
    }
    wrapper.output_paths = {"model": str(model_dir)}
    messages = []
    wrapper._log = messages.append

    wrapper._prune_incomplete_sft_checkpoints()

    assert complete.is_dir()
    assert not partial.exists()
    assert messages == [f"Removed incomplete SFT resume checkpoint: {partial}"]


def test_strict_adaptive_atomize_rejects_dropped_edges():
    with pytest.raises(RuntimeError, match="must render every intermediate view"):
        StrictAdaptiveInteractiveViewPlanningGraphBuilder._require_complete_atomization(
            {
                "multi_edges": 9,
                "rendered": 8,
                "dropped": 1,
                "leftover_removed": 0,
            }
        )


def test_adaptive_traj_output_requires_strict_atomize_marker(tmp_path):
    output = tmp_path / "sft_data"
    output.mkdir()
    (output / "action_gen.json").write_text("[{}]\n", encoding="utf-8")
    (output / "dataset_info.json").write_text(
        json.dumps({"action_gen": {"file_name": "action_gen.json"}}),
        encoding="utf-8",
    )
    module = SimpleNamespace(
        config={"graph_builder": {"atomize": {"enabled": True}}},
        paths=SimpleNamespace(sft_data=output),
    )

    assert not AdaptiveGraphRLController._traj_output_is_complete(module)

    graph_dir = output.parent / "graph"
    graph_dir.mkdir()
    (graph_dir / "adaptive_atomize_state.json").write_text(
        json.dumps(
            {
                "multi_edges": 9,
                "rendered": 12,
                "dropped": 0,
                "leftover_removed": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert AdaptiveGraphRLController._traj_output_is_complete(module)


def test_rebuilding_traj_output_invalidates_stale_sft_checkpoint(tmp_path):
    controller = AdaptiveGraphRLController(_controller_config(tmp_path))
    sft_base = tmp_path / "experiment" / "iter_000" / "sft"
    sft_model = sft_base / "sft_model"
    checkpoint = sft_model / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")

    wrapper = object.__new__(AdaptiveLFWrapper)
    wrapper.output_paths = {
        "base_dir": str(sft_base),
        "model": str(sft_model),
    }

    controller._clear_sft_output([wrapper])

    assert not sft_base.exists()


def test_rollout_prefix_requires_json_and_images_for_every_step(tmp_path):
    rollout_dir = tmp_path / "rollout_data"
    for step in (1, 2):
        (rollout_dir / f"image_{step}" / "images_0").mkdir(parents=True)
        (rollout_dir / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")
        (rollout_dir / f"image_{step}" / "images_0" / "0.png").write_bytes(b"png")

    assert durable_rollout_prefix(rollout_dir) == 2
    (rollout_dir / "image_2" / "images_0" / "0.png").unlink()
    assert durable_rollout_prefix(rollout_dir) == 1


def test_stale_rl_marker_is_not_authoritative_after_state_rollback(tmp_path):
    cfg = _controller_config(tmp_path)["adaptive_schedule"]
    store = AdaptiveStateStore(tmp_path, cfg)
    store.initialize()
    (tmp_path / "adaptive_schedule_config.json").write_text(
        json.dumps(cfg), encoding="utf-8"
    )

    model = tmp_path / "iter_000" / "rl" / "rl_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    marker = model.parent / ".adaptive_rl_done.json"
    marker.write_text(
        json.dumps({"decision": "switch_to_sft", "round": 0, "step": 60}),
        encoding="utf-8",
    )

    wrapper = AdaptiveSlimeWrapper(
        config={"_iter_num": 0, "training_steps": 801},
        input_paths={"model": "unused"},
        output_paths={"base_dir": str(model.parent), "model": str(model)},
    )
    assert not wrapper.is_already_complete()


def test_recovery_rolls_back_checkpoint_ahead_of_rollout_prefix(tmp_path):
    controller = AdaptiveGraphRLController(_controller_config(tmp_path))
    root_state = controller.store.initialize()
    root_state["phase"] = "sft"
    root_state["decision"] = "switch_to_sft"
    root_state["decision_ready"] = True
    root_state["decision_step"] = 60
    root_state["steps_by_round"] = {"0": 60}
    root_state["total_rl_steps"] = 60
    root_state["rounds"] = {"0": {"last_observed_step": 60}}
    controller.store.save(root_state)

    rollout_dir = controller.experiment_dir / "iter_000" / "rl" / "rollout_data"
    for step in range(1, 38):
        (rollout_dir / f"image_{step}").mkdir(parents=True)
        (rollout_dir / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")
        (rollout_dir / f"image_{step}" / "0.png").write_bytes(b"png")

    checkpoint_root = rollout_dir.parent / "slime_checkpoints"
    state20 = copy.deepcopy(root_state)
    state20["phase"] = "rl"
    state20["decision"] = "continue_rl"
    state20["decision_ready"] = False
    state20["decision_step"] = 20
    state20["steps_by_round"] = {"0": 20}
    state20["total_rl_steps"] = 20
    state20["rounds"] = {"0": {"last_observed_step": 20}}
    checkpoints = {}
    for step, state in ((20, state20), (60, root_state)):
        checkpoint = checkpoint_root / f"iter_{step - 1:07d}"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(b"data")
        controller.store.snapshot_to(checkpoint, state=state)
        write_checkpoint_manifest(checkpoint)
        checkpoints[step] = checkpoint

    stale_critic = checkpoint_root / "critic" / "iter_0000059"
    stale_critic.mkdir(parents=True)
    stale_dataset = (
        checkpoint_root / "rollout" / "global_dataset_state_dict_59.pt"
    )
    stale_dataset.parent.mkdir(parents=True)
    stale_dataset.write_bytes(b"dataset")
    stale_hf = rollout_dir.parent / "hf_checkpoints" / "rollout_59"
    stale_hf.mkdir(parents=True)

    stale_model = rollout_dir.parent / "rl_model"
    stale_model.mkdir()
    (controller.experiment_dir / "iter_000" / "traj_to_sft").mkdir()
    (controller.experiment_dir / "iter_000" / "sft").mkdir()

    controller._recover_or_guard_state()

    restored = controller.store.load()
    assert restored["phase"] == "rl"
    assert restored["total_rl_steps"] == 20
    assert durable_rollout_prefix(rollout_dir) == 20
    assert not (rollout_dir / "21.jsonl").exists()
    assert not (rollout_dir / "image_21").exists()
    assert checkpoints[20].is_dir()
    assert not checkpoints[60].exists()
    assert not stale_critic.exists()
    assert not stale_dataset.exists()
    assert not stale_hf.exists()
    assert not stale_model.exists()
    assert not (controller.experiment_dir / "iter_000" / "traj_to_sft").exists()
    assert not (controller.experiment_dir / "iter_000" / "sft").exists()


def test_recovery_accepts_latched_checkpoint_with_jsonl_but_without_images(tmp_path):
    controller = AdaptiveGraphRLController(_controller_config(tmp_path))
    root_state = controller.store.initialize()
    root_state.update(
        {
            "phase": "rl",
            "decision": "continue_rl",
            "decision_step": 60,
            "latched": True,
            "steps_by_round": {"0": 60},
            "total_rl_steps": 60,
            "rounds": {"0": {"last_observed_step": 60}},
        }
    )
    controller.store.save(root_state)

    checkpoint = (
        controller.experiment_dir
        / "iter_000"
        / "rl"
        / "slime_checkpoints"
        / "iter_0000059"
    )
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    controller.store.snapshot_to(checkpoint, state=root_state)
    write_checkpoint_manifest(checkpoint)
    rollout_dir = checkpoint.parent.parent / "rollout_data"
    for step in range(1, 61):
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")
        (rollout_dir / f"{step}.complete").write_text("complete\n", encoding="utf-8")

    controller._recover_or_guard_state()

    restored = controller.store.load()
    assert restored["latched"] is True
    assert restored["total_rl_steps"] == 60
    assert checkpoint.is_dir()


def test_rl_state_ahead_rejects_model_only_best_snapshot(tmp_path):
    controller = AdaptiveGraphRLController(_controller_config(tmp_path))
    root_state = controller.store.initialize()
    snapshot_rel = "adaptive_best_snapshots/round_000_step_000020"
    root_state.update(
        {
            "phase": "rl",
            "decision": "continue_rl",
            "decision_step": 20,
            "latched": True,
            "steps_by_round": {"0": 20},
            "total_rl_steps": 20,
            "run_best": {
                "score": 0.1,
                "round": 0,
                "step": 20,
                "checkpoint": snapshot_rel,
            },
            "rounds": {
                "0": {
                    "best_score": 0.1,
                    "best_step": 20,
                    "checkpoint": snapshot_rel,
                    "last_observed_step": 20,
                }
            },
        }
    )
    controller.store.save(root_state)
    snapshot = controller.experiment_dir / snapshot_rel
    hf_model = snapshot / "actor" / "huggingface"
    hf_model.mkdir(parents=True)
    (hf_model / "config.json").write_text("{}\n", encoding="utf-8")
    (hf_model / "model.safetensors").write_bytes(b"weights")
    controller.store.snapshot_to(snapshot, state=root_state)
    write_snapshot_manifest(snapshot)
    _write_complete_rollouts(
        controller.experiment_dir / "iter_000" / "rl" / "rollout_data", 20
    )

    with pytest.raises(RuntimeError, match="best-model snapshot is model-only"):
        controller._recover_or_guard_state()


def test_missing_root_state_can_recover_from_step_zero_snapshot(tmp_path):
    controller = AdaptiveGraphRLController(_controller_config(tmp_path))
    state = controller.store.initialize()
    snapshot = (
        controller.experiment_dir
        / "adaptive_best_snapshots"
        / "round_000_step_000000"
    )
    hf_model = snapshot / "actor" / "huggingface"
    hf_model.mkdir(parents=True)
    (hf_model / "config.json").write_text("{}\n", encoding="utf-8")
    (hf_model / "model.safetensors").write_bytes(b"weights")
    controller.store.snapshot_to(snapshot, state=state)
    write_snapshot_manifest(snapshot)
    controller.store.path.unlink()

    checkpoint_root = (
        controller.experiment_dir / "iter_000" / "rl" / "slime_checkpoints"
    )
    stale_checkpoint = checkpoint_root / "iter_0000000"
    stale_checkpoint.mkdir(parents=True)
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    tracker.write_text("0", encoding="utf-8")
    stale_rollout = (
        controller.experiment_dir / "iter_000" / "rl" / "rollout_data" / "1.jsonl"
    )
    stale_rollout.parent.mkdir(parents=True)
    stale_rollout.write_text("{}\n", encoding="utf-8")

    controller._recover_or_guard_state()

    restored = controller.store.load()
    assert restored["total_rl_steps"] == 0
    assert not tracker.exists()
    assert not stale_checkpoint.exists()
    assert not stale_rollout.exists()
