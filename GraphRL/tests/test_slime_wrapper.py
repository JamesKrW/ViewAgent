from __future__ import annotations

import json
from pathlib import Path

from graphrl.adaptive.model import write_checkpoint_manifest
from graphrl.slime.config import build_launch_spec
from graphrl.slime import launcher as launcher_module
from graphrl.slime.wrapper import SlimeWrapper
from graphrl.vagen.utils.command_builder import build_vagen_command


def _wrapper(tmp_path: Path, *, training_steps: int = 3) -> SlimeWrapper:
    envs = tmp_path / "envs.yaml"
    envs.write_text(
        "envs:\n  - name: Sokoban\n    n_envs: 1\n    seed: [0]\n",
        encoding="utf-8",
    )
    return SlimeWrapper(
        config={
            "training_steps": training_steps,
            "_iter_num": 0,
            "slime": {
                "train_envs": str(envs),
                "eval_envs": str(envs),
                "num_gpus": 2,
                "actor_gpus": 1,
                "rollout_gpus": 1,
            },
        },
        input_paths={"model": str(tmp_path / "base_model")},
        output_paths={
            "base_dir": str(tmp_path / "rl"),
            "model": str(tmp_path / "rl" / "rl_model"),
        },
    )


def _checkpoint(root: Path, rollout_id: int, *, committed: bool) -> Path:
    path = root / "slime_checkpoints" / f"iter_{rollout_id:07d}"
    path.mkdir(parents=True)
    (path / "state.pt").write_bytes(f"actor-{rollout_id}".encode())
    if committed:
        write_checkpoint_manifest(path)
    return path


def _rollout(root: Path, step: int) -> None:
    directory = root / "rollout_data"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")
    (directory / f"{step}.complete").write_text("complete\n", encoding="utf-8")


def _hf_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def test_resume_repairs_tracker_to_latest_checkpoint_with_complete_rollout(tmp_path):
    wrapper = _wrapper(tmp_path)
    root = wrapper.output_dir
    first = _checkpoint(root, 0, committed=True)
    newer = _checkpoint(root, 1, committed=False)
    _rollout(root, 1)
    tracker = root / "slime_checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.write_text("1", encoding="utf-8")
    stale_hf = root / "hf_checkpoints" / "rollout_1"
    _hf_model(stale_hf)
    stale_output = Path(wrapper.output_paths["model"])
    _hf_model(stale_output)
    wrapper.completion_marker.write_text("{}\n", encoding="utf-8")

    wrapper._repair_resume_checkpoint()

    assert tracker.read_text(encoding="utf-8") == "0"
    assert first.is_dir()
    assert not newer.exists()
    assert not stale_hf.exists()
    assert not stale_output.exists()
    assert not wrapper.completion_marker.exists()


def test_resume_falls_back_to_initial_model_without_complete_checkpoint(tmp_path):
    wrapper = _wrapper(tmp_path)
    root = wrapper.output_dir
    checkpoint = _checkpoint(root, 0, committed=False)
    _rollout(root, 1)
    tracker = root / "slime_checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.write_text("0", encoding="utf-8")

    wrapper._repair_resume_checkpoint()

    assert not tracker.exists()
    assert not checkpoint.exists()


def test_completed_phase_requires_terminal_checkpoint_and_rollout(tmp_path):
    wrapper = _wrapper(tmp_path, training_steps=1)
    root = wrapper.output_dir
    checkpoint = _checkpoint(root, 0, committed=True)
    _hf_model(Path(wrapper.output_paths["model"]))
    wrapper.completion_marker.write_text(
        json.dumps({"config_sha256": "wrong"}), encoding="utf-8"
    )
    assert not wrapper.is_already_complete()

    from graphrl.slime.wrapper import _fingerprint

    wrapper.completion_marker.write_text(
        json.dumps({"config_sha256": _fingerprint(wrapper.launch_spec())}),
        encoding="utf-8",
    )
    assert not wrapper.is_already_complete()

    _rollout(root, 1)
    assert wrapper.is_already_complete()

    (checkpoint / "state.pt").write_bytes(b"x")
    assert not wrapper.is_already_complete()


def test_runtime_dataset_expansion_uses_data_seed_not_trainer_seed(
    tmp_path, monkeypatch
):
    envs = tmp_path / "envs.yaml"
    envs.write_text("envs: []\n", encoding="utf-8")
    spec = build_launch_spec(
        {
            "training_steps": 1,
            "slime": {
                "train_envs": str(envs),
                "eval_envs": str(envs),
                "num_gpus": 2,
                "actor_gpus": 1,
                "rollout_gpus": 1,
                "data_seed": 7,
                "seed": 42,
            },
        },
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        output_dir=tmp_path / "run",
    )
    observed = []

    def fake_build_rows(path, base_seed=0, response_length_per_turn=None):
        del path, response_length_per_turn
        observed.append(base_seed)
        return []

    monkeypatch.setattr(launcher_module, "build_rows", fake_build_rows)
    launcher_module._write_runtime_configs(spec)
    assert observed == [7, 7]


def test_legacy_command_builder_targets_slime_launcher(tmp_path):
    envs = tmp_path / "envs.yaml"
    envs.write_text("envs: []\n", encoding="utf-8")
    output = tmp_path / "run"

    command = build_vagen_command(
        {
            "training_steps": 1,
            "slime": {
                "train_envs": str(envs),
                "eval_envs": str(envs),
                "num_gpus": 2,
                "actor_gpus": 1,
                "rollout_gpus": 1,
            },
        },
        "Qwen/Qwen2.5-VL-3B-Instruct",
        output,
    )

    assert command[1:3] == ["-m", "graphrl.slime.launcher"]
    payload = json.loads((output / "slime_launch.json").read_text(encoding="utf-8"))
    assert payload["model_path"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert payload["num_rollout"] == 1


def test_train_args_use_stable_wandb_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    wrapper = _wrapper(tmp_path)
    args = launcher_module.build_train_args(wrapper.launch_spec())

    index = args.index("--wandb-run-id")
    first = args[index + 1]
    second = launcher_module.build_train_args(wrapper.launch_spec())[index + 1]
    assert first == second
    assert len(first) == 16


def test_megatron_source_root_uses_explicit_build(tmp_path, monkeypatch):
    build = tmp_path / "Megatron-LM"
    (build / "megatron" / "training").mkdir(parents=True)
    monkeypatch.setenv("VAGEN_SLIME_MEGATRON_ROOT", str(build))

    assert launcher_module._megatron_source_root() == build.resolve()


def test_launch_spec_preserves_zero_rollout_eval_only_mode(tmp_path):
    envs = tmp_path / "envs.yaml"
    envs.write_text("envs: []\n", encoding="utf-8")
    spec = build_launch_spec(
        {
            "training_steps": 0,
            "slime": {
                "train_envs": str(envs),
                "eval_envs": str(envs),
                "num_gpus": 2,
                "actor_gpus": 1,
                "rollout_gpus": 1,
            },
        },
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        output_dir=tmp_path / "run",
    )

    assert spec.num_rollout == 0
