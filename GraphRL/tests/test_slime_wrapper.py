from __future__ import annotations

import json
import os
import sys
from importlib.machinery import ModuleSpec
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


def test_megatron_source_root_uses_packaged_runtime(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    training = site_packages / "megatron" / "training"
    training.mkdir(parents=True)
    module_spec = ModuleSpec("megatron.training", loader=None, is_package=True)
    module_spec.submodule_search_locations = [str(training)]
    monkeypatch.delenv("VAGEN_SLIME_MEGATRON_ROOT", raising=False)
    monkeypatch.setattr(launcher_module, "VAGEN_SLIME_ROOT", tmp_path / "missing")
    monkeypatch.setattr(
        launcher_module.importlib.util,
        "find_spec",
        lambda name: module_spec if name == "megatron.training" else None,
    )

    assert launcher_module._megatron_source_root() == site_packages.resolve()


def test_model_type_uses_hf_metadata_for_snapshot_directory(tmp_path, monkeypatch):
    snapshot = tmp_path / "cc594898137f460bfe9f0759e9844b3ce807cfb5"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct"}),
        encoding="utf-8",
    )
    models = tmp_path / "slime" / "scripts" / "models"
    models.mkdir(parents=True)
    (models / "qwen2.5-7B.sh").write_text("", encoding="utf-8")
    monkeypatch.setattr(launcher_module, "VAGEN_SLIME_ROOT", tmp_path)

    assert launcher_module._model_type(str(snapshot), None) == "qwen2.5-7B"


def test_launch_prefers_relocated_cli_shim_before_interpreter_bin(tmp_path, monkeypatch):
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    monkeypatch.setenv("GRAPHRL_CLI_SHIM_DIR", str(shim_dir))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(launcher_module, "_model_type", lambda *args: "qwen2.5-7B")
    monkeypatch.setattr(launcher_module, "build_train_args", lambda spec: ["--help"])
    monkeypatch.setattr(launcher_module, "_resolve_model_path", lambda path: path)
    monkeypatch.setattr(launcher_module, "_megatron_source_root", lambda: tmp_path)
    monkeypatch.setattr(launcher_module, "execute_train", lambda **kwargs: None)

    wrapper = _wrapper(tmp_path)
    launcher_module.launch(wrapper.launch_spec())

    path_parts = os.environ["PATH"].split(os.pathsep)
    assert path_parts[0] == str(shim_dir)
    assert path_parts[1] == str(Path(sys.executable).resolve().parent)


def test_launch_forwards_explicit_slime_host_ip_to_ray_runtime(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setenv("SLIME_HOST_IP", "127.0.0.1")
    monkeypatch.setattr(launcher_module, "_model_type", lambda *args: "qwen2.5-7B")
    monkeypatch.setattr(launcher_module, "build_train_args", lambda spec: ["--help"])
    monkeypatch.setattr(launcher_module, "_resolve_model_path", lambda path: path)
    monkeypatch.setattr(launcher_module, "_megatron_source_root", lambda: tmp_path)
    monkeypatch.setattr(
        launcher_module,
        "execute_train",
        lambda **kwargs: captured.update(kwargs),
    )

    launcher_module.launch(_wrapper(tmp_path).launch_spec())

    assert captured["extra_env_vars"]["SLIME_HOST_IP"] == "127.0.0.1"


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


def test_colocate_uses_all_physical_gpus_for_actor_and_rollout(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(launcher_module, "build_rows", lambda *args, **kwargs: [])
    envs = tmp_path / "envs.yaml"
    envs.write_text("envs: []\n", encoding="utf-8")
    spec = build_launch_spec(
        {
            "training_steps": 1,
            "slime": {
                "train_envs": str(envs),
                "eval_envs": str(envs),
                "num_gpus": 8,
                "colocate": True,
            },
        },
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        output_dir=tmp_path / "run",
    )

    assert spec.actor_gpus == 8
    assert spec.rollout_gpus == 8
    assert spec.colocate is True
    args = launcher_module.build_train_args(spec)
    assert "--colocate" in args
    assert args[args.index("--actor-num-gpus-per-node") + 1] == "8"
    assert args[args.index("--rollout-num-gpus") + 1] == "8"
    assert args[args.index("--num-gpus-per-node") + 1] == "8"


def test_cuda_graph_can_be_disabled_for_multi_engine_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(launcher_module, "build_rows", lambda *args, **kwargs: [])
    envs = tmp_path / "envs.yaml"
    envs.write_text("envs: []\n", encoding="utf-8")
    spec = build_launch_spec(
        {
            "training_steps": 1,
            "slime": {
                "train_envs": str(envs),
                "eval_envs": str(envs),
                "num_gpus": 8,
                "colocate": True,
                "sglang_disable_cuda_graph": True,
            },
        },
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        output_dir=tmp_path / "run",
    )

    assert spec.sglang_disable_cuda_graph is True
    assert "--sglang-disable-cuda-graph" in launcher_module.build_train_args(spec)


def test_non_colocate_rejects_oversubscribed_gpu_counts(tmp_path):
    envs = tmp_path / "envs.yaml"
    envs.write_text("envs: []\n", encoding="utf-8")

    try:
        build_launch_spec(
            {
                "slime": {
                    "train_envs": str(envs),
                    "eval_envs": str(envs),
                    "num_gpus": 8,
                    "actor_gpus": 8,
                    "rollout_gpus": 8,
                },
            },
            model_path="Qwen/Qwen2.5-VL-3B-Instruct",
            output_dir=tmp_path / "run",
        )
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("expected oversubscribed disjoint placement to fail")
