from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from graphrl.adaptive.model import write_checkpoint_manifest
from graphrl.llama_factory.lf_wrapper import (
    LFWrapper,
    _ensure_torchrun_on_path,
    _llamafactory_cli_command,
)
from graphrl.state import ModuleState
from graphrl.traj_to_sft.traj_to_sft_base import TrajToSFTModule, TrajToSFTPaths
from graphrl.utils.progress import detect_progress


def _model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def test_llamafactory_command_uses_console_script_when_available(tmp_path, monkeypatch):
    config = tmp_path / "sft.yaml"
    monkeypatch.setattr("graphrl.llama_factory.lf_wrapper.shutil.which", lambda _: "/bin/lf")

    assert _llamafactory_cli_command("train", config) == [
        "/bin/lf", "train", str(config),
    ]


def test_llamafactory_command_falls_back_to_current_python(tmp_path, monkeypatch):
    config = tmp_path / "sft.yaml"
    monkeypatch.setattr("graphrl.llama_factory.lf_wrapper.shutil.which", lambda _: None)
    monkeypatch.setattr("graphrl.llama_factory.lf_wrapper.sys.executable", "/env/python")

    assert _llamafactory_cli_command("export", config) == [
        "/env/python", "-m", "llamafactory.cli", "export", str(config),
    ]


def test_sft_exposes_current_environment_torchrun_when_path_is_not_activated(
    tmp_path, monkeypatch,
):
    python_bin = tmp_path / "env" / "bin"
    python_bin.mkdir(parents=True)
    (python_bin / "torchrun").touch()
    monkeypatch.setattr(
        "graphrl.llama_factory.lf_wrapper.sys.executable", str(python_bin / "python")
    )
    monkeypatch.setattr(
        "graphrl.llama_factory.lf_wrapper.shutil.which", lambda *_args, **_kwargs: None
    )
    env = {"PATH": "/usr/bin"}

    _ensure_torchrun_on_path(env)

    assert env["PATH"] == f"{python_bin}:/usr/bin"


def test_sft_preserves_existing_relocated_torchrun_shim(monkeypatch):
    monkeypatch.setattr(
        "graphrl.llama_factory.lf_wrapper.shutil.which",
        lambda *_args, **_kwargs: "/tmp/conda-entrypoints/torchrun",
    )
    env = {"PATH": "/tmp/conda-entrypoints:/usr/bin"}

    _ensure_torchrun_on_path(env)

    assert env["PATH"] == "/tmp/conda-entrypoints:/usr/bin"


def test_fixed_resume_rejects_uncommitted_slime_rl_model(tmp_path):
    iteration = tmp_path / "iter_000"
    rl = iteration / "rl"
    _model(rl / "rl_model")
    (rl / "slime_launch.json").write_text("{}\n", encoding="utf-8")

    start_iter, start_phase, output = detect_progress(tmp_path, 1)

    assert (start_iter, start_phase, output) == (0, 0, None)


def test_fixed_resume_accepts_only_fully_committed_slime_rl_phase(tmp_path):
    iteration = tmp_path / "iter_000"
    rl = iteration / "rl"
    _model(rl / "rl_model")
    checkpoint = rl / "slime_checkpoints" / "iter_0000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"state")
    write_checkpoint_manifest(checkpoint)
    rollout = rl / "rollout_data"
    rollout.mkdir(parents=True)
    for step in (1, 2):
        (rollout / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")
        (rollout / f"{step}.complete").write_text("complete\n", encoding="utf-8")
    (rl / ".slime_rl_done.json").write_text(
        json.dumps(
            {
                "backend": "vagen-slime",
                "rollout_id": 1,
                "num_rollout": 2,
                "config_sha256": "test",
            }
        ),
        encoding="utf-8",
    )

    start_iter, start_phase, output = detect_progress(tmp_path, 1)

    assert (start_iter, start_phase) == (0, 1)
    assert output is not None
    assert output.model_path == str(rl / "rl_model")


def test_fixed_resume_keeps_pre_slime_model_directory_compatible(tmp_path):
    model = tmp_path / "iter_000" / "rl" / "rl_model"
    _model(model)

    start_iter, start_phase, output = detect_progress(tmp_path, 1)

    assert (start_iter, start_phase) == (0, 1)
    assert output is not None
    assert output.model_path == str(model)


def test_fixed_resume_rejects_partial_sft_model(tmp_path):
    iteration = tmp_path / "iter_000"
    sft_model = iteration / "sft" / "sft_model"
    sft_model.mkdir(parents=True)
    (sft_model / "config.json").write_text("{}\n", encoding="utf-8")

    start_iter, start_phase, output = detect_progress(tmp_path, 1)

    assert (start_iter, start_phase, output) == (0, 0, None)


def test_phase_wrappers_require_complete_outputs(tmp_path):
    sft_model = tmp_path / "sft_model"
    sft_model.mkdir()
    (sft_model / "config.json").write_text("{}\n", encoding="utf-8")
    sft = LFWrapper(
        config={},
        input_paths={},
        output_paths={"model": str(sft_model)},
    )
    assert not sft.is_already_complete()
    (sft_model / "model.safetensors").write_bytes(b"weights")
    assert sft.is_already_complete()

    paths = TrajToSFTPaths(
        base_dir=tmp_path,
        rollout_data=tmp_path / "rollout_data",
        rl_model=tmp_path / "rl_model",
        sft_data=tmp_path / "sft_data",
    )
    paths.sft_data.mkdir()
    (paths.sft_data / "dataset_info.json").write_text("{}\n", encoding="utf-8")
    traj = TrajToSFTModule(config={}, paths=paths)
    assert not traj.is_already_complete()
    (paths.sft_data / ".phase_done").write_text("done\n", encoding="utf-8")
    assert traj.is_already_complete()


def test_fixed_sft_failure_retains_checkpoint_for_next_resume(tmp_path):
    model = tmp_path / "sft_model"
    checkpoint = model / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    wrapper = LFWrapper(
        config={},
        input_paths={},
        output_paths={"model": str(model)},
    )
    wrapper._process = SimpleNamespace(poll=lambda: 1)
    wrapper._state = ModuleState.LAUNCHED

    assert not wrapper.is_done()
    assert wrapper.state == ModuleState.FAILED
    wrapper.kill()
    assert checkpoint.is_dir()
