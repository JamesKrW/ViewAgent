from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from graphrl.adaptive.model import (
    is_complete_checkpoint_manifest,
    is_complete_snapshot,
    write_checkpoint_manifest,
)
from graphrl.adaptive.rollouts import rollout_step_is_complete
from graphrl.adaptive.state import (
    CONFIG_FILENAME,
    DECISION_CONTINUE_RL,
    DECISION_SWITCH_TO_SFT,
    PHASE_FINALIZE_ROUND,
    PHASE_SFT,
    AdaptiveStateStore,
    normalize_adaptive_config,
)
from graphrl.slime.adaptive_wrapper import AdaptiveSlimeWrapper
from graphrl.slime import rollout as rollout_module
from graphrl.slime.schedule import SlimeAdaptiveSchedule

METRIC = "val-aux/ae/traj_success/mean@1"


def _config(**overrides):
    raw = {
        "metric": METRIC,
        "direction": "maximize",
        "eval_every_steps": 20,
        "min_delta": 0.03,
        "min_relative_delta": 0.10,
        "sft_patience": 3,
        "latch_threshold": 0.1,
        "finish_patience": 8,
        "total_rl_steps": 801,
    }
    raw.update(overrides)
    return normalize_adaptive_config(raw)


def _schedule(tmp_path: Path, round_index: int, cfg: dict) -> SlimeAdaptiveSchedule:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps(cfg), encoding="utf-8")
    return SlimeAdaptiveSchedule(tmp_path, round_index)


def _checkpoint_root(schedule: SlimeAdaptiveSchedule) -> Path:
    return (
        schedule.experiment_root
        / f"iter_{schedule.round_index:03d}"
        / "rl"
        / "slime_checkpoints"
    )


def _write_hf_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def _write_rollout_step(
    schedule: SlimeAdaptiveSchedule,
    step: int,
    *,
    images: bool = True,
) -> None:
    rollout_dir = _checkpoint_root(schedule).parent / "rollout_data"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")
    if images:
        image_dir = rollout_dir / f"image_{step}" / "images_0"
        image_dir.mkdir(parents=True)
        (image_dir / "0.png").write_bytes(b"png")
    (rollout_dir / f"{step}.complete").write_text("complete\n", encoding="utf-8")


def _write_checkpoint(
    schedule: SlimeAdaptiveSchedule,
    step: int,
    *,
    state: dict | None = None,
    images: bool = True,
) -> Path:
    rollout_id = step - 1
    root = _checkpoint_root(schedule)
    checkpoint = root / f"iter_{rollout_id:07d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"actor-state")
    critic = root / "critic" / checkpoint.name
    critic.mkdir(parents=True)
    (critic / "data.pt").write_bytes(b"critic-state")
    dataset = root / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"dataset-state")
    schedule.store.snapshot_to(checkpoint, state=state)
    write_checkpoint_manifest(checkpoint, (critic, dataset))
    for value in range(1, step + 1):
        _write_rollout_step(schedule, value, images=images)
    (root / "latest_checkpointed_iteration.txt").write_text(
        str(rollout_id), encoding="utf-8"
    )
    return checkpoint


def test_step_zero_is_round_baseline_and_late_improvement_prevents_sft(tmp_path):
    cfg = _config()
    round0 = _schedule(tmp_path, 0, cfg)
    assert round0.observe({METRIC: 0.00490196}, 0)["decision"] == DECISION_CONTINUE_RL
    round0.observe({METRIC: 0.00490196}, 20)
    round0.observe({METRIC: 0.00490196}, 40)
    assert round0.observe({METRIC: 0.00490196}, 60)["decision"] == DECISION_SWITCH_TO_SFT

    round0.store.set_phase(0, PHASE_FINALIZE_ROUND)
    round0.store.begin_next_round(0)
    round1 = _schedule(tmp_path, 1, cfg)
    baseline = round1.observe({METRIC: 0.0}, 0)
    assert baseline["misses"] == 0
    assert not baseline["new_run_best"]
    assert round1.observe({METRIC: 0.0}, 20)["misses"] == 1
    assert round1.observe({METRIC: 0.0}, 40)["misses"] == 2
    improved = round1.observe({METRIC: 0.05882353}, 60)
    assert improved["decision"] == DECISION_CONTINUE_RL
    assert improved["misses"] == 0
    assert improved["new_run_best"]


def test_real_viewsuite_curve_does_not_trigger_sft(tmp_path):
    schedule = _schedule(tmp_path, 0, _config())
    observations = (
        (0, 0.018518518518518517),
        (20, 0.021164021164021163),
        (40, 0.03439153439153439),
        (60, 0.047619047619047616),
    )
    for step, score in observations:
        result = schedule.observe({METRIC: score}, step)
        assert result["decision"] == DECISION_CONTINUE_RL
        assert result["misses"] == 0


def test_absolute_or_relative_gain_resets_patience(tmp_path):
    relative = _schedule(tmp_path / "relative", 0, _config())
    relative.observe({METRIC: 0.02}, 0)
    result = relative.observe({METRIC: 0.023}, 20)
    assert result["absolute_gain"] < 0.03
    assert result["relative_gain"] >= 0.10
    assert result["patience_reset"] is True

    absolute = _schedule(
        tmp_path / "absolute", 0, _config(min_delta=0.03, min_relative_delta=1.0)
    )
    absolute.observe({METRIC: 0.10}, 0)
    result = absolute.observe({METRIC: 0.131}, 20)
    assert result["absolute_gain"] >= 0.03
    assert result["relative_gain"] < 1.0
    assert result["patience_reset"] is True


def test_small_peaks_accumulate_against_patience_anchor(tmp_path):
    schedule = _schedule(
        tmp_path, 0, _config(min_delta=0.20, min_relative_delta=0.10)
    )
    schedule.observe({METRIC: 1.0}, 0)
    small_peak = schedule.observe({METRIC: 1.05}, 20)
    assert small_peak["is_best"] is True
    assert small_peak["patience_reset"] is False
    assert small_peak["misses"] == 1
    assert small_peak["patience_anchor_score"] == 1.0

    cumulative = schedule.observe({METRIC: 1.11}, 40)
    assert cumulative["patience_reset"] is True
    assert cumulative["misses"] == 0
    assert cumulative["patience_anchor_score"] == 1.11


def test_zero_anchor_accepts_any_strict_positive_gain(tmp_path):
    schedule = _schedule(tmp_path, 0, _config())
    schedule.observe({METRIC: 0.0}, 0)
    result = schedule.observe({METRIC: 0.0001}, 20)
    assert result["relative_gain"] == float("inf")
    assert result["patience_reset"] is True


def test_latch_disables_images_but_keeps_jsonl_complete(tmp_path):
    schedule = _schedule(tmp_path, 0, _config())
    assert schedule.should_capture_rollout_images() is True
    result = schedule.observe({METRIC: 0.1}, 0)
    assert result["latched"] is True
    assert schedule.should_capture_rollout_images() is False

    _write_rollout_step(schedule, 1, images=False)
    rollout_dir = _checkpoint_root(schedule).parent / "rollout_data"
    assert rollout_step_is_complete(rollout_dir, 1)
    assert not (rollout_dir / "image_1").exists()


def test_rollout_export_omits_image_directory_when_capture_is_off(
    tmp_path, monkeypatch
):
    rollout_dir = tmp_path / "rollout_data"
    staged = rollout_dir / ".staging" / "step_1" / "episode"
    staged.mkdir(parents=True)
    (staged / "record.json").write_text(
        json.dumps({"input": "prompt", "score": 0.0, "images": []}) + "\n",
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run=SimpleNamespace(extra={"legacy_rollout_dir": str(rollout_dir)})
    )
    monkeypatch.setattr(rollout_module, "_get_context", lambda _args: context)
    sample = SimpleNamespace(metadata={"graphrl_rollout_key": "episode"})

    rollout_module.finalize_rollout_step(SimpleNamespace(), 0, [sample])

    assert (rollout_dir / "1.jsonl").is_file()
    assert (rollout_dir / "1.complete").is_file()
    assert not (rollout_dir / "image_1").exists()


def test_resume_restores_controller_budget_patience_and_latch(tmp_path):
    schedule = _schedule(tmp_path, 0, _config(sft_patience=5, total_rl_steps=200))
    schedule.observe({METRIC: 0.0}, 0)
    schedule.observe({METRIC: 0.0}, 20)
    saved = schedule.store.load()
    _write_checkpoint(schedule, 20, state=saved)
    schedule.observe({METRIC: 0.2}, 40)

    restored = schedule.reconcile_resume(_checkpoint_root(schedule), 20)
    assert restored["total_rl_steps"] == 20
    assert restored["rounds"]["0"]["last_observed_step"] == 20
    assert restored["rounds"]["0"]["misses"] == 1
    assert restored["latched"] is False


def test_terminal_decision_becomes_ready_only_after_complete_checkpoint(tmp_path):
    schedule = _schedule(tmp_path, 0, _config(eval_every_steps=1, sft_patience=1))
    initial_model = tmp_path / "initial_model"
    _write_hf_model(initial_model)
    schedule.observe({METRIC: 0.0}, 0)
    schedule.commit(
        checkpoint_root=_checkpoint_root(schedule),
        rollout_id=-1,
        hf_model=initial_model,
        initial=True,
    )
    decision = schedule.observe({METRIC: 0.0}, 1)
    assert decision["decision"] == DECISION_SWITCH_TO_SFT
    assert schedule.store.load()["decision_ready"] is False

    checkpoint = _write_checkpoint(schedule, 1, state=schedule.store.load())
    schedule.commit(
        checkpoint_root=_checkpoint_root(schedule),
        rollout_id=0,
        hf_model=initial_model,
    )
    assert schedule.store.load()["decision_ready"] is True
    assert is_complete_checkpoint_manifest(checkpoint)

    (checkpoint / "data.pt").write_bytes(b"truncated")
    assert not is_complete_checkpoint_manifest(checkpoint)


def test_strict_peak_is_snapshotted_even_without_patience_reset(tmp_path):
    schedule = _schedule(
        tmp_path, 0, _config(min_delta=0.20, min_relative_delta=0.10)
    )
    initial_model = tmp_path / "initial_model"
    peak_model = tmp_path / "peak_model"
    _write_hf_model(initial_model)
    _write_hf_model(peak_model)

    schedule.observe({METRIC: 1.0}, 0)
    schedule.commit(
        checkpoint_root=_checkpoint_root(schedule),
        rollout_id=-1,
        hf_model=initial_model,
        initial=True,
    )
    peak = schedule.observe({METRIC: 1.05}, 20)
    assert peak["is_best"] is True
    assert peak["patience_reset"] is False
    _write_checkpoint(schedule, 20, state=schedule.store.load())
    schedule.commit(
        checkpoint_root=_checkpoint_root(schedule),
        rollout_id=19,
        hf_model=peak_model,
    )

    expected = tmp_path / "adaptive_best_snapshots/round_000_step_000020"
    assert is_complete_snapshot(expected)
    assert (tmp_path / "best_val_run").resolve() == expected.resolve()
    assert (tmp_path / "iter_000/rl/best_val").resolve() == expected.resolve()


def test_adaptive_wrapper_materializes_committed_round_best(tmp_path):
    schedule = _schedule(tmp_path, 0, _config(eval_every_steps=1, sft_patience=1))
    initial_model = tmp_path / "initial_model"
    _write_hf_model(initial_model)
    schedule.observe({METRIC: 0.0}, 0)
    schedule.commit(
        checkpoint_root=_checkpoint_root(schedule),
        rollout_id=-1,
        hf_model=initial_model,
        initial=True,
    )
    schedule.observe({METRIC: 0.0}, 1)
    _write_checkpoint(schedule, 1, state=schedule.store.load())
    schedule.commit(
        checkpoint_root=_checkpoint_root(schedule),
        rollout_id=0,
        hf_model=initial_model,
    )

    output = tmp_path / "iter_000" / "rl" / "rl_model"
    wrapper = AdaptiveSlimeWrapper(
        config={"_iter_num": 0, "training_steps": 10},
        input_paths={"model": "unused"},
        output_paths={
            "base_dir": str(tmp_path / "iter_000" / "rl"),
            "model": str(output),
        },
    )
    assert wrapper.is_already_complete()
    assert (output / "model.safetensors").is_file()
    assert (output.parent / ".slime_rl_done.json").is_file()


def test_checkpoint_manifest_covers_critic_and_dataset_state(tmp_path):
    schedule = _schedule(tmp_path, 0, _config())
    schedule.observe({METRIC: 0.0}, 0)
    checkpoint = _write_checkpoint(schedule, 1, state=schedule.store.load())
    assert is_complete_checkpoint_manifest(checkpoint)

    critic_state = _checkpoint_root(schedule) / "critic" / checkpoint.name / "data.pt"
    critic_state.write_bytes(b"damaged")
    assert not is_complete_checkpoint_manifest(checkpoint)


def test_config_change_is_rejected_on_resume(tmp_path):
    AdaptiveStateStore(tmp_path, _config()).initialize()
    try:
        AdaptiveStateStore(tmp_path, _config(min_delta=0.04)).load()
    except RuntimeError as exc:
        assert "parameters differ" in str(exc)
    else:
        raise AssertionError("config mismatch should fail closed")


def test_round_advances_only_after_finalize_phase(tmp_path):
    store = AdaptiveStateStore(tmp_path, _config())
    store.initialize()
    store.set_phase(0, PHASE_SFT)
    try:
        store.begin_next_round(0)
    except RuntimeError as exc:
        assert "finalize_round" in str(exc)
    else:
        raise AssertionError("round advanced before finalize phase")

    state = store.load()
    state["steps_by_round"] = {"0": 60}
    state["total_rl_steps"] = 60
    state["latched"] = True
    state["rounds"] = {"0": {"misses": 3, "last_observed_step": 60}}
    store.save(state)
    store.set_phase(0, PHASE_FINALIZE_ROUND)
    state = store.begin_next_round(0)
    assert state["round"] == 1
    assert state["phase"] == "rl"
    assert state["total_rl_steps"] == 60
    assert state["latched"] is True
