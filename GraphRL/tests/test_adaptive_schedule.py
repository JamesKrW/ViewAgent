from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from graphrl.adaptive.model import (
    is_complete_checkpoint_manifest,
    is_complete_snapshot,
)
from graphrl.adaptive.state import (
    CONFIG_FILENAME,
    DECISION_CONTINUE_RL,
    DECISION_SWITCH_TO_SFT,
    PHASE_FINALIZE_ROUND,
    PHASE_SFT,
    AdaptiveStateStore,
    normalize_adaptive_config,
)
from graphrl.vagen.adaptive_vagen_wrapper import AdaptiveVagenWrapper
from graphrl.adaptive.trainer import AdaptivePPOTrainer
from vagen.training.trainer.ppo_trainer import VagenPPOTrainer
from graphrl.adaptive.rollouts import (
    durable_rollout_prefix,
    rollout_step_is_complete,
)
from graphrl.adaptive.schedule import AdaptiveSchedule

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


def _schedule(tmp_path: Path, round_index: int, cfg: dict) -> AdaptiveSchedule:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / CONFIG_FILENAME).write_text(json.dumps(cfg), encoding="utf-8")
    local_dir = tmp_path / f"iter_{round_index:03d}" / "rl" / "verl_checkpoints"
    trainer_config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(local_dir))
    )
    return AdaptiveSchedule(trainer_config, str(tmp_path), round_index)


def _write_rollout_step(schedule: AdaptiveSchedule, step: int) -> None:
    rollout_dir = schedule.default_local_dir.parent / "rollout_data"
    image_dir = rollout_dir / f"image_{step}"
    image_dir.mkdir(parents=True)
    (rollout_dir / f"{step}.jsonl").write_text("{}\n", encoding="utf-8")
    (image_dir / "0.png").write_bytes(b"png")


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
    assert not baseline["is_run_best"]

    assert round1.observe({METRIC: 0.0}, 20)["misses"] == 1
    assert round1.observe({METRIC: 0.0}, 40)["misses"] == 2
    improved = round1.observe({METRIC: 0.05882353}, 60)
    assert improved["decision"] == DECISION_CONTINUE_RL
    assert improved["misses"] == 0
    assert improved["is_run_best"]


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

    state = schedule.store.load()
    round_state = state["rounds"]["0"]
    assert round_state["best_score"] == observations[-1][1]
    assert round_state["patience_anchor_score"] == observations[-1][1]


def test_absolute_or_relative_gain_resets_patience(tmp_path):
    relative = _schedule(tmp_path / "relative", 0, _config())
    relative.observe({METRIC: 0.02}, 0)
    relative_result = relative.observe({METRIC: 0.023}, 20)
    assert relative_result["absolute_gain"] < 0.03
    assert relative_result["relative_gain"] >= 0.10
    assert relative_result["patience_reset"] is True
    assert relative_result["misses"] == 0

    absolute = _schedule(
        tmp_path / "absolute",
        0,
        _config(min_delta=0.03, min_relative_delta=1.0),
    )
    absolute.observe({METRIC: 0.10}, 0)
    absolute_result = absolute.observe({METRIC: 0.131}, 20)
    assert absolute_result["absolute_gain"] >= 0.03
    assert absolute_result["relative_gain"] < 1.0
    assert absolute_result["patience_reset"] is True
    assert absolute_result["misses"] == 0


def test_small_peaks_accumulate_against_patience_anchor(tmp_path):
    schedule = _schedule(
        tmp_path,
        0,
        _config(min_delta=0.20, min_relative_delta=0.10),
    )
    schedule.observe({METRIC: 1.0}, 0)

    small_peak = schedule.observe({METRIC: 1.05}, 20)
    assert small_peak["is_best"] is True
    assert small_peak["patience_reset"] is False
    assert small_peak["misses"] == 1
    assert small_peak["iteration_best_score"] == 1.05
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
    assert result["misses"] == 0


def test_latch_disables_future_rollout_image_capture(tmp_path):
    schedule = _schedule(tmp_path, 0, _config())
    assert schedule.should_capture_rollout_images() is True

    result = schedule.observe({METRIC: 0.1}, 0)

    assert result["latched"] is True
    assert schedule.should_capture_rollout_images() is False


def test_adaptive_trainer_keeps_jsonl_but_disables_images_after_latch(
    monkeypatch,
):
    """After the latch, images stop and the JSONL keeps going.

    The backend separates the two: ``_fit_dump_data`` writes the JSONL through its
    own super() call and ``_vagen_dump_images`` writes the frames, so suppressing
    images is one override and the JSONL is untouched by construction. The older
    backend wrote both from ``_log_rollout_data``, which is why this used to assert
    on a private ``_log_image_enable`` being toggled around a single call.
    """
    dumped = []
    monkeypatch.setattr(
        VagenPPOTrainer,
        "_vagen_dump_images",
        lambda self, batch: dumped.append(batch),
    )
    trainer = object.__new__(AdaptivePPOTrainer)
    trainer._adaptive_capture_disabled_logged = False
    trainer._run_schedule = SimpleNamespace(
        should_capture_rollout_images=lambda: False
    )

    trainer._vagen_dump_images("batch-after-latch")
    assert dumped == [], "images must not reach the backend once latched"
    assert trainer._adaptive_capture_disabled_logged is True

    trainer._run_schedule = SimpleNamespace(
        should_capture_rollout_images=lambda: True
    )
    trainer._vagen_dump_images("batch-before-latch")
    assert dumped == ["batch-before-latch"]


def test_resume_restores_state_snapshot_from_loaded_checkpoint(tmp_path):
    cfg = _config(sft_patience=5)
    schedule = _schedule(tmp_path, 0, cfg)
    schedule.observe({METRIC: 0.0}, 0)
    schedule.observe({METRIC: 0.0}, 20)
    checkpoint = schedule.default_local_dir / "global_step_20"
    schedule.store.snapshot_to(checkpoint)
    schedule.observe({METRIC: 0.0}, 40)

    restored = schedule.reconcile_resume(20)
    round_state = restored["rounds"]["0"]
    assert round_state["last_observed_step"] == 20
    assert round_state["misses"] == 1


def test_resume_checkpoint_is_authoritative_even_at_same_step(tmp_path):
    cfg = _config(sft_patience=5)
    schedule = _schedule(tmp_path, 0, cfg)
    schedule.observe({METRIC: 0.0}, 0)
    schedule.observe({METRIC: 0.0}, 20)
    checkpoint = schedule.default_local_dir / "global_step_20"
    schedule.store.snapshot_to(checkpoint)
    schedule.store.update(
        lambda state: state["rounds"]["0"].update({"misses": 99})
    )

    restored = schedule.reconcile_resume(20)
    assert restored["rounds"]["0"]["misses"] == 1


def test_multi_round_resume_restores_budget_patience_latch_and_round(tmp_path):
    cfg = _config(total_rl_steps=200)
    store = AdaptiveStateStore(tmp_path, cfg)
    state = store.initialize()
    state.update(
        {
            "round": 2,
            "phase": "rl",
            "decision": DECISION_CONTINUE_RL,
            "decision_round": 2,
            "decision_step": 20,
            "decision_ready": False,
            "latched": True,
            "steps_by_round": {"0": 60, "1": 40, "2": 20},
            "total_rl_steps": 120,
            "run_best": {
                "score": 0.12,
                "round": 1,
                "step": 40,
                "checkpoint": "adaptive_best_snapshots/round_001_step_000040",
            },
            "rounds": {
                "0": {"misses": 3, "last_observed_step": 60},
                "1": {"misses": 3, "last_observed_step": 40},
                "2": {
                    "best_score": 0.11,
                    "best_step": 0,
                    "patience_anchor_score": 0.11,
                    "patience_anchor_step": 0,
                    "misses": 1,
                    "last_observed_step": 20,
                    "last_score": 0.10,
                    "decision": DECISION_CONTINUE_RL,
                },
            },
        }
    )
    store.save(state)

    schedule = _schedule(tmp_path, 2, cfg)
    checkpoint = schedule.default_local_dir / "global_step_20"
    schedule.store.snapshot_to(checkpoint)

    # Simulate a root state that advanced after the last durable checkpoint.
    schedule.store.update(
        lambda current: current.update(
            {
                "latched": False,
                "steps_by_round": {"0": 60, "1": 40, "2": 40},
                "total_rl_steps": 140,
                "rounds": {
                    **current["rounds"],
                    "2": {**current["rounds"]["2"], "misses": 2},
                },
            }
        )
    )

    restored = schedule.reconcile_resume(20)
    assert restored["round"] == 2
    assert restored["phase"] == "rl"
    assert restored["steps_by_round"] == {"0": 60, "1": 40, "2": 20}
    assert restored["total_rl_steps"] == 120
    assert restored["rounds"]["2"]["misses"] == 1
    assert restored["rounds"]["2"]["patience_anchor_score"] == 0.11
    assert restored["latched"] is True
    assert restored["run_best"]["score"] == 0.12


def test_unscheduled_resume_eval_does_not_consume_patience(tmp_path):
    cfg = _config(sft_patience=5)
    schedule = _schedule(tmp_path, 0, cfg)
    schedule.observe({METRIC: 0.0}, 0)
    schedule.observe({METRIC: 0.0}, 20)

    assert not schedule.should_observe_resume_validation(35)
    assert schedule.should_observe_resume_validation(40)

    budget_cfg = _config(total_rl_steps=35)
    budget_root = tmp_path / "budget"
    budget_root.mkdir()
    budget_schedule = _schedule(budget_root, 0, budget_cfg)
    budget_schedule.observe({METRIC: 0.0}, 0)
    budget_schedule.observe({METRIC: 0.0}, 20)
    assert budget_schedule.should_observe_resume_validation(35)


def test_terminal_decision_becomes_ready_only_after_checkpoint(tmp_path):
    cfg = _config(eval_every_steps=1, sft_patience=1)
    schedule = _schedule(tmp_path, 0, cfg)
    schedule.observe({METRIC: 0.0}, 0)
    decision = schedule.observe({METRIC: 0.0}, 1)
    assert decision["decision"] == DECISION_SWITCH_TO_SFT
    assert schedule.store.load()["decision_ready"] is False

    checkpoint = schedule.default_local_dir / "global_step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    _write_rollout_step(schedule, 1)
    (schedule.default_local_dir / "latest_checkpointed_iteration.txt").write_text(
        "1", encoding="utf-8"
    )
    schedule.commit_checkpoint_state(1)
    state = schedule.store.load()
    assert state["decision_ready"] is True
    assert (checkpoint / "adaptive_schedule_state.json").is_file()
    assert is_complete_checkpoint_manifest(checkpoint)

    (checkpoint / "data.pt").write_bytes(b"truncated")
    assert not is_complete_checkpoint_manifest(checkpoint)


def test_latched_checkpoint_does_not_require_rollout_payload(tmp_path):
    schedule = _schedule(
        tmp_path, 0, _config(eval_every_steps=1, sft_patience=5)
    )
    schedule.observe({METRIC: 0.0}, 0)
    result = schedule.observe({METRIC: 0.1}, 1)
    assert result["latched"] is True

    checkpoint = schedule.default_local_dir / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    (schedule.default_local_dir / "latest_checkpointed_iteration.txt").write_text(
        "1", encoding="utf-8"
    )

    schedule.commit_checkpoint_state(1)

    assert (checkpoint / "adaptive_schedule_state.json").is_file()
    assert is_complete_checkpoint_manifest(checkpoint)


class _FakeActor:
    def save_checkpoint(self, actor_dir, _remote, _step, max_ckpt_to_keep=None):
        hf_dir = Path(actor_dir) / "huggingface"
        hf_dir.mkdir(parents=True)
        (hf_dir / "config.json").write_text("{}", encoding="utf-8")
        (hf_dir / "model.safetensors").write_bytes(b"weights")
        (Path(actor_dir) / "optimizer.pt").write_bytes(b"discard")


def test_strict_peak_is_saved_even_without_patience_reset(tmp_path):
    schedule = _schedule(
        tmp_path,
        0,
        _config(min_delta=0.20, min_relative_delta=0.10),
    )
    first = schedule.observe({METRIC: 1.0}, 0)
    schedule.save_best_checkpoint(
        actor_rollout_wg=_FakeActor(), global_steps=0, score=first["score"]
    )

    peak = schedule.observe({METRIC: 1.05}, 20)
    assert peak["is_best"] is True
    assert peak["patience_reset"] is False
    schedule.save_best_checkpoint(
        actor_rollout_wg=_FakeActor(), global_steps=20, score=peak["score"]
    )

    expected = tmp_path / "adaptive_best_snapshots/round_000_step_000020"
    assert is_complete_snapshot(expected)
    assert (tmp_path / "best_val_run").resolve() == expected.resolve()
    assert (tmp_path / "iter_000/rl/best_val").resolve() == expected.resolve()


def test_best_checkpoint_and_terminal_rl_materialization(tmp_path):
    cfg = _config(eval_every_steps=1, sft_patience=1)
    schedule = _schedule(tmp_path, 0, cfg)
    first = schedule.observe({METRIC: 0.0}, 0)
    schedule.save_best_checkpoint(
        actor_rollout_wg=_FakeActor(), global_steps=0, score=first["score"]
    )
    assert (tmp_path / "best_val_run" / "actor" / "huggingface" / "config.json").is_file()
    assert (
        tmp_path / "iter_000" / "rl" / "best_val" / "actor" / "huggingface" / "config.json"
    ).is_file()

    schedule.observe({METRIC: 0.0}, 1)
    checkpoint = schedule.default_local_dir / "global_step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    _write_rollout_step(schedule, 1)
    (schedule.default_local_dir / "latest_checkpointed_iteration.txt").write_text(
        "1", encoding="utf-8"
    )
    schedule.commit_checkpoint_state(1)
    wrapper = AdaptiveVagenWrapper(
        config={"_iter_num": 0, "training_steps": 10, "vagen_dir": "."},
        input_paths={"model": "unused"},
        output_paths={
            "base_dir": str(tmp_path / "iter_000" / "rl"),
            "model": str(tmp_path / "iter_000" / "rl" / "rl_model"),
        },
    )
    assert wrapper.is_already_complete()
    model = tmp_path / "iter_000" / "rl" / "rl_model"
    assert model.is_dir() and not model.is_symlink()
    assert (model / "model.safetensors").is_file()


def test_config_change_is_rejected_on_resume(tmp_path):
    cfg = _config()
    AdaptiveStateStore(tmp_path, cfg).initialize()
    changed = _config(min_delta=0.04)
    try:
        AdaptiveStateStore(tmp_path, changed).load()
    except RuntimeError as exc:
        assert "parameters differ" in str(exc)
    else:
        raise AssertionError("config mismatch should fail closed")


def test_round_advances_only_after_finalize_phase(tmp_path):
    cfg = _config()
    store = AdaptiveStateStore(tmp_path, cfg)
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
    assert state["steps_by_round"] == {"0": 60}
    assert state["latched"] is True
    assert state["rounds"]["0"]["misses"] == 3
    assert "1" not in state["rounds"]


def test_tracker_repairs_missing_checkpoint_state_from_atomic_root(tmp_path):
    cfg = _config(sft_patience=5)
    schedule = _schedule(tmp_path, 0, cfg)
    schedule.observe({METRIC: 0.0}, 0)
    schedule.observe({METRIC: 0.0}, 20)
    checkpoint = schedule.default_local_dir / "global_step_20"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    (schedule.default_local_dir / "latest_checkpointed_iteration.txt").write_text(
        "20", encoding="utf-8"
    )

    trainer = object.__new__(AdaptivePPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(schedule.default_local_dir))
    )
    trainer._run_schedule = schedule
    trainer._repair_adaptive_checkpoint_tracker()

    restored = json.loads(
        (checkpoint / "adaptive_schedule_state.json").read_text(encoding="utf-8")
    )
    assert restored["rounds"]["0"]["last_observed_step"] == 20


def test_tracker_rejects_unlatched_checkpoint_ahead_of_rollouts(tmp_path):
    schedule = _schedule(tmp_path, 0, _config(sft_patience=5))
    schedule.observe({METRIC: 0.0}, 0)
    schedule.observe({METRIC: 0.0}, 20)
    checkpoint = schedule.default_local_dir / "global_step_20"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    schedule.store.snapshot_to(checkpoint)
    tracker = schedule.default_local_dir / "latest_checkpointed_iteration.txt"
    tracker.write_text("20", encoding="utf-8")

    trainer = object.__new__(AdaptivePPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(schedule.default_local_dir))
    )
    trainer._run_schedule = schedule
    trainer._repair_adaptive_checkpoint_tracker()

    assert checkpoint.is_dir()
    assert not tracker.exists()


def test_tracker_accepts_latched_checkpoint_without_rollout_images(tmp_path):
    schedule = _schedule(tmp_path, 0, _config(sft_patience=5))
    schedule.observe({METRIC: 0.0}, 0)
    schedule.observe({METRIC: 0.1}, 20)
    checkpoint = schedule.default_local_dir / "global_step_20"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    schedule.store.snapshot_to(checkpoint)
    tracker = schedule.default_local_dir / "latest_checkpointed_iteration.txt"
    tracker.write_text("1", encoding="utf-8")

    trainer = object.__new__(AdaptivePPOTrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(schedule.default_local_dir))
    )
    trainer._run_schedule = schedule
    trainer._repair_adaptive_checkpoint_tracker()

    assert tracker.read_text(encoding="utf-8") == "20"



def test_rollout_step_is_complete_on_jsonl_alone(tmp_path):
    """verl writes no marker and no frames; the JSONL has to be the signal.

    Requiring image_<step>/ kept durable_rollout_prefix at 0 and killed every run
    at its first checkpoint.
    """
    (tmp_path / "1.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    assert rollout_step_is_complete(tmp_path, 1)
    assert durable_rollout_prefix(tmp_path) == 1


def test_empty_jsonl_is_not_complete(tmp_path):
    (tmp_path / "1.jsonl").write_text("", encoding="utf-8")
    assert not rollout_step_is_complete(tmp_path, 1)


def test_marker_still_overrides_where_one_exists(tmp_path):
    """A SLIME-era directory keeps the stronger guarantee."""
    (tmp_path / "1.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (tmp_path / "1.complete").write_text("partial", encoding="utf-8")
    assert not rollout_step_is_complete(tmp_path, 1)
    (tmp_path / "1.complete").write_text("complete", encoding="utf-8")
    assert rollout_step_is_complete(tmp_path, 1)


class _RecordingSchedule:
    """Counts observations; everything else is the minimum _validate touches."""

    def __init__(self):
        self.observed_steps = []

    def reconcile_resume(self, step):
        pass

    def should_observe_resume_validation(self, step):
        return True

    def observe(self, val_metrics, global_steps):
        self.observed_steps.append(int(global_steps))
        return {}


def test_every_scheduled_validation_is_observed():
    """Observing only the first validation makes the whole controller inert.

    Patience never accumulates, no round-best is recorded past step 0, and
    neither the SFT switch nor the early stop can ever fire -- the run simply
    trains to the step budget with no sign anything is wrong.
    """
    trainer = object.__new__(AdaptivePPOTrainer)
    trainer._run_schedule = _RecordingSchedule()
    trainer._adaptive_first_validation = True

    # Bypass VagenPPOTrainer._validate, which needs a live worker group.
    import graphrl.adaptive.trainer as mod

    original = mod.VagenPPOTrainer._validate
    mod.VagenPPOTrainer._validate = lambda self: {}
    try:
        for step in (0, 20, 40, 60):
            trainer.global_steps = step
            AdaptivePPOTrainer._validate(trainer)
    finally:
        mod.VagenPPOTrainer._validate = original

    assert trainer._run_schedule.observed_steps == [0, 20, 40, 60]
