"""RayPPOTrainer variant used only by the adaptive GraphRL entry point."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from graphrl.adaptive.model import (
    CHECKPOINT_MANIFEST,
    is_complete_checkpoint_manifest,
)
from graphrl.adaptive.rollouts import (
    durable_rollout_prefix,
    required_rollout_prefix,
)
from graphrl.adaptive.state import STATE_FILENAME

from vagen.ray_trainer import RayPPOTrainer
from vagen.utils.adaptive_schedule import AdaptiveSchedule


class AdaptiveRayPPOTrainer(RayPPOTrainer):
    """Adds step-0 scheduling and checkpoint-coupled adaptive state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        experiment_root = os.environ.get("GRAPHRL_ADAPTIVE_EXPERIMENT_DIR")
        round_index = os.environ.get("GRAPHRL_ADAPTIVE_ROUND")
        if not experiment_root or round_index is None:
            raise RuntimeError(
                "AdaptiveRayPPOTrainer requires GRAPHRL_ADAPTIVE_EXPERIMENT_DIR "
                "and GRAPHRL_ADAPTIVE_ROUND"
            )
        # RayPPOTrainer's normal RunSchedule remains untouched for legacy jobs;
        # this subclass replaces it only in the separate adaptive entry point.
        self._run_schedule = AdaptiveSchedule(
            self.config, experiment_root, int(round_index)
        )
        self._adaptive_first_validation = True
        self._adaptive_capture_disabled_logged = False

    def _load_checkpoint(self):
        self._repair_adaptive_checkpoint_tracker()
        return super()._load_checkpoint()

    def _repair_adaptive_checkpoint_tracker(self) -> None:
        """Point verl only at checkpoints carrying the matching controller state."""
        checkpoint_root = Path(self.config.trainer.default_local_dir).resolve()
        tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
        tracked_step = None
        try:
            tracked_step = int(tracker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass

        if tracked_step is not None:
            tracked_dir = checkpoint_root / f"global_step_{tracked_step}"
            state_path = tracked_dir / STATE_FILENAME
            if (
                tracked_dir.is_dir()
                and (tracked_dir / "data.pt").is_file()
                and not state_path.is_file()
            ):
                root_state = self._run_schedule.store.load()
                round_state = dict(
                    (root_state.get("rounds") or {}).get(str(self._run_schedule.iteration))
                    or {}
                )
                if (
                    int(round_state.get("last_observed_step", -1)) == tracked_step
                    and int(root_state.get("decision_step", -1)) == tracked_step
                ):
                    self._run_schedule.store.snapshot_to(tracked_dir, state=root_state)
                    print(
                        f"[AdaptiveSchedule] repaired checkpoint state at step "
                        f"{tracked_step} from atomic root state"
                    )

        valid_steps = []
        rollout_dir = checkpoint_root.parent / "rollout_data"
        durable_prefix = None
        for checkpoint in checkpoint_root.glob("global_step_*"):
            try:
                step = int(checkpoint.name.rsplit("_", 1)[-1])
                state = json.loads(
                    (checkpoint / STATE_FILENAME).read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            checkpoint_is_complete = (
                (checkpoint / "data.pt").is_file()
                and (checkpoint / "actor").is_dir()
                and state.get("config_fingerprint")
                == self._run_schedule.store.config_fingerprint
                and (
                    not (checkpoint / CHECKPOINT_MANIFEST).exists()
                    or is_complete_checkpoint_manifest(checkpoint)
                )
            )
            required_prefix = required_rollout_prefix(state, step)
            if checkpoint_is_complete and required_prefix:
                if durable_prefix is None:
                    durable_prefix = durable_rollout_prefix(rollout_dir)
                checkpoint_is_complete = durable_prefix >= required_prefix
            if checkpoint_is_complete:
                valid_steps.append(step)

        if valid_steps:
            latest = max(valid_steps)
            if tracked_step != latest:
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(
                    prefix=".latest_checkpointed_iteration.", dir=str(checkpoint_root)
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write(str(latest))
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(tmp_name, tracker)
                except Exception:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                    raise
                print(
                    f"[AdaptiveSchedule] checkpoint tracker repaired -> step {latest}"
                )
        elif tracker.exists():
            tracker.unlink()
            print(
                "[AdaptiveSchedule] removed tracker with no self-contained "
                "adaptive checkpoint; restarting this round from its input model"
            )

    def _validate(self):
        metrics = super()._validate()
        if self._adaptive_first_validation:
            self._adaptive_first_validation = False
            self._run_schedule.reconcile_resume(self.global_steps)
            if not self._run_schedule.should_observe_resume_validation(
                self.global_steps
            ):
                print(
                    f"[AdaptiveSchedule] resume-only validation at unscheduled "
                    f"step {self.global_steps}; patience state unchanged"
                )
                return metrics
            decision = self._run_schedule.observe(
                val_metrics=metrics, global_steps=self.global_steps
            )
            if decision.get("is_best"):
                self._run_schedule.save_best_checkpoint(
                    actor_rollout_wg=self.actor_rollout_wg,
                    global_steps=self.global_steps,
                    score=decision["score"],
                )
            if decision.get("stop_run") or decision.get("switch_to_sft"):
                self._schedule_stop = True
        return metrics

    def _log_rollout_data(self, *args, **kwargs):
        """Keep rollout JSONL, but stop materializing images after the latch."""
        if self._run_schedule.should_capture_rollout_images():
            return super()._log_rollout_data(*args, **kwargs)

        if not self._adaptive_capture_disabled_logged:
            print(
                "[AdaptiveSchedule] rollout image capture disabled after "
                "threshold latch; JSONL logging remains enabled"
            )
            self._adaptive_capture_disabled_logged = True

        image_logging_was_enabled = self._log_image_enable
        self._log_image_enable = False
        try:
            return super()._log_rollout_data(*args, **kwargs)
        finally:
            self._log_image_enable = image_logging_was_enabled

    def _save_checkpoint(self):
        # Drain payloads scheduled before the latest validation.  Pre-latch
        # checkpoints require them for a possible TrajToSFT phase; a latch-step
        # checkpoint may have one final in-flight dump because rollout logging
        # happens immediately before validation.
        self._flush_image_dumps()
        super()._save_checkpoint()
        snapshot = self._run_schedule.commit_checkpoint_state(self.global_steps)
        print(f"[AdaptiveSchedule] checkpoint state -> {snapshot}")
