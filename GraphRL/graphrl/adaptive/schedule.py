"""Metric-driven RL/SFT schedule.

Moved here from the backend fork (``vagen/utils/adaptive_schedule.py``). Nothing
about it was ever VAGEN's: every import but one already pointed back at
``graphrl.adaptive``, and that one is now inlined below. Living inside the
submodule meant a backend swap took the schedule with it.

Used by ``graphrl.main_adaptive`` only.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any

from graphrl.adaptive.model import (
    is_complete_snapshot,
    write_checkpoint_manifest,
    write_snapshot_manifest,
)
from graphrl.adaptive.rollouts import (
    durable_rollout_prefix,
    required_rollout_prefix,
)
from graphrl.adaptive.state import (
    CONFIG_FILENAME,
    DECISION_CONTINUE_RL,
    DECISION_STOP_RUN,
    DECISION_SWITCH_TO_SFT,
    PHASE_RL,
    AdaptiveStateStore,
    normalize_adaptive_config,
)



def _prune_to_huggingface(actor_local_path: str) -> None:
    """Delete everything under ``actor_local_path`` except ``huggingface/``.

    Inlined from the backend rather than imported. It was the one thing this
    module took from VAGEN, it is sixteen lines of stdlib, and depending on a
    private helper inside a submodule is what made this file live there in the
    first place.
    """
    hf_subdir = os.path.join(actor_local_path, "huggingface")
    if not os.path.isdir(actor_local_path):
        return
    for entry in os.listdir(actor_local_path):
        path = os.path.join(actor_local_path, entry)
        if path == hf_subdir:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


class AdaptiveSchedule:
    """Persistent adaptive state machine used by the adaptive trainer."""

    enabled = True

    def __init__(self, config, experiment_root: str, iteration: int):
        self.experiment_root = Path(experiment_root).expanduser().resolve()
        self.iteration = int(iteration)
        config_path = self.experiment_root / CONFIG_FILENAME
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"missing or invalid adaptive config: {config_path}") from exc

        self.cfg = normalize_adaptive_config(raw)
        self.metric = self.cfg["metric"]
        self.direction = self.cfg["direction"]
        self.min_delta = self.cfg["min_delta"]
        self.min_relative_delta = self.cfg["min_relative_delta"]
        self.sft_patience = self.cfg["sft_patience"]
        self.finish_patience = self.cfg["finish_patience"]
        self.latch_threshold = self.cfg["latch_threshold"]
        self.total_budget = self.cfg["total_rl_steps"]
        self.default_local_dir = Path(config.trainer.default_local_dir).resolve()
        self.store = AdaptiveStateStore(self.experiment_root, self.cfg)
        self.store.initialize()

    def reconcile_resume(self, global_steps: int) -> dict[str, Any]:
        """Align root state with the checkpoint actually restored by VAGEN."""
        state = self.store.load()
        if int(state["round"]) != self.iteration:
            raise RuntimeError(
                f"adaptive trainer round {self.iteration} does not match state "
                f"round {state['round']}"
            )
        round_state = dict((state.get("rounds") or {}).get(str(self.iteration)) or {})
        recorded_step = round_state.get("last_observed_step")
        if int(global_steps) > 0:
            checkpoint_dir = self.default_local_dir / f"global_step_{int(global_steps)}"
            print(
                f"[AdaptiveSchedule] restoring authoritative state from "
                f"checkpoint step {global_steps}"
            )
            return self.store.restore_from(checkpoint_dir)

        if recorded_step is None or int(recorded_step) == 0:
            return state

        checkpoint_dir = self._step_zero_snapshot(state)
        print(
            f"[AdaptiveSchedule] reconciling state from restored checkpoint "
            f"step {global_steps}"
        )
        return self.store.restore_from(checkpoint_dir)

    def should_observe_resume_validation(self, global_steps: int) -> bool:
        """Avoid charging patience for an unscheduled eval caused only by resume."""
        step = int(global_steps)
        state = self.store.load()
        round_state = dict((state.get("rounds") or {}).get(str(self.iteration)) or {})
        last_observed = round_state.get("last_observed_step")
        if last_observed is None or step <= int(last_observed):
            return True

        prior_steps = sum(
            int(value)
            for key, value in (state.get("steps_by_round") or {}).items()
            if int(key) != self.iteration
        )
        is_scheduled = step % int(self.cfg["eval_every_steps"]) == 0
        exhausts_budget = prior_steps + step >= self.total_budget
        return is_scheduled or exhausts_budget

    def should_capture_rollout_images(self) -> bool:
        """Whether new rollout images can still be consumed by future SFT."""
        state = self.store.load()
        return (
            int(state.get("round", -1)) == self.iteration
            and state.get("phase") == PHASE_RL
            and not bool(state.get("latched"))
        )

    def observe(self, val_metrics: dict, global_steps: int) -> dict[str, Any]:
        """Record one validation and return the explicit controller decision."""
        score = val_metrics.get(self.metric)
        if not isinstance(score, Real):
            raise TypeError(
                f"adaptive metric {self.metric!r} is absent or non-numeric"
            )
        score = float(score)
        step = int(global_steps)

        state = self.store.load()
        if int(state["round"]) != self.iteration or state.get("phase") != PHASE_RL:
            raise RuntimeError(
                f"adaptive validation arrived in state round={state.get('round')} "
                f"phase={state.get('phase')!r}; expected round={self.iteration}, phase='rl'"
            )

        rounds = dict(state.get("rounds") or {})
        key = str(self.iteration)
        round_state = dict(rounds.get(key) or {})
        last_step = round_state.get("last_observed_step")

        if last_step is not None and step < int(last_step):
            raise RuntimeError(
                f"adaptive validation step moved backwards ({step} < {last_step}); "
                "resume state was not reconciled"
            )
        if last_step is not None and step == int(last_step):
            run_best = state.get("run_best") or {}
            same_run_best = (
                int(run_best.get("round", -1)) == self.iteration
                and int(run_best.get("step", -1)) == step
            )
            needs_checkpoint_repair = (
                not is_complete_snapshot(self._iteration_best_snapshot())
                or (same_run_best and not is_complete_snapshot(self._global_best_snapshot()))
            )
            return self._decision_result(
                state,
                round_state,
                score,
                is_iteration_best=needs_checkpoint_repair,
                is_run_best=same_run_best,
            )

        # Checkpoint selection and patience use deliberately different
        # references.  Any strict peak should be recoverable as the best
        # model, while patience only resets after a meaningful absolute OR
        # relative gain from its last reset anchor.
        iteration_best = round_state.get("best_score")
        is_iteration_best = iteration_best is None or self._strictly_better(
            score, float(iteration_best)
        )
        if is_iteration_best:
            round_state["best_score"] = score
            round_state["best_step"] = step
            round_state["checkpoint"] = self._snapshot_relpath(step)
            print(
                f"[AdaptiveSchedule] new round-best {score:.4f} "
                f"(round {self.iteration}, step {step})"
            )

        patience_anchor = round_state.get("patience_anchor_score")
        if patience_anchor is None:
            patience_reset = True
            absolute_gain = 0.0
            relative_gain = 0.0
            round_state["patience_anchor_score"] = score
            round_state["patience_anchor_step"] = step
            round_state["misses"] = 0
        else:
            patience_reset, absolute_gain, relative_gain = self._patience_improved(
                score, float(patience_anchor)
            )
            if patience_reset:
                round_state["patience_anchor_score"] = score
                round_state["patience_anchor_step"] = step
                round_state["misses"] = 0
                print(
                    "[AdaptiveSchedule] patience reset "
                    f"(absolute_gain={absolute_gain:.6f}, "
                    f"relative_gain={relative_gain:.2%}, "
                    f"round {self.iteration}, step {step})"
                )
            else:
                round_state["misses"] = int(round_state.get("misses", 0)) + 1
        round_state["last_absolute_gain"] = absolute_gain
        round_state["last_relative_gain"] = relative_gain
        round_state["patience_reset"] = patience_reset

        run_best = state.get("run_best")
        run_best_score = None if run_best is None else float(run_best["score"])
        is_run_best = run_best_score is None or self._strictly_better(
            score, run_best_score
        )
        if is_run_best:
            state["run_best"] = {
                "score": score,
                "round": self.iteration,
                "step": step,
                "checkpoint": self._snapshot_relpath(step),
            }
            print(
                f"[AdaptiveSchedule] new run-best {score:.4f} "
                f"(round {self.iteration}, step {step})"
            )

        if self._threshold_reached(float(state["run_best"]["score"])):
            if not state.get("latched"):
                print(
                    "[AdaptiveSchedule] threshold latched; SFT and rollout image "
                    "capture are disabled"
                )
            state["latched"] = True

        steps_by_round = dict(state.get("steps_by_round") or {})
        steps_by_round[key] = max(int(steps_by_round.get(key, 0)), step)
        state["steps_by_round"] = steps_by_round
        state["total_rl_steps"] = sum(int(value) for value in steps_by_round.values())

        misses = int(round_state.get("misses", 0))
        if state["total_rl_steps"] >= self.total_budget:
            decision = DECISION_STOP_RUN
            state["stop_reason"] = "total_rl_steps"
            print(
                f"[AdaptiveSchedule] total RL budget reached "
                f"({state['total_rl_steps']}/{self.total_budget})"
            )
        elif state.get("latched") and misses >= self.finish_patience:
            decision = DECISION_STOP_RUN
            state["stop_reason"] = "finish_patience"
            print(
                f"[AdaptiveSchedule] finish_patience {self.finish_patience} reached"
            )
        elif not state.get("latched") and misses >= self.sft_patience:
            decision = DECISION_SWITCH_TO_SFT
            print(
                f"[AdaptiveSchedule] sft_patience {self.sft_patience} reached -> SFT"
            )
        else:
            decision = DECISION_CONTINUE_RL

        round_state["last_observed_step"] = step
        round_state["last_score"] = score
        round_state["decision"] = decision
        rounds[key] = round_state
        state["rounds"] = rounds
        state["decision"] = decision
        state["decision_round"] = self.iteration
        state["decision_step"] = step
        state["decision_ready"] = False
        self.store.save(state)

        return self._decision_result(
            state,
            round_state,
            score,
            is_iteration_best=is_iteration_best,
            is_run_best=is_run_best,
        )

    def save_best_checkpoint(
        self, *, actor_rollout_wg, global_steps: int, score: float
    ) -> None:
        """Save one immutable round-best snapshot and update stable symlinks."""
        step = int(global_steps)
        state = self.store.load()
        round_state = dict((state.get("rounds") or {}).get(str(self.iteration)) or {})
        if int(round_state.get("best_step", -1)) != step:
            return

        relative_snapshot = self._snapshot_relpath(step)
        snapshot = self.experiment_root / relative_snapshot
        if not is_complete_snapshot(snapshot):
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{snapshot.name}.", dir=str(snapshot.parent)
                )
            )
            try:
                actor_dir = staging / "actor"
                actor_rollout_wg.save_checkpoint(
                    str(actor_dir), None, step, max_ckpt_to_keep=None
                )
                _prune_to_huggingface(str(actor_dir))
                (staging / "metadata.json").write_text(
                    json.dumps(
                        {
                            "metric": self.metric,
                            "score": float(score),
                            "round": self.iteration,
                            "step": step,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (staging / "adaptive_schedule_state.json").write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                write_snapshot_manifest(staging)
                if snapshot.exists():
                    shutil.rmtree(snapshot)
                os.replace(staging, snapshot)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise

        self._replace_symlink(
            self.default_local_dir.parent / "best_val", snapshot
        )
        run_best = state.get("run_best") or {}
        if (
            int(run_best.get("round", -1)) == self.iteration
            and int(run_best.get("step", -1)) == step
        ):
            self._replace_symlink(self.experiment_root / "best_val_run", snapshot)
        self._garbage_collect_snapshots()
        print(f"[AdaptiveSchedule] best checkpoint -> {snapshot}")

    def snapshot_checkpoint_state(self, global_steps: int) -> Path:
        checkpoint_dir = self.default_local_dir / f"global_step_{int(global_steps)}"
        return self.store.snapshot_to(checkpoint_dir)

    def commit_checkpoint_state(self, global_steps: int) -> Path:
        """Attach state to a durable checkpoint, then expose terminal decisions."""
        step = int(global_steps)
        checkpoint_dir = self.default_local_dir / f"global_step_{step}"
        tracker = self.default_local_dir / "latest_checkpointed_iteration.txt"
        try:
            tracked_step = int(tracker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"regular checkpoint tracker is not durable for step {step}: {tracker}"
            ) from exc
        if tracked_step != step or not (checkpoint_dir / "data.pt").is_file():
            raise RuntimeError(
                f"regular checkpoint is incomplete for step {step}: {checkpoint_dir}"
            )
        state = self.store.load()
        required_rollout_step = required_rollout_prefix(state, step)
        if required_rollout_step:
            rollout_dir = self.default_local_dir.parent / "rollout_data"
            rollout_step = durable_rollout_prefix(rollout_dir)
            if rollout_step < required_rollout_step:
                raise RuntimeError(
                    "regular checkpoint cannot commit before its rollout payload: "
                    f"checkpoint_step={step}, "
                    f"required_rollout_step={required_rollout_step}, "
                    f"durable_rollout_step={rollout_step}"
                )

        terminal = (
            int(state.get("decision_round", -1)) == self.iteration
            and int(state.get("decision_step", -1)) == step
            and state.get("decision")
            in {
                DECISION_SWITCH_TO_SFT,
                DECISION_STOP_RUN,
            }
        )
        if terminal:
            state["decision_ready"] = True
            state["decision_checkpoint_step"] = step
        # The checkpoint copy is written before the root state can make a
        # terminal decision executable. A crash at either boundary therefore
        # resumes from a self-contained regular checkpoint.
        target = self.store.snapshot_to(checkpoint_dir, state=state)
        write_checkpoint_manifest(checkpoint_dir)
        if terminal:
            self.store.save(state)
        return target

    def _strictly_better(self, score: float, best: float) -> bool:
        if self.direction == "maximize":
            return score > best
        return score < best

    def _patience_improved(
        self, score: float, anchor: float
    ) -> tuple[bool, float, float]:
        if self.direction == "maximize":
            absolute_gain = score - anchor
        else:
            absolute_gain = anchor - score
        if absolute_gain <= 0:
            return False, absolute_gain, 0.0

        denominator = abs(anchor)
        relative_gain = (
            float("inf")
            if denominator <= 1e-12
            else absolute_gain / denominator
        )
        reset = (
            absolute_gain >= self.min_delta
            or relative_gain >= self.min_relative_delta
        )
        return reset, absolute_gain, relative_gain

    def _threshold_reached(self, score: float) -> bool:
        if self.direction == "maximize":
            return score >= self.latch_threshold
        return score <= self.latch_threshold

    def _snapshot_relpath(self, step: int) -> str:
        return f"adaptive_best_snapshots/round_{self.iteration:03d}_step_{step:06d}"

    def _iteration_best_snapshot(self) -> Path:
        return self.default_local_dir.parent / "best_val"

    def _global_best_snapshot(self) -> Path:
        return self.experiment_root / "best_val_run"

    def _step_zero_snapshot(self, state: dict[str, Any]) -> Path:
        round_state = dict((state.get("rounds") or {}).get(str(self.iteration)) or {})
        relative = round_state.get("checkpoint")
        if int(round_state.get("best_step", -1)) != 0 or not relative:
            raise RuntimeError(
                "adaptive state is ahead of a scratch model load and has no "
                "step-0 state snapshot"
            )
        snapshot = self.experiment_root / relative
        state_snapshot = snapshot / "adaptive_schedule_state.json"
        if not state_snapshot.is_file():
            raise RuntimeError(f"step-0 adaptive state snapshot is missing: {state_snapshot}")
        # AdaptiveStateStore.restore_from expects the state file inside a dir.
        return snapshot

    @staticmethod
    def _replace_symlink(link: Path, target: Path) -> None:
        link.parent.mkdir(parents=True, exist_ok=True)
        tmp = link.with_name(f".{link.name}.tmp")
        if tmp.exists() or tmp.is_symlink():
            if tmp.is_dir() and not tmp.is_symlink():
                shutil.rmtree(tmp)
            else:
                tmp.unlink()
        os.symlink(os.path.relpath(target, link.parent), tmp)
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        os.replace(tmp, link)

    def _garbage_collect_snapshots(self) -> None:
        root = self.experiment_root / "adaptive_best_snapshots"
        if not root.is_dir():
            return
        live_targets = set()
        links = [
            self.experiment_root / "best_val_run",
            *self.experiment_root.glob("iter_*/rl/best_val"),
        ]
        for link in links:
            if link.is_symlink():
                try:
                    live_targets.add(link.resolve())
                except OSError:
                    pass
        for candidate in root.iterdir():
            if candidate.is_dir() and candidate.resolve() not in live_targets:
                shutil.rmtree(candidate, ignore_errors=True)

    @staticmethod
    def _decision_result(
        state: dict[str, Any],
        round_state: dict[str, Any],
        score: float,
        *,
        is_iteration_best: bool,
        is_run_best: bool,
    ) -> dict[str, Any]:
        decision = round_state.get("decision", state.get("decision"))
        return {
            # RayPPOTrainer uses is_best to decide whether to call
            # save_best_checkpoint; in adaptive mode this means round-best.
            "is_best": bool(is_iteration_best),
            "is_run_best": bool(is_run_best),
            "score": score,
            "switch_to_sft": decision == DECISION_SWITCH_TO_SFT,
            "stop_run": decision == DECISION_STOP_RUN,
            "decision": decision,
            "latched": bool(state.get("latched")),
            "best_score": (state.get("run_best") or {}).get("score"),
            "best_iteration": (state.get("run_best") or {}).get("round"),
            "best_step": (state.get("run_best") or {}).get("step"),
            "steps_spent": int(state.get("total_rl_steps", 0)),
            "iteration_best_score": round_state.get("best_score"),
            "iteration_best_step": round_state.get("best_step"),
            "patience_anchor_score": round_state.get("patience_anchor_score"),
            "patience_anchor_step": round_state.get("patience_anchor_step"),
            "absolute_gain": round_state.get("last_absolute_gain"),
            "relative_gain": round_state.get("last_relative_gain"),
            "patience_reset": bool(round_state.get("patience_reset", False)),
            "misses": int(round_state.get("misses", 0)),
        }
