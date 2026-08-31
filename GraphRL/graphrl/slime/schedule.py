"""Checkpoint-coupled adaptive RL controller for the SLIME training driver."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any

from graphrl.adaptive.model import (
    is_complete_hf_model,
    write_checkpoint_manifest,
    write_snapshot_manifest,
)
from graphrl.adaptive.rollouts import durable_rollout_prefix, required_rollout_prefix
from graphrl.adaptive.state import (
    CONFIG_FILENAME,
    DECISION_CONTINUE_RL,
    DECISION_STOP_RUN,
    DECISION_SWITCH_TO_SFT,
    PHASE_RL,
    AdaptiveStateStore,
)


class SlimeAdaptiveSchedule:
    """Persist early-stop decisions independently of SLIME process lifetime."""

    def __init__(self, experiment_root: str | Path, round_index: int) -> None:
        self.experiment_root = Path(experiment_root).expanduser().resolve()
        self.round_index = int(round_index)
        config_path = self.experiment_root / CONFIG_FILENAME
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"missing or invalid adaptive config: {config_path}") from exc
        self.store = AdaptiveStateStore(self.experiment_root, raw)
        self.cfg = self.store.config
        self.store.initialize()

    def observe(self, metrics: dict[str, Any], step: int) -> dict[str, Any]:
        metric = self.cfg["metric"]
        score = metrics.get(metric)
        if not isinstance(score, Real):
            raise TypeError(f"adaptive metric {metric!r} is absent or non-numeric")
        score = float(score)
        step = int(step)
        state = self.store.load()
        if int(state["round"]) != self.round_index or state.get("phase") != PHASE_RL:
            raise RuntimeError(
                f"adaptive validation arrived for round {self.round_index}, but state is "
                f"round={state.get('round')} phase={state.get('phase')!r}"
            )
        rounds = dict(state.get("rounds") or {})
        key = str(self.round_index)
        current = dict(rounds.get(key) or {})
        previous_step = current.get("last_observed_step")
        if previous_step is not None and step < int(previous_step):
            raise RuntimeError(f"validation step moved backwards: {step} < {previous_step}")
        if previous_step is not None and step == int(previous_step):
            return {
                "decision": current.get("decision", DECISION_CONTINUE_RL),
                "new_round_best": False,
                "new_run_best": False,
                "is_best": False,
                "score": score,
                "step": step,
                "misses": int(current.get("misses", 0)),
                "patience_reset": bool(current.get("patience_reset", False)),
                "absolute_gain": float(current.get("last_absolute_gain", 0.0)),
                "relative_gain": float(current.get("last_relative_gain", 0.0)),
                "iteration_best_score": current.get("best_score"),
                "patience_anchor_score": current.get("patience_anchor_score"),
                "latched": bool(state.get("latched")),
            }

        best = current.get("best_score")
        new_round_best = best is None or self._better(score, float(best))
        if new_round_best:
            current["best_score"] = score
            current["best_step"] = step
            current["checkpoint"] = None

        anchor = current.get("patience_anchor_score")
        if anchor is None:
            reset, absolute, relative = True, 0.0, 0.0
        else:
            absolute = score - float(anchor) if self.cfg["direction"] == "maximize" else float(anchor) - score
            relative = float("inf") if absolute > 0 and abs(float(anchor)) <= 1e-12 else (
                absolute / abs(float(anchor)) if absolute > 0 else 0.0
            )
            reset = absolute > 0 and (
                absolute >= self.cfg["min_delta"]
                or relative >= self.cfg["min_relative_delta"]
            )
        if reset:
            current["patience_anchor_score"] = score
            current["patience_anchor_step"] = step
            current["misses"] = 0
        else:
            current["misses"] = int(current.get("misses", 0)) + 1
        current.update(
            {
                "last_observed_step": step,
                "last_score": score,
                "last_absolute_gain": absolute,
                "last_relative_gain": relative,
                "patience_reset": reset,
            }
        )

        run_best = state.get("run_best")
        new_run_best = run_best is None or self._better(score, float(run_best["score"]))
        if new_run_best:
            state["run_best"] = {
                "score": score,
                "round": self.round_index,
                "step": step,
                "checkpoint": None,
            }
        if self._threshold(float(state["run_best"]["score"])):
            state["latched"] = True

        steps = dict(state.get("steps_by_round") or {})
        steps[key] = max(int(steps.get(key, 0)), step)
        state["steps_by_round"] = steps
        state["total_rl_steps"] = sum(int(value) for value in steps.values())
        misses = int(current.get("misses", 0))
        if state["total_rl_steps"] >= self.cfg["total_rl_steps"]:
            decision = DECISION_STOP_RUN
            state["stop_reason"] = "total_rl_steps"
        elif state.get("latched") and misses >= self.cfg["finish_patience"]:
            decision = DECISION_STOP_RUN
            state["stop_reason"] = "finish_patience"
        elif not state.get("latched") and misses >= self.cfg["sft_patience"]:
            decision = DECISION_SWITCH_TO_SFT
        else:
            decision = DECISION_CONTINUE_RL

        current["decision"] = decision
        rounds[key] = current
        state["rounds"] = rounds
        state["decision"] = decision
        state["decision_round"] = self.round_index
        state["decision_step"] = step
        state["decision_ready"] = False
        self.store.save(state)
        return {
            "decision": decision,
            "new_round_best": new_round_best,
            "new_run_best": new_run_best,
            "is_best": new_round_best,
            "score": score,
            "step": step,
            "misses": misses,
            "patience_reset": reset,
            "absolute_gain": absolute,
            "relative_gain": relative,
            "iteration_best_score": current.get("best_score"),
            "patience_anchor_score": current.get("patience_anchor_score"),
            "latched": bool(state.get("latched")),
        }

    def should_capture_rollout_images(self) -> bool:
        """Images stop at the latch; JSONL and validation continue."""

        return not bool(self.store.load().get("latched"))

    def reconcile_resume(self, checkpoint_root: str | Path, start_rollout_id: int) -> dict[str, Any]:
        """Restore controller state from the exact SLIME checkpoint being resumed."""

        if int(start_rollout_id) <= 0:
            return self.store.load()
        rollout_id = int(start_rollout_id) - 1
        checkpoint = Path(checkpoint_root) / f"iter_{rollout_id:07d}"
        state_path = checkpoint / self.store.path.name
        if not state_path.is_file():
            raise RuntimeError(
                f"SLIME resumes rollout {rollout_id}, but its adaptive state is missing: "
                f"{state_path}"
            )
        restored = self.store.restore_from(checkpoint)
        required = required_rollout_prefix(restored, rollout_id + 1)
        if required:
            rollout_dir = Path(checkpoint_root).parent / "rollout_data"
            durable = durable_rollout_prefix(rollout_dir)
            if durable < required:
                raise RuntimeError(
                    f"checkpoint step {required} is ahead of durable rollout step {durable}"
                )
        return restored

    def commit(
        self,
        *,
        checkpoint_root: str | Path,
        rollout_id: int,
        hf_model: str | Path,
        initial: bool = False,
        related_paths: tuple[str | Path, ...] = (),
    ) -> dict[str, Any]:
        """Commit best-model material and make a terminal decision executable."""

        state = self.store.load()
        current = dict((state.get("rounds") or {}).get(str(self.round_index)) or {})
        step = int(current.get("last_observed_step", 0))
        best_step = int(current.get("best_step", -1))
        source = Path(hf_model)
        if best_step == step and current.get("checkpoint") is None:
            if not is_complete_hf_model(source):
                raise RuntimeError(f"best checkpoint source is incomplete: {source}")
            snapshot = (
                self.experiment_root
                / "adaptive_best_snapshots"
                / f"round_{self.round_index:03d}_step_{step:06d}"
            )
            relative = str(snapshot.relative_to(self.experiment_root))
            current["checkpoint"] = relative
            state["rounds"][str(self.round_index)] = current
            run_best = state.get("run_best") or {}
            is_run_best = (
                int(run_best.get("round", -1)) == self.round_index
                and int(run_best.get("step", -1)) == step
            )
            if is_run_best:
                run_best["checkpoint"] = relative
                state["run_best"] = run_best

            if not snapshot.exists():
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=f".{snapshot.name}.", dir=snapshot.parent))
                try:
                    target = staging / "actor" / "huggingface"
                    target.parent.mkdir(parents=True)
                    shutil.copytree(source, target)
                    (staging / "metadata.json").write_text(
                        json.dumps(
                            {
                                "metric": self.cfg["metric"],
                                "score": current["best_score"],
                                "round": self.round_index,
                                "step": step,
                                "initial": bool(initial),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    self.store._write_atomic(staging / self.store.path.name, state)
                    write_snapshot_manifest(staging)
                    os.replace(staging, snapshot)
                except Exception:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            self._replace_link(
                self.experiment_root / f"iter_{self.round_index:03d}" / "rl" / "best_val",
                snapshot,
            )
            if is_run_best:
                self._replace_link(self.experiment_root / "best_val_run", snapshot)

        terminal = state.get("decision") in {DECISION_SWITCH_TO_SFT, DECISION_STOP_RUN}
        if terminal:
            state["decision_ready"] = True
            state["decision_checkpoint_step"] = step
        if not initial:
            checkpoint = Path(checkpoint_root) / f"iter_{int(rollout_id):07d}"
            if not checkpoint.is_dir():
                raise RuntimeError(f"SLIME checkpoint directory is missing: {checkpoint}")
            # State first, manifest last. The manifest is the commit record.
            self.store.snapshot_to(checkpoint, state=state)
            write_checkpoint_manifest(checkpoint, related_paths)
        self.store.save(state)
        return state

    @staticmethod
    def _replace_link(link: Path, target: Path) -> None:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        os.symlink(os.path.relpath(target, link.parent), link)

    def decision(self) -> str:
        return str(self.store.load().get("decision", DECISION_CONTINUE_RL))

    def _better(self, score: float, reference: float) -> bool:
        return score > reference if self.cfg["direction"] == "maximize" else score < reference

    def _threshold(self, score: float) -> bool:
        threshold = float(self.cfg["latch_threshold"])
        return score >= threshold if self.cfg["direction"] == "maximize" else score <= threshold


__all__ = ["SlimeAdaptiveSchedule"]
