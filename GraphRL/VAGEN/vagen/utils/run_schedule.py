"""Run-level RL schedule: when RL stops, whether SFT runs, and the run's best score.

RL continues while it improves; SFT runs when it stops. Once the run's best passes
``best_metric_threshold`` the schedule latches: no further SFT, and ``run_patience``
ends the run instead. ``total_accumulated_rl_step`` caps the run.

State is a JSON file under the experiment root: each iteration is a separate process.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict

_STATE_FILENAME = "rl_schedule_state.json"

_DEFAULTS = {
    "enabled": False,
    "metric": "val-aux/ae/traj_success/mean@1",
    "min_delta": 0.03,
    "iter_patience": 3,
    "run_patience": 8,
    "best_metric_threshold": 0.1,
    "total_accumulated_rl_step": 801,
}


class RunSchedule:
    """Construct per trainer; call :meth:`observe` after each in-loop validation."""

    def __init__(self, config, experiment_root: str, iteration: int):
        cfg = dict(_DEFAULTS)
        raw = config.trainer.get("rl_schedule", None)
        if raw is None:
            # The block also exists at the pipeline top level, where the controller
            # reads it. That copy never reaches verl's config tree, so a schedule
            # configured only there would silently do nothing.
            print("[RunSchedule] no trainer.rl_schedule; schedule disabled")
        cfg.update(dict(raw or {}))
        self.cfg = cfg
        self.enabled = bool(cfg["enabled"])
        self.metric = str(cfg["metric"])
        self.min_delta = float(cfg["min_delta"])
        self.iter_patience = int(cfg["iter_patience"])
        self.run_patience = int(cfg["run_patience"])
        self.best_metric_threshold = float(cfg["best_metric_threshold"])
        self.total_budget = int(cfg["total_accumulated_rl_step"])
        self.iteration = int(iteration)
        self.state_path = os.path.join(experiment_root, _STATE_FILENAME)
        self._misses = 0

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"best_score": None, "best_iteration": None, "best_step": None,
                    "latched": False, "steps_spent": 0}

    def save(self, state: Dict[str, Any]) -> None:
        # Rename: a crash must not leave a truncated file that reads as "no best".
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.state_path) or ".")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, sort_keys=True)
            os.replace(tmp, self.state_path)
        except Exception:
            os.unlink(tmp)
            raise

    def observe(self, val_metrics: dict, global_steps: int) -> Dict[str, Any]:
        """Record one validation; return is_best / stop_run / switch_to_sft / latched."""
        out = {"is_best": False, "stop_run": False, "switch_to_sft": False,
               "latched": False, "best_score": None, "best_iteration": None,
               "best_step": None, "steps_spent": 0, "score": None}
        if not self.enabled:
            return out

        score = val_metrics.get(self.metric)
        if score is None or not isinstance(score, (int, float)):
            print(f"[RunSchedule] metric {self.metric!r} absent; no decision")
            return out
        score = float(score)
        out["score"] = score

        st = self.load()
        best = st.get("best_score")
        latched = bool(st.get("latched")) or (
            best is not None and best > self.best_metric_threshold)

        if best is None or score > best + self.min_delta:
            st["best_score"] = score
            st["best_iteration"] = self.iteration
            st["best_step"] = global_steps
            self._misses = 0
            out["is_best"] = True
            print(f"[RunSchedule] new run-best {score:.4f} "
                  f"(iter {self.iteration}, step {global_steps})")
        else:
            self._misses += 1

        best_now = st.get("best_score")
        if best_now is not None and best_now > self.best_metric_threshold:
            if not latched:
                print("[RunSchedule] latched: no further SFT, graph can be released")
            latched = True
        st["latched"] = latched

        # Max step per iteration, summed: idempotent when an iteration resumes.
        by_iter = dict(st.get("steps_by_iter") or {})
        key = str(self.iteration)
        by_iter[key] = max(int(by_iter.get(key, 0)), int(global_steps))
        st["steps_by_iter"] = by_iter
        st["steps_spent"] = sum(by_iter.values())
        self.save(st)

        if st["steps_spent"] >= self.total_budget:
            out["stop_run"] = True
            print(f"[RunSchedule] budget spent "
                  f"({st['steps_spent']}/{self.total_budget})")

        patience = self.run_patience if latched else self.iter_patience
        if self._misses >= patience and not out["stop_run"]:
            if latched:
                out["stop_run"] = True
                print(f"[RunSchedule] run_patience {patience} reached")
            else:
                out["switch_to_sft"] = True
                print(f"[RunSchedule] iter_patience {patience} reached -> SFT")

        # The controller is a separate process and reads this to end the run.
        if out["stop_run"]:
            st["stop_run"] = True
            st["stop_reason"] = ("budget" if st["steps_spent"] >= self.total_budget
                                 else "run_patience")
            self.save(st)

        out.update(latched=latched, best_score=st.get("best_score"),
                   best_iteration=st.get("best_iteration"),
                   best_step=st.get("best_step"),
                   steps_spent=st.get("steps_spent"))
        return out

    def save_best_checkpoint(self, *, actor_rollout_wg, global_steps: int,
                             score: float) -> None:
        """Mirror the actor to ``<experiment_root>/best_val_run/``.

        Outside every ``iter_XXX``, which ``max_actor_ckpt_to_keep`` prunes.
        HuggingFace model only, no optimizer state.
        """
        if not self.enabled:
            return
        import shutil
        from vagen.utils.best_val import _prune_to_huggingface

        root = os.path.join(os.path.dirname(self.state_path), "best_val_run")
        actor_dir = os.path.join(root, "actor")
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)
        try:
            actor_rollout_wg.save_checkpoint(actor_dir, None, global_steps,
                                             max_ckpt_to_keep=None)
            _prune_to_huggingface(actor_dir)
        except Exception as e:
            # Must never kill training.
            print(f"[RunSchedule][WARN] best checkpoint save failed: "
                  f"{type(e).__name__}: {e}")
            return
        with open(os.path.join(root, "metadata.json"), "w") as f:
            json.dump({"score": score, "iteration": self.iteration,
                       "global_step": global_steps, "metric": self.metric},
                      f, indent=2, sort_keys=True)
        print(f"[RunSchedule] run-best checkpoint -> {root}")
