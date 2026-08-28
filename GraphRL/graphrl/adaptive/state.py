"""Persistent state for the metric-driven GraphRL controller.

The adaptive pipeline intentionally uses different config/state filenames from
the legacy ``rl_schedule`` implementation.  This prevents a new run from
silently consuming an old experiment's decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "adaptive_schedule_config.json"
STATE_FILENAME = "adaptive_schedule_state.json"
SCHEMA_VERSION = 2

PHASE_RL = "rl"
PHASE_TRAJ_TO_SFT = "traj_to_sft"
PHASE_SFT = "sft"
PHASE_FINALIZE_ROUND = "finalize_round"
PHASE_FINISHED = "finished"

DECISION_PENDING = "pending"
DECISION_CONTINUE_RL = "continue_rl"
DECISION_SWITCH_TO_SFT = "switch_to_sft"
DECISION_STOP_RUN = "stop_run"

_DEFAULTS = {
    "direction": "maximize",
    "eval_every_steps": 20,
    "min_delta": 0.03,
    "min_relative_delta": 0.10,
    "sft_patience": 3,
    "latch_threshold": 0.1,
    "finish_patience": 8,
    "total_rl_steps": 801,
}


def normalize_adaptive_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a validated, JSON-serializable adaptive schedule config."""
    cfg = dict(_DEFAULTS)
    cfg.update(dict(raw or {}))

    if not cfg.get("metric"):
        raise ValueError("adaptive_schedule.metric is required")
    if cfg["direction"] not in {"maximize", "minimize"}:
        raise ValueError("adaptive_schedule.direction must be maximize or minimize")

    cfg["metric"] = str(cfg["metric"])
    cfg["direction"] = str(cfg["direction"])
    cfg["eval_every_steps"] = int(cfg["eval_every_steps"])
    cfg["min_delta"] = float(cfg["min_delta"])
    cfg["min_relative_delta"] = float(cfg["min_relative_delta"])
    cfg["sft_patience"] = int(cfg["sft_patience"])
    cfg["latch_threshold"] = float(cfg["latch_threshold"])
    cfg["finish_patience"] = int(cfg["finish_patience"])
    cfg["total_rl_steps"] = int(cfg["total_rl_steps"])

    for key in ("eval_every_steps", "sft_patience", "finish_patience", "total_rl_steps"):
        if cfg[key] <= 0:
            raise ValueError(f"adaptive_schedule.{key} must be positive")
    if cfg["min_delta"] < 0:
        raise ValueError("adaptive_schedule.min_delta must be non-negative")
    if cfg["min_relative_delta"] < 0:
        raise ValueError(
            "adaptive_schedule.min_relative_delta must be non-negative"
        )
    return cfg


def _fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _initial_state(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": dict(config),
        "config_fingerprint": _fingerprint(config),
        "round": 0,
        "phase": PHASE_RL,
        "decision": DECISION_PENDING,
        "decision_round": 0,
        "decision_step": None,
        "decision_ready": False,
        "latched": False,
        "stop_reason": None,
        "steps_by_round": {},
        "total_rl_steps": 0,
        "run_best": None,
        "rounds": {},
    }


class AdaptiveStateStore:
    """Atomic reader/writer with config and schema compatibility checks."""

    def __init__(self, experiment_dir: str | Path, config: dict[str, Any]):
        self.experiment_dir = Path(experiment_dir).expanduser().resolve()
        self.config = normalize_adaptive_config(config)
        self.path = self.experiment_dir / STATE_FILENAME

    @property
    def config_fingerprint(self) -> str:
        return _fingerprint(self.config)

    def initialize(self) -> dict[str, Any]:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            return self.load()
        state = _initial_state(self.config)
        self.save(state)
        return state

    def load(self) -> dict[str, Any]:
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self.initialize()
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"adaptive state is unreadable: {self.path}") from exc

        if state.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"adaptive state schema mismatch in {self.path}: "
                f"expected {SCHEMA_VERSION}, got {state.get('schema_version')!r}"
            )
        expected = _fingerprint(self.config)
        if state.get("config_fingerprint") != expected:
            raise RuntimeError(
                "adaptive schedule parameters differ from the persisted run; "
                "use a new experiment directory instead of changing them in place"
            )
        return state

    def save(self, state: dict[str, Any]) -> None:
        self._write_atomic(self.path, state)

    @staticmethod
    def _write_atomic(path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def update(self, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        state = self.load()
        mutate(state)
        self.save(state)
        return state

    def set_phase(self, round_index: int, phase: str) -> dict[str, Any]:
        if phase not in {
            PHASE_RL,
            PHASE_TRAJ_TO_SFT,
            PHASE_SFT,
            PHASE_FINALIZE_ROUND,
            PHASE_FINISHED,
        }:
            raise ValueError(f"unknown adaptive phase: {phase}")

        def mutate(state: dict[str, Any]) -> None:
            if int(state["round"]) != int(round_index):
                raise RuntimeError(
                    f"cannot set phase for round {round_index}; state is round {state['round']}"
                )
            state["phase"] = phase

        return self.update(mutate)

    def begin_next_round(self, completed_round: int) -> dict[str, Any]:
        def mutate(state: dict[str, Any]) -> None:
            if int(state["round"]) != int(completed_round):
                raise RuntimeError(
                    f"cannot complete round {completed_round}; state is round {state['round']}"
                )
            if state.get("phase") != PHASE_FINALIZE_ROUND:
                raise RuntimeError(
                    f"cannot complete round {completed_round}; required phase "
                    f"{PHASE_FINALIZE_ROUND!r}, got {state.get('phase')!r}"
                )
            next_round = int(completed_round) + 1
            state["round"] = next_round
            state["phase"] = PHASE_RL
            state["decision"] = DECISION_PENDING
            state["decision_round"] = next_round
            state["decision_step"] = None
            state["decision_ready"] = False

        return self.update(mutate)

    def finish(self, reason: str) -> dict[str, Any]:
        def mutate(state: dict[str, Any]) -> None:
            state["phase"] = PHASE_FINISHED
            state["decision"] = DECISION_STOP_RUN
            state["decision_ready"] = True
            state["stop_reason"] = str(reason)

        return self.update(mutate)

    def snapshot_to(
        self, checkpoint_dir: str | Path, state: dict[str, Any] | None = None
    ) -> Path:
        """Copy the current atomic state beside a resumable verl checkpoint."""
        state = self.load() if state is None else state
        target = Path(checkpoint_dir) / STATE_FILENAME
        self._write_atomic(target, state)
        return target

    def restore_from(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        source = Path(checkpoint_dir) / STATE_FILENAME
        try:
            state = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"checkpoint has no valid adaptive state snapshot: {source}"
            ) from exc
        if state.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"adaptive checkpoint state schema mismatch: {source}")
        if state.get("config_fingerprint") != _fingerprint(self.config):
            raise RuntimeError(f"adaptive checkpoint config mismatch: {source}")
        self.save(state)
        return state
