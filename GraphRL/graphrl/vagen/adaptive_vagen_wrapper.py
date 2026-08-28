"""VAGEN wrapper for the isolated adaptive controller."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

from graphrl.adaptive.model import is_complete_hf_model
from graphrl.adaptive.state import (
    CONFIG_FILENAME,
    DECISION_STOP_RUN,
    DECISION_SWITCH_TO_SFT,
    AdaptiveStateStore,
)
from graphrl.state import ModuleState
from graphrl.vagen.utils.command_builder import build_vagen_command
from graphrl.vagen.vagen_wrapper import VagenWrapper


class AdaptiveVagenWrapper(VagenWrapper):
    """Treat an adaptive decision, rather than a fixed step, as RL completion."""

    def launch(self) -> None:
        output_dir = Path(self.output_paths["base_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(output_dir)
        self._log(f"Command: {' '.join(cmd[:6])} ...")

        log_file = output_dir / "rl_training.log"
        self._log_file_handle = open(log_file, "w")  # noqa: SIM115 - closed by kill()
        vagen_dir = Path(self.config["vagen_dir"]).expanduser()
        self._process = subprocess.Popen(
            cmd,
            cwd=str(vagen_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=1,
            universal_newlines=True,
        )
        self._log_thread = threading.Thread(
            target=self._forward_output, daemon=True, name="adaptive-rl-log-fwd"
        )
        self._log_thread.start()
        self._state = ModuleState.LAUNCHED
        self._log(f"Launched adaptive trainer (PID: {self._process.pid})")

    def _build_command(self, output_dir: Path) -> list[str]:
        cmd = build_vagen_command(
            config=self.config,
            model_path=self.input_paths["model"],
            output_dir=output_dir,
        )
        cmd[2] = "vagen.main_ppo_adaptive"
        # total_training_steps is the remaining run budget expressed in this
        # round's local coordinates. Ensure the epoch loop cannot end first.
        cmd.append(f"trainer.total_epochs={max(1, int(self.config['training_steps']))}")
        return cmd

    def is_done(self) -> bool:
        if self._process is None:
            return self.is_already_complete()
        return_code = self._process.poll()
        if return_code is None:
            return False
        if return_code != 0:
            self._state = ModuleState.FAILED
            self._log(f"Adaptive trainer exited with code {return_code}")
            return False

        decision = self._terminal_decision()
        if decision is None:
            self._state = ModuleState.FAILED
            self._log("Adaptive trainer exited without a terminal schedule decision")
            return False
        self._materialize_round_best(decision)
        self._state = ModuleState.DONE
        return True

    def is_already_complete(self) -> bool:
        decision = self._terminal_decision()
        if decision is None:
            return False
        self._materialize_round_best(decision)
        return True

    def _store(self) -> AdaptiveStateStore:
        experiment_dir = Path(self.output_paths["base_dir"]).parents[1]
        config_path = experiment_dir / CONFIG_FILENAME
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return AdaptiveStateStore(experiment_dir, raw)

    def _terminal_decision(self) -> str | None:
        state = self._store().load()
        round_index = int(self.config["_iter_num"])
        if int(state.get("decision_round", -1)) != round_index:
            return None
        if not state.get("decision_ready"):
            return None
        decision = state.get("decision")
        if decision in {DECISION_SWITCH_TO_SFT, DECISION_STOP_RUN}:
            return str(decision)
        return None

    def _materialize_round_best(self, decision: str) -> None:
        state = self._store().load()
        round_index = int(self.config["_iter_num"])
        round_state = dict((state.get("rounds") or {}).get(str(round_index)) or {})
        relative_snapshot = round_state.get("checkpoint")
        if not relative_snapshot:
            raise RuntimeError(
                f"round {round_index} ended with {decision} but has no best checkpoint"
            )
        source = self._store().experiment_dir / relative_snapshot / "actor" / "huggingface"
        if not is_complete_hf_model(source):
            raise RuntimeError(f"adaptive round-best checkpoint is incomplete: {source}")

        dest = Path(self.output_paths["model"])
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if decision == DECISION_STOP_RUN:
            # No downstream SFT consumes this model. A relative symlink keeps the
            # final round resumable without duplicating a multi-GB HF snapshot.
            os.symlink(os.path.relpath(source, dest.parent), dest)
        else:
            # TrajToSFT/SFT and legacy cleanup expect a real per-round RL output.
            shutil.copytree(source, dest)
        self._completion_marker().write_text(
            json.dumps(
                {
                    "decision": decision,
                    "round": round_index,
                    "step": state.get("decision_step"),
                    "source": relative_snapshot,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._log(f"Materialized round-best model from {source}")

    def _completion_marker(self) -> Path:
        return Path(self.output_paths["base_dir"]) / ".adaptive_rl_done.json"
