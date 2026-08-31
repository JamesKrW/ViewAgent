"""GraphRL phase wrapper for the adaptive synchronous SLIME driver."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from graphrl.adaptive.model import is_complete_hf_model
from graphrl.adaptive.state import (
    CONFIG_FILENAME,
    DECISION_STOP_RUN,
    DECISION_SWITCH_TO_SFT,
    AdaptiveStateStore,
)
from graphrl.slime.wrapper import SlimeWrapper
from graphrl.state import ModuleState


class AdaptiveSlimeWrapper(SlimeWrapper):
    """Complete an RL phase only after a durable adaptive decision exists."""

    name = "RL(SLIME-adaptive)"
    adaptive = True

    def _store(self) -> AdaptiveStateStore:
        experiment_dir = self.output_dir.parents[1]
        raw = json.loads((experiment_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
        return AdaptiveStateStore(experiment_dir, raw)

    def _terminal_decision(self) -> str | None:
        state = self._store().load()
        if int(state.get("decision_round", -1)) != int(self.config["_iter_num"]):
            return None
        if not state.get("decision_ready"):
            return None
        decision = state.get("decision")
        return str(decision) if decision in {DECISION_SWITCH_TO_SFT, DECISION_STOP_RUN} else None

    def _materialize_best(self, decision: str) -> None:
        store = self._store()
        state = store.load()
        round_index = int(self.config["_iter_num"])
        current = dict((state.get("rounds") or {}).get(str(round_index)) or {})
        relative = current.get("checkpoint")
        if not relative:
            raise RuntimeError(f"adaptive round {round_index} has no committed best checkpoint")
        source = store.experiment_dir / relative / "actor" / "huggingface"
        if not is_complete_hf_model(source):
            raise RuntimeError(f"adaptive best model is incomplete: {source}")
        destination = Path(self.output_paths["model"])
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if decision == DECISION_STOP_RUN:
            os.symlink(os.path.relpath(source, destination.parent), destination)
        else:
            shutil.copytree(source, destination)
        self.completion_marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": "vagen-slime",
                    "adaptive": True,
                    "decision": decision,
                    "round": round_index,
                    "step": state.get("decision_step"),
                    "source": relative,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def is_done(self) -> bool:
        if self._process is None:
            return self.is_already_complete()
        return_code = self._process.poll()
        if return_code is None:
            return False
        if return_code != 0:
            self._state = ModuleState.FAILED
            self._log(f"adaptive SLIME trainer exited with code {return_code}")
            return False
        decision = self._terminal_decision()
        if decision is None:
            self._state = ModuleState.FAILED
            self._log("adaptive SLIME exited without a durable terminal decision")
            return False
        self._materialize_best(decision)
        self._state = ModuleState.DONE
        return True

    def is_already_complete(self) -> bool:
        try:
            decision = self._terminal_decision()
        except (OSError, ValueError, RuntimeError):
            return False
        if decision is None:
            return False
        self._materialize_best(decision)
        return True


__all__ = ["AdaptiveSlimeWrapper"]
