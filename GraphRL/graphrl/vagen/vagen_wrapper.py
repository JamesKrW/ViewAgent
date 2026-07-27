"""
RL phase wrapper for the VAGEN/verl backend.

Mono-backend by design — the project commits to VAGEN for RL. ``VagenWrapper``
is the only RL implementation; no registry, no abstract layer.

Responsibilities:
  - Spawn ``vagen.main_ppo`` as a subprocess (own process group)
  - Forward stdout to a per-iteration ``rl_training.log`` and stdout
  - Run a ``CheckpointMonitor`` daemon thread that promotes the final
    verl checkpoint into ``rl_model/``
  - Provide ``is_already_complete()`` for resume (checks for a complete
    ``rl_model/`` and falls back to recovering the latest ``verl_checkpoints``
    snapshot if ``rl_model/`` was lost)

Cleanup of intermediate dirs (``verl_checkpoints``, ``rollout_data``, …) is
handled by the controller's iteration-level cleanup, NOT here.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from graphrl.state import ModuleOutput, ModuleState
from graphrl.vagen.utils.command_builder import build_vagen_command
from graphrl.utils.checkpoint_monitor import CheckpointMonitor
from graphrl.utils.process import kill_process_group

logger = logging.getLogger(__name__)


class VagenWrapper:
    """RL training phase backed by VAGEN/verl.

    Required ``input_paths``:
        ``model``     -- starting HF model directory.

    Required ``output_paths``:
        ``base_dir``  -- ``iter_XXX/rl/``
        ``model``     -- ``iter_XXX/rl_model/`` (populated by CheckpointMonitor)

    Notable config keys:
        ``vagen_dir``             -- VAGEN repo root (must contain ``vagen/main_ppo``)
        ``training_steps``        -- target step count (used by CheckpointMonitor)
        ``timeout``               -- watchdog seconds (consumed by controller)
        ``poll_interval``         -- is_done() polling cadence
        ``hydra_overrides``       -- nested dict flattened onto the VAGEN CLI

    Note: ``graph_builder`` and ``merge_interval`` are NO LONGER read here —
    rollout-to-graph conversion has moved to the TrajToSFT phase.
    """

    name = "RL"

    def __init__(
        self,
        config: Dict[str, Any],
        input_paths: Dict[str, str],
        output_paths: Dict[str, str],
    ):
        self.config = config
        self.input_paths = input_paths
        self.output_paths = output_paths
        self._state = ModuleState.IDLE
        self._process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._log_file_handle = None
        self._ckpt_monitor: Optional[CheckpointMonitor] = None

    @property
    def state(self) -> ModuleState:
        return self._state

    # ── lifecycle ─────────────────────────────────────────────────────────

    def launch(self) -> None:
        output_dir = Path(self.output_paths["base_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.input_paths["model"]

        cmd = build_vagen_command(
            config=self.config,
            model_path=model_path,
            output_dir=output_dir,
        )
        self._log(f"Command: {' '.join(cmd[:6])} ...")

        log_file = output_dir / "rl_training.log"
        self._log_file_handle = open(log_file, "w")

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
            target=self._forward_output, daemon=True, name="rl-log-fwd"
        )
        self._log_thread.start()

        # CheckpointMonitor promotes the final verl checkpoint to ``rl_model/``.
        # ``clean_intermediates`` is False here — iteration-level cleanup is
        # the controller's job, not the module's.
        self._ckpt_monitor = CheckpointMonitor(
            source_dir=output_dir / "verl_checkpoints",
            dest_path=Path(self.output_paths["model"]),
            target_steps=self.config.get("training_steps", 5),
            interval=self.config.get("poll_interval", 30),
            clean_intermediates=False,
        )
        self._ckpt_monitor.start()

        self._state = ModuleState.LAUNCHED
        self._log(f"Launched (PID: {self._process.pid})")

    def is_done(self) -> bool:
        model_path = Path(self.output_paths["model"])
        if not _is_model_dir_complete(model_path):
            # Fail fast: if the RL subprocess has exited (crashed) without
            # producing a complete rl_model, mark FAILED so the controller
            # raises instead of polling until the (72h) watchdog timeout.
            if self._process is not None and self._process.poll() is not None:
                if self._process.returncode != 0:
                    self._state = ModuleState.FAILED
            return False
        if self._process and self._process.poll() is None:
            return False
        if self._state == ModuleState.LAUNCHED:
            self._state = ModuleState.DONE
        return True

    def kill(self) -> None:
        if self._ckpt_monitor:
            self._ckpt_monitor.stop()
            self._ckpt_monitor = None
        if self._process and self._process.poll() is None:
            kill_process_group(self._process, timeout=10)
        if self._log_file_handle:
            try:
                self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None
        self._state = ModuleState.TERMINATED
        self._log("Killed")

    def is_already_complete(self) -> bool:
        """Resume check: ``rl_model/`` complete, or recoverable from verl ckpt."""
        model_path = Path(self.output_paths.get("model", ""))
        if _is_model_dir_complete(model_path):
            return True

        ckpt_dir = Path(self.output_paths["base_dir"]) / "verl_checkpoints"
        target_steps = self.config.get("training_steps", 5)
        if _try_recover_from_checkpoint(ckpt_dir, model_path, target_steps):
            self._log("Recovered model from existing checkpoint, skipping VAGEN training")
            return True
        return False

    def get_output(self) -> ModuleOutput:
        out = ModuleOutput()
        model_path = self.output_paths.get("model")
        if model_path and Path(model_path).exists():
            out.model_path = model_path
        return out

    # ── helpers ───────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        logger.info("[%s] %s", self.name, msg)

    def _forward_output(self) -> None:
        try:
            for line in iter(self._process.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
                if self._log_file_handle and not self._log_file_handle.closed:
                    self._log_file_handle.write(line)
                    self._log_file_handle.flush()
        except Exception:
            pass


# ── shared utilities (used by RLModule + recovery flow) ───────────────────


def _is_model_dir_complete(path: Path) -> bool:
    """Return True if ``path`` looks like a loadable HuggingFace model dir."""
    if not path.exists() or not (path / "config.json").exists():
        return False
    safetensors = list(path.glob("*.safetensors"))
    bins = list(path.glob("*.bin"))
    if not safetensors and not bins:
        return False
    if safetensors and any("of" in f.name for f in safetensors):
        if not (path / "model.safetensors.index.json").exists():
            return False
    if bins and any("of" in f.name for f in bins):
        if not (path / "pytorch_model.bin.index.json").exists():
            return False
    return True


def _try_recover_from_checkpoint(ckpt_dir: Path, dest: Path, target_steps: int) -> bool:
    """Find a complete verl checkpoint at ``>= target_steps`` and copy it to dest."""
    if not ckpt_dir.exists():
        return False
    import shutil

    ckpts = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("global_step_"):
            try:
                ckpts.append((d, int(d.name.split("_")[-1])))
            except ValueError:
                continue
    ckpts.sort(key=lambda x: x[1])
    if not ckpts:
        return False

    best_dir = None
    for ckpt_path, step in reversed(ckpts):
        if step < target_steps:
            break
        hf_path = ckpt_path / "actor" / "huggingface"
        if _is_model_dir_complete(hf_path):
            best_dir = hf_path
            break
    if best_dir is None:
        return False

    if dest.exists():
        shutil.rmtree(dest)
    CheckpointMonitor._copy_checkpoint(best_dir, dest)
    return True
