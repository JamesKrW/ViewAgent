"""Resume-preserving LLaMA-Factory wrapper for adaptive runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from graphrl.llama_factory.lf_wrapper import LFWrapper
from graphrl.state import ModuleState
from graphrl.utils.process import kill_process_group


class AdaptiveLFWrapper(LFWrapper):
    """Keep SFT checkpoints on interruption and reject failed subprocesses."""

    def launch(self) -> None:
        # Warm Storage mirrors directory trees incrementally.  A preemption can
        # therefore leave a newer checkpoint-* directory only partially copied
        # beside an older complete checkpoint.  The shared config generator
        # intentionally keeps its legacy "highest directory wins" behaviour;
        # adaptive runs validate their own resume candidates before invoking it.
        self._prune_incomplete_sft_checkpoints()
        super().launch()

    def is_done(self) -> bool:
        if not self._is_lora and self._process is not None:
            return_code = self._process.poll()
            if return_code is not None and return_code != 0:
                self._state = ModuleState.FAILED
                self._log(f"SFT training failed (exit code {return_code})")
                return False
            done = super().is_done()
            if return_code == 0 and not done:
                self._state = ModuleState.FAILED
                self._log("SFT process exited cleanly without a complete model")
            return done
        return super().is_done()

    def kill(self) -> None:
        if self._ckpt_monitor:
            self._ckpt_monitor.stop()
            self._ckpt_monitor = None
        if self._process and self._process.poll() is None:
            kill_process_group(self._process, timeout=10)
        if self._merge_process and self._merge_process.poll() is None:
            kill_process_group(self._merge_process, timeout=10)
        for handle in (self._log_file_handle, self._merge_log_handle):
            if handle:
                try:
                    handle.close()
                except OSError:
                    pass
        self._log_file_handle = None
        self._merge_log_handle = None
        # Unlike the legacy wrapper, retain checkpoint-* directories so a
        # preempted adaptive job can resume SFT instead of restarting it.
        self._state = ModuleState.TERMINATED
        self._log("Killed; resumable SFT checkpoints retained")

    def has_complete_output(self) -> bool:
        model_dir = Path(self.output_paths["model"])
        if not (model_dir / "config.json").is_file():
            return False
        return bool(list(model_dir.glob("*.safetensors")) or list(model_dir.glob("*.bin")))

    def _prune_incomplete_sft_checkpoints(self) -> None:
        train_output_dir = self._lora_adapter_dir if self._is_lora else Path(
            self.output_paths["model"]
        )
        if not train_output_dir.is_dir():
            return

        world_size = max(1, int(self.config.get("n_gpus", 1)))
        uses_deepspeed = bool(
            (self.config.get("hydra_overrides") or {}).get("deepspeed")
        )
        for checkpoint in train_output_dir.glob("checkpoint-*"):
            if not checkpoint.is_dir():
                continue
            if self._is_resumable_sft_checkpoint(
                checkpoint,
                world_size=world_size,
                uses_deepspeed=uses_deepspeed,
                is_lora=self._is_lora,
            ):
                continue
            shutil.rmtree(checkpoint, ignore_errors=True)
            self._log(f"Removed incomplete SFT resume checkpoint: {checkpoint}")

    @staticmethod
    def _is_resumable_sft_checkpoint(
        checkpoint: Path,
        *,
        world_size: int,
        uses_deepspeed: bool,
        is_lora: bool,
    ) -> bool:
        try:
            checkpoint_step = int(checkpoint.name.rsplit("-", 1)[-1])
            trainer_state = json.loads(
                (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
            )
            if int(trainer_state["global_step"]) != checkpoint_step:
                return False
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

        if not (checkpoint / "config.json").is_file():
            return False
        weight_prefix = "adapter_model" if is_lora else "model"
        weight_files = [
            *checkpoint.glob(f"{weight_prefix}*.safetensors"),
            *checkpoint.glob(f"{weight_prefix}*.bin"),
        ]
        if not weight_files or any(path.stat().st_size == 0 for path in weight_files):
            return False

        rng_files = list(checkpoint.glob("rng_state*.pth"))
        if len(rng_files) < world_size or any(
            path.stat().st_size == 0 for path in rng_files
        ):
            return False

        if uses_deepspeed:
            try:
                tag = (checkpoint / "latest").read_text(encoding="utf-8").strip()
            except OSError:
                return False
            state_dir = checkpoint / tag
            model_states = list(state_dir.rglob("*model_states.pt"))
            optim_states = list(state_dir.rglob("*optim_states.pt"))
            return (
                bool(tag)
                and state_dir.is_dir()
                and bool(model_states)
                and all(path.stat().st_size > 0 for path in model_states)
                and len(optim_states) >= world_size
                and all(path.stat().st_size > 0 for path in optim_states)
            )

        required = [checkpoint / "optimizer.pt", checkpoint / "scheduler.pt"]
        return all(path.is_file() and path.stat().st_size > 0 for path in required)
