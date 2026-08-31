"""Resume-preserving LLaMA-Factory wrapper for adaptive runs."""

from __future__ import annotations

from pathlib import Path

from graphrl.adaptive.model import is_complete_hf_model
from graphrl.llama_factory.lf_wrapper import LFWrapper
from graphrl.state import ModuleState
from graphrl.utils.process import kill_process_group


class AdaptiveLFWrapper(LFWrapper):
    """Retain all valid SFT checkpoints after an adaptive interruption."""

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
        self._state = ModuleState.TERMINATED
        self._log("Killed; resumable SFT checkpoints retained")

    def has_complete_output(self) -> bool:
        return is_complete_hf_model(Path(self.output_paths["model"]))


__all__ = ["AdaptiveLFWrapper"]
