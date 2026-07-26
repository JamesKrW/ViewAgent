"""
SFT phase wrapper for the LLaMA-Factory backend.

Mono-backend by design — the project commits to LLaMA-Factory for SFT.
``LFWrapper`` is the only SFT implementation; no registry, no abstract layer.

Two-phase behaviour for LoRA: (1) train adapter into ``_lora_adapter/``,
(2) run ``llamafactory-cli export`` to merge the adapter into a full model.
For full fine-tuning, only phase 1 runs.

Cleanup of intermediate ``checkpoint-N/`` dirs is handled by the controller's
iteration-level cleanup, NOT here. The minimal in-module cleanup just stops
helper threads and subprocess on ``kill()``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from graphrl.state import ModuleOutput, ModuleState
from graphrl.llama_factory.utils.config_generator import (
    generate_sft_config,
    generate_merge_config,
)
from graphrl.utils.checkpoint_monitor import CheckpointMonitor
from graphrl.utils.process import kill_process_group

logger = logging.getLogger(__name__)


class LFWrapper:
    """SFT training phase backed by LLaMA-Factory.

    Required ``input_paths``:
        ``model``     -- starting HF model directory (typically rl_model/)
        ``sft_data``  -- directory containing dataset_info.json + per-dataset .json files

    Required ``output_paths``:
        ``base_dir``  -- ``iter_XXX/sft/``  (logs, generated config)
        ``model``     -- ``iter_XXX/sft_model/`` (final HF model)

    Notable config keys:
        ``llama_factory_dir``  -- LLaMA-Factory repo root (used as cwd)
        ``n_gpus``             -- exposed via NPROC_PER_NODE
        ``hydra_overrides``    -- nested dict that becomes the LF training YAML
        ``_project_name``      -- WANDB_PROJECT (set by controller)
    """

    name = "SFT"

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
        # LoRA merge state
        self._merge_process: Optional[subprocess.Popen] = None
        self._merge_log_thread: Optional[threading.Thread] = None
        self._merge_log_handle = None

    @property
    def state(self) -> ModuleState:
        return self._state

    @property
    def _is_lora(self) -> bool:
        return self.config.get("hydra_overrides", {}).get("finetuning_type") == "lora"

    @property
    def _lora_adapter_dir(self) -> Path:
        return Path(self.output_paths["model"]) / "_lora_adapter"

    # ── lifecycle ─────────────────────────────────────────────────────────

    def launch(self) -> None:
        output_dir = Path(self.output_paths["base_dir"])
        model_dir = Path(self.output_paths["model"])
        output_dir.mkdir(parents=True, exist_ok=True)

        sft_config_path = generate_sft_config(
            config=self.config,
            model_path=self.input_paths["model"],
            dataset_dir=self.input_paths["sft_data"],
            output_dir=model_dir,
        )
        self._log(f"Config generated at {sft_config_path}")

        cmd = ["llamafactory-cli", "train", str(sft_config_path)]

        project_name = self.config.get("_project_name", "graphrl")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "WANDB_PROJECT": project_name}
        if self.config.get("force_torchrun", True):
            env["FORCE_TORCHRUN"] = "1"
        n_gpus = self.config.get("n_gpus")
        if n_gpus:
            env["NPROC_PER_NODE"] = str(n_gpus)

        log_file = output_dir / "sft_training.log"
        self._log_file_handle = open(log_file, "w")

        llama_factory_dir = self.config.get("llama_factory_dir")
        cwd = str(Path(llama_factory_dir).expanduser()) if llama_factory_dir else None

        self._process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=1,
            universal_newlines=True,
        )

        self._log_thread = threading.Thread(
            target=self._forward_output, daemon=True, name="sft-log-fwd"
        )
        self._log_thread.start()

        # For LoRA, monitor the adapter subdir; for full FT, monitor model_dir.
        # Cleaning intermediate ``checkpoint-N/`` is fine here (LF rotates them).
        ckpt_source = self._lora_adapter_dir if self._is_lora else model_dir
        self._ckpt_monitor = CheckpointMonitor(
            source_dir=ckpt_source,
            dest_path=ckpt_source,
            target_steps=None,  # detect by config.json presence
            interval=self.config.get("poll_interval", 30),
            clean_intermediates=True,
        )
        self._ckpt_monitor.start()

        self._state = ModuleState.LAUNCHED
        self._log(f"Launched (PID: {self._process.pid})")

    def is_done(self) -> bool:
        model_dir = Path(self.output_paths["model"])
        if self._is_lora:
            return self._is_done_lora(model_dir)

        if not (model_dir / "config.json").exists():
            return False
        safetensors = list(model_dir.glob("*.safetensors"))
        bins = list(model_dir.glob("*.bin"))
        if not safetensors and not bins:
            return False
        if self._process and self._process.poll() is None:
            return False
        if self._state == ModuleState.LAUNCHED:
            _patch_text_model_type(model_dir / "config.json")
            _patch_image_processor_fast(model_dir)
            _drop_llamafactory_readme(model_dir)
            self._state = ModuleState.DONE
        return True

    def _is_done_lora(self, model_dir: Path) -> bool:
        adapter_dir = self._lora_adapter_dir

        # Phase 1: training process
        if self._process and self._process.poll() is None:
            return False
        if self._process is not None:
            train_returncode = self._process.returncode
            kill_process_group(self._process, timeout=10)
            self._process = None
            if self._log_file_handle:
                try:
                    self._log_file_handle.close()
                except Exception:
                    pass
                self._log_file_handle = None
            if self._ckpt_monitor:
                self._ckpt_monitor.stop()
                self._ckpt_monitor = None
            if train_returncode != 0:
                self._log(f"LoRA training failed (exit code {train_returncode})")
                return False
            self._log("Training process cleaned up, GPUs released")

        if not (adapter_dir / "adapter_config.json").exists():
            return False

        # Phase 2: merge
        if self._merge_process is None:
            self._start_merge(model_dir)
            return False

        if self._merge_process.poll() is None:
            return False

        if not (model_dir / "config.json").exists():
            if self._merge_process.returncode != 0:
                self._log(
                    f"LoRA merge failed (exit code {self._merge_process.returncode})"
                )
            return False
        safetensors = list(model_dir.glob("*.safetensors"))
        bins = list(model_dir.glob("*.bin"))
        if not safetensors and not bins:
            return False

        if self._state == ModuleState.LAUNCHED:
            _patch_text_model_type(model_dir / "config.json")
            _patch_image_processor_fast(model_dir)
            _drop_llamafactory_readme(model_dir)
            self._state = ModuleState.DONE
            self._log("LoRA adapter merged successfully")
        return True

    def kill(self) -> None:
        if self._ckpt_monitor:
            self._ckpt_monitor.stop()
            self._ckpt_monitor = None
        if self._process and self._process.poll() is None:
            kill_process_group(self._process, timeout=10)
        if self._merge_process and self._merge_process.poll() is None:
            kill_process_group(self._merge_process, timeout=10)
        for h in (self._log_file_handle, self._merge_log_handle):
            if h:
                try:
                    h.close()
                except Exception:
                    pass
        self._log_file_handle = None
        self._merge_log_handle = None

        # Clean intermediate ``checkpoint-N/`` dirs (LF leftovers; not resume-relevant).
        model_dir = Path(self.output_paths["model"])
        if model_dir.exists():
            for d in model_dir.iterdir():
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    shutil.rmtree(d, ignore_errors=True)
        if self._is_lora and self._lora_adapter_dir.exists():
            for d in self._lora_adapter_dir.iterdir():
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    shutil.rmtree(d, ignore_errors=True)

        self._state = ModuleState.TERMINATED
        self._log("Killed")

    def is_already_complete(self) -> bool:
        model_dir = Path(self.output_paths.get("model", ""))
        if (model_dir / "config.json").exists():
            return True
        return False

    def get_output(self) -> ModuleOutput:
        model_path = self.output_paths.get("model")
        if model_path and (Path(model_path) / "config.json").exists():
            _patch_text_model_type(Path(model_path) / "config.json")
            _drop_llamafactory_readme(Path(model_path))
            return ModuleOutput(model_path=model_path)
        return ModuleOutput()

    # ── internals ─────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        logger.info("[%s] %s", self.name, msg)

    def _start_merge(self, model_dir: Path) -> None:
        self._log("Starting LoRA adapter merge...")
        output_dir = Path(self.output_paths["base_dir"])
        model_path = self.input_paths["model"]
        template = self.config.get("hydra_overrides", {}).get("template", "qwen2_vl")

        merge_config_path = generate_merge_config(
            model_path=model_path,
            adapter_dir=self._lora_adapter_dir,
            export_dir=model_dir,
            template=template,
        )

        cmd = ["llamafactory-cli", "export", str(merge_config_path)]
        llama_factory_dir = self.config.get("llama_factory_dir")
        cwd = str(Path(llama_factory_dir).expanduser()) if llama_factory_dir else None

        merge_log_file = output_dir / "sft_merge.log"
        self._merge_log_handle = open(merge_log_file, "w")

        self._merge_process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        self._merge_log_thread = threading.Thread(
            target=self._forward_process_output,
            args=(self._merge_process, self._merge_log_handle),
            daemon=True,
            name="sft-merge-log-fwd",
        )
        self._merge_log_thread.start()
        self._log(f"Merge process started (PID: {self._merge_process.pid})")

    def _forward_process_output(self, process, log_handle) -> None:
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
                if log_handle and not log_handle.closed:
                    log_handle.write(line)
                    log_handle.flush()
        except Exception:
            pass

    def _forward_output(self) -> None:
        self._forward_process_output(self._process, self._log_file_handle)


def _patch_text_model_type(config_path: Path) -> None:
    """Fix ``text_config.model_type`` for sglang compatibility (Qwen2.5-VL)."""
    try:
        cfg = json.loads(config_path.read_text())
        text_cfg = cfg.get("text_config", {})
        if text_cfg.get("model_type") == "qwen2_5_vl_text":
            text_cfg["model_type"] = "qwen2_5_vl"
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")
            logger.info("[SFT] Patched text_config.model_type: qwen2_5_vl_text -> qwen2_5_vl")
    except Exception as e:
        logger.warning("[SFT] Failed to patch model config: %s", e)


def _patch_image_processor_fast(model_dir: Path) -> None:
    """Ensure the saved processor uses the *fast* image processor (InternVL).

    LLaMA-Factory saves the slow ``GotOcr2ImageProcessor``, but vLLM's InternVL
    rollout requires ``GotOcr2ImageProcessorFast`` (otherwise ``init_device``
    raises ``GotOcr2ImageProcessorFast is expected but got GotOcr2ImageProcessor``).
    Image preprocessing params are unchanged by SFT, so flipping the type is safe.
    """
    pp = model_dir / "preprocessor_config.json"
    try:
        if pp.is_file():
            cfg = json.loads(pp.read_text())
            if cfg.get("image_processor_type") == "GotOcr2ImageProcessor":
                cfg["image_processor_type"] = "GotOcr2ImageProcessorFast"
                pp.write_text(json.dumps(cfg, indent=2) + "\n")
                logger.info(
                    "[SFT] Patched image_processor_type: GotOcr2ImageProcessor -> "
                    "GotOcr2ImageProcessorFast in %s", pp,
                )
    except Exception as e:
        logger.warning("[SFT] Failed to patch preprocessor_config: %s", e)


def _drop_llamafactory_readme(model_dir: Path) -> None:
    """Delete LlamaFactory's auto-generated ``README.md``.

    Its YAML frontmatter sets ``base_model:`` to a local absolute path, which
    HuggingFace's model-card validator parses as a malformed repo id and
    rejects on upload. The README content is just an LF training-stats
    template — no signal worth keeping.
    """
    readme = model_dir / "README.md"
    if readme.is_file():
        try:
            readme.unlink()
            logger.info("[SFT] Removed LlamaFactory-generated README.md from %s", model_dir)
        except Exception as e:
            logger.warning("[SFT] Failed to remove %s: %s", readme, e)
