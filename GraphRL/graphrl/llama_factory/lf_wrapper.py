"""
SFT phase wrapper for the LLaMA-Factory backend.

Mono-backend by design — the project commits to LLaMA-Factory for SFT.
``LFWrapper`` is the only SFT implementation; no registry, no abstract layer.

Two-phase behaviour for LoRA: (1) train adapter into ``_lora_adapter/``,
(2) run ``llamafactory-cli export`` to merge the adapter into a full model.
For full fine-tuning, only phase 1 runs.

Complete intermediate ``checkpoint-N/`` dirs are retained after interruption
for exact resume and removed only after the SFT phase has completed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from graphrl.adaptive.model import is_complete_hf_model
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
        ``model``     -- ``iter_XXX/sft/sft_model/`` (final HF model)

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

        # LLaMA-Factory chooses the highest checkpoint directory by name.
        # Remove only incomplete copies before config generation so a partial
        # warm-storage sync cannot hide an older, valid resume point.
        self._prune_incomplete_sft_checkpoints()

        sft_config_path = generate_sft_config(
            config=self.config,
            model_path=self.input_paths["model"],
            dataset_dir=self.input_paths["sft_data"],
            output_dir=model_dir,
        )
        self._log(f"Config generated at {sft_config_path}")

        cmd = ["llamafactory-cli", "train", str(sft_config_path)]

        project_name = self.config.get("_project_name", "graphrl")
        experiment_name = self.config.get("_experiment_name", "graphrl_pipeline")
        iteration = int(self.config.get("_iter_num", 0))
        identity = "\0".join(
            (project_name, experiment_name, str(iteration), "sft", str(model_dir.resolve()))
        )
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "WANDB_PROJECT": project_name}
        if env.get("WANDB_MODE", "online") != "disabled":
            env.setdefault("WANDB_RUN_ID", hashlib.sha256(identity.encode()).hexdigest()[:16])
            env.setdefault("WANDB_RESUME", "allow")
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

        if self._process is not None:
            return_code = self._process.poll()
            if return_code is not None and return_code != 0:
                self._state = ModuleState.FAILED
                self._log(f"SFT training failed (exit code {return_code})")
                return False
        if not is_complete_hf_model(model_dir):
            if self._process is not None and self._process.poll() == 0:
                self._state = ModuleState.FAILED
                self._log("SFT process exited cleanly without a complete model")
            return False
        if self._process and self._process.poll() is None:
            return False
        if self._state == ModuleState.LAUNCHED:
            _patch_text_model_type(model_dir / "config.json")
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

        if not is_complete_hf_model(model_dir):
            if self._merge_process.returncode != 0:
                self._log(
                    f"LoRA merge failed (exit code {self._merge_process.returncode})"
                )
            return False

        if self._state == ModuleState.LAUNCHED:
            _patch_text_model_type(model_dir / "config.json")
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

        completed = self._state == ModuleState.DONE
        if completed:
            # Successful phases no longer need intermediate trainer shards.
            model_dir = Path(self.output_paths["model"])
            if model_dir.exists():
                for directory in model_dir.iterdir():
                    if directory.is_dir() and directory.name.startswith("checkpoint-"):
                        shutil.rmtree(directory, ignore_errors=True)
            if self._is_lora and self._lora_adapter_dir.exists():
                for directory in self._lora_adapter_dir.iterdir():
                    if directory.is_dir() and directory.name.startswith("checkpoint-"):
                        shutil.rmtree(directory, ignore_errors=True)
            self._log("Released resources; completed SFT checkpoints cleaned")
        else:
            self._state = ModuleState.TERMINATED
            self._log("Killed; resumable SFT checkpoints retained")

    def is_already_complete(self) -> bool:
        model_dir = Path(self.output_paths.get("model", ""))
        return is_complete_hf_model(model_dir)

    def get_output(self) -> ModuleOutput:
        model_path = self.output_paths.get("model")
        if model_path and is_complete_hf_model(model_path):
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
