"""GraphRL phase wrapper for the synchronous VAGEN-SLIME trainer."""

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
from typing import Any

from graphrl.adaptive.model import (
    is_complete_checkpoint_manifest,
    is_complete_hf_model,
)
from graphrl.adaptive.rollouts import durable_rollout_prefix
from graphrl.slime.config import SlimeLaunchSpec, build_launch_spec
from graphrl.state import ModuleOutput, ModuleState
from graphrl.utils.process import kill_process_group

logger = logging.getLogger(__name__)


def _fingerprint(spec: SlimeLaunchSpec) -> str:
    value = spec.to_dict()
    # Resume is derived from the checkpoint tracker and necessarily changes
    # after a successful run; it is not part of the experiment identity.
    value.pop("resume", None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _rollout_id(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except ValueError:
        return -1


class SlimeWrapper:
    """Run one RL phase and materialize its final Hugging Face checkpoint."""

    name = "RL(SLIME)"
    adaptive = False

    def __init__(
        self,
        config: dict[str, Any],
        input_paths: dict[str, str],
        output_paths: dict[str, str],
    ) -> None:
        self.config = config
        self.input_paths = input_paths
        self.output_paths = output_paths
        self._state = ModuleState.IDLE
        self._process: subprocess.Popen | None = None
        self._log_thread: threading.Thread | None = None
        self._log_file_handle = None

    @property
    def state(self) -> ModuleState:
        return self._state

    @property
    def output_dir(self) -> Path:
        return Path(self.output_paths["base_dir"]).resolve()

    @property
    def completion_marker(self) -> Path:
        return self.output_dir / ".slime_rl_done.json"

    def launch_spec(self) -> SlimeLaunchSpec:
        return build_launch_spec(
            self.config,
            model_path=self.input_paths["model"],
            output_dir=self.output_dir,
            adaptive=self.adaptive,
        )

    def launch(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.adaptive:
            self._repair_resume_checkpoint()
        spec = self.launch_spec()
        config_path = self.output_dir / "slime_launch.json"
        config_path.write_text(
            json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "graphrl.slime.launcher",
            "--config",
            str(config_path),
        ]
        self._log(f"Command: {' '.join(command)}")
        self._log_file_handle = (self.output_dir / "rl_training.log").open(
            "a", encoding="utf-8"
        )
        self._process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[3]),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            bufsize=1,
            universal_newlines=True,
        )
        self._log_thread = threading.Thread(
            target=self._forward_output,
            daemon=True,
            name="slime-rl-log-fwd",
        )
        self._log_thread.start()
        self._state = ModuleState.LAUNCHED

    def _latest_hf_checkpoint(self, *, require_terminal: bool) -> Path | None:
        candidates = sorted(
            (path for path in (self.output_dir / "hf_checkpoints").glob("rollout_*")
             if is_complete_hf_model(path)),
            key=_rollout_id,
        )
        if not candidates:
            return None
        if require_terminal:
            expected = self.launch_spec().num_rollout - 1
            return next(
                (path for path in reversed(candidates) if _rollout_id(path) == expected),
                None,
            )
        return candidates[-1]

    def _terminal_resume_checkpoint_is_complete(self) -> bool:
        spec = self.launch_spec()
        if spec.num_rollout <= 0:
            return True
        rollout_id = spec.num_rollout - 1
        checkpoint = self.output_dir / "slime_checkpoints" / f"iter_{rollout_id:07d}"
        return (
            is_complete_checkpoint_manifest(checkpoint)
            and durable_rollout_prefix(self.output_dir / "rollout_data")
            >= spec.num_rollout
        )

    def _repair_resume_checkpoint(self) -> None:
        """Point SLIME at the latest checkpoint consistent with rollout data.

        A process can die after Megatron advances its tracker but before the
        critic, dataset cursor, rollout JSONL, or commit manifest is durable.
        Only manifests written after all of those pieces are complete are
        eligible resume points.
        """

        checkpoint_root = self.output_dir / "slime_checkpoints"
        tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
        advertised: int | None = None
        if tracker.is_file():
            try:
                advertised = int(tracker.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                self._log(f"ignoring invalid SLIME checkpoint tracker: {tracker}")

        rollout_dir = self.output_dir / "rollout_data"
        durable = durable_rollout_prefix(rollout_dir)
        valid = []
        for checkpoint in checkpoint_root.glob("iter_*"):
            rollout_id = _rollout_id(checkpoint)
            if (
                0 <= rollout_id
                and rollout_id + 1 <= durable
                and is_complete_checkpoint_manifest(checkpoint)
            ):
                valid.append(rollout_id)
        selected = max(valid, default=-1)
        if selected != advertised:
            advertised_label = "none" if advertised is None else str(advertised)
            selected_label = "initial model" if selected < 0 else f"rollout {selected}"
            self._log(
                f"SLIME checkpoint {advertised_label} is not the latest complete "
                f"resume point; selecting {selected_label}"
            )
        if selected < 0:
            tracker.unlink(missing_ok=True)
        else:
            tracker.parent.mkdir(parents=True, exist_ok=True)
            temporary = tracker.with_name(f".{tracker.name}.tmp")
            temporary.write_text(str(selected), encoding="utf-8")
            os.replace(temporary, tracker)
        # Even when the tracker itself was already correct, copied or crashed
        # runs may contain newer uncommitted directories.  Remove them before
        # Megatron attempts to save the same rollout id again.
        self._prune_generated_after(selected)

    def _prune_generated_after(self, rollout_id: int) -> None:
        checkpoint_root = self.output_dir / "slime_checkpoints"
        for base in (checkpoint_root, checkpoint_root / "critic"):
            for path in base.glob("iter_*"):
                if _rollout_id(path) > rollout_id:
                    shutil.rmtree(path, ignore_errors=True)
        for path in (checkpoint_root / "rollout").glob("global_dataset_state_dict_*.pt"):
            try:
                dataset_rollout_id = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if dataset_rollout_id > rollout_id:
                path.unlink(missing_ok=True)
        for path in (self.output_dir / "hf_checkpoints").glob("rollout_*"):
            if _rollout_id(path) > rollout_id:
                shutil.rmtree(path, ignore_errors=True)

        keep_step = rollout_id + 1
        rollout_root = self.output_dir / "rollout_data"
        staging_root = rollout_root / ".staging"
        for path in staging_root.glob("step_*"):
            try:
                step = int(path.name[5:])
            except ValueError:
                continue
            if step > keep_step:
                shutil.rmtree(path, ignore_errors=True)
        for path in rollout_root.glob("*"):
            name = path.name
            if name.startswith("image_"):
                suffix = name[6:]
            elif name.endswith(".complete"):
                suffix = name[: -len(".complete")]
            elif name.endswith(".jsonl"):
                suffix = path.stem
            else:
                continue
            try:
                step = int(suffix)
            except ValueError:
                continue
            if step <= keep_step:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        self.completion_marker.unlink(missing_ok=True)
        destination = Path(self.output_paths.get("model", ""))
        if destination.is_symlink() or destination.is_file():
            destination.unlink(missing_ok=True)
        elif destination.is_dir():
            shutil.rmtree(destination)

    def _materialize(self, source: Path) -> None:
        destination = Path(self.output_paths["model"])
        temporary = destination.with_name(f".{destination.name}.tmp")
        if temporary.exists() or temporary.is_symlink():
            if temporary.is_dir() and not temporary.is_symlink():
                shutil.rmtree(temporary)
            else:
                temporary.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, temporary, symlinks=True)
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(temporary, destination)
        spec = self.launch_spec()
        self.completion_marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": "vagen-slime",
                    "source": str(source),
                    "rollout_id": _rollout_id(source),
                    "num_rollout": spec.num_rollout,
                    "config_sha256": _fingerprint(spec),
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
            self._log(f"SLIME trainer exited with code {return_code}")
            return False
        source = self._latest_hf_checkpoint(require_terminal=True)
        if source is None:
            self._state = ModuleState.FAILED
            self._log("SLIME exited without a complete terminal HF checkpoint")
            return False
        if not self._terminal_resume_checkpoint_is_complete():
            self._state = ModuleState.FAILED
            self._log(
                "SLIME exited without a complete terminal checkpoint and rollout prefix"
            )
            return False
        self._materialize(source)
        self._state = ModuleState.DONE
        return True

    def is_already_complete(self) -> bool:
        destination = Path(self.output_paths.get("model", ""))
        if not is_complete_hf_model(destination) or not self.completion_marker.is_file():
            return False
        try:
            marker = json.loads(self.completion_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            marker.get("config_sha256") == _fingerprint(self.launch_spec())
            and self._terminal_resume_checkpoint_is_complete()
        )

    def kill(self) -> None:
        if self._process is not None and self._process.poll() is None:
            kill_process_group(self._process, timeout=10)
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except OSError:
                pass
            self._log_file_handle = None
        if self._state != ModuleState.DONE:
            self._state = ModuleState.TERMINATED

    def get_output(self) -> ModuleOutput:
        path = self.output_paths.get("model")
        return ModuleOutput(model_path=path if path and is_complete_hf_model(path) else None)

    def _forward_output(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        try:
            for line in iter(self._process.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
                if self._log_file_handle is not None and not self._log_file_handle.closed:
                    self._log_file_handle.write(line)
                    self._log_file_handle.flush()
        except Exception:  # pragma: no cover - logging must not kill training
            logger.exception("SLIME log forwarding failed")

    def _log(self, message: str) -> None:
        logger.info("[%s] %s", self.name, message)


__all__ = ["SlimeWrapper"]
