"""Compatibility command builder backed by VAGEN-SLIME.

The public function keeps its historical name for downstream GraphRL code, but
it no longer constructs a ``vagen.main_ppo``/verl command. New code should use
``graphrl.slime.config`` and ``graphrl.slime.launcher`` directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from graphrl.slime.config import build_launch_spec


def build_vagen_command(
    config: dict[str, Any],
    model_path: str,
    output_dir: Path,
) -> list[str]:
    """Build the equivalent VAGEN-SLIME launcher command.

    Existing callers can retain the old function name while receiving the same
    command used by :class:`graphrl.slime.wrapper.SlimeWrapper`.
    """

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = build_launch_spec(
        config,
        model_path=model_path,
        output_dir=output,
        adaptive=bool(config.get("_adaptive", False)),
    )
    config_path = output / "slime_launch.json"
    temporary = config_path.with_name(f".{config_path.name}.tmp")
    temporary.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, config_path)
    return [
        sys.executable,
        "-m",
        "graphrl.slime.launcher",
        "--config",
        str(config_path),
    ]


def _flatten(value: Any, prefix: str) -> list[str]:
    """Retain the old utility for downstream imports; it is not launched."""

    if not isinstance(value, dict):
        return [f"{prefix}={_format_value(value)}"]
    output: list[str] = []
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        output.extend(_flatten(item, child))
    return output


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return str(value).replace(" ", "")
    return str(value)


__all__ = ["build_vagen_command"]
