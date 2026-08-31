"""Completeness checks for per-step rollout data used by adaptive SFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def rollout_step_is_complete(rollout_dir: str | Path, step: int) -> bool:
    """Return whether one rollout step was atomically published.

    New SLIME runs publish ``<step>.complete`` only after their JSONL and any
    enabled images are durable.  Images are intentionally optional after the
    adaptive latch, while JSONL remains available for diagnostics.  The image
    check is retained as a fallback for older VAGEN experiment directories.
    """
    root = Path(rollout_dir)
    jsonl = root / f"{int(step)}.jsonl"
    images = root / f"image_{int(step)}"
    marker = root / f"{int(step)}.complete"
    try:
        if not jsonl.is_file() or jsonl.stat().st_size <= 0:
            return False
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip() == "complete"
        if not images.is_dir():
            return False
        return any(path.is_file() and path.stat().st_size > 0 for path in images.rglob("*"))
    except OSError:
        return False


def durable_rollout_prefix(rollout_dir: str | Path) -> int:
    """Return the largest N for which every rollout step 1..N is complete."""
    step = 1
    while rollout_step_is_complete(rollout_dir, step):
        step += 1
    return step - 1


def required_rollout_prefix(
    state: dict[str, Any], checkpoint_step: int
) -> int:
    """Return the rollout prefix needed to resume one adaptive checkpoint.

    Every checkpoint must have the matching JSONL/commit marker even after the
    success latch.  The latch disables image capture only; it does not disable
    rollout accounting, W&B validation, or the durable JSONL audit trail.
    """
    return max(0, int(checkpoint_step))
