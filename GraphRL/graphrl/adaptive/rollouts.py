"""Completeness checks for per-step rollout data used by adaptive SFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def rollout_step_is_complete(rollout_dir: str | Path, step: int) -> bool:
    """Return whether one rollout step was published in full.

    TrajToSFT consumes both ``<step>.jsonl`` and the matching
    ``image_<step>/`` tree. A SLIME-era ``<step>.complete`` marker is also a
    sufficient publication signal because it was written only after both were
    durable. VAGEN/verl does not write that marker, so require at least one
    materialized frame there.
    """
    root = Path(rollout_dir)
    jsonl = root / f"{int(step)}.jsonl"
    marker = root / f"{int(step)}.complete"
    try:
        if not jsonl.is_file() or jsonl.stat().st_size <= 0:
            return False
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip() == "complete"
        images = root / f"image_{int(step)}"
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

    Before the success threshold is latched, every trajectory may feed a later
    TrajToSFT phase, so the checkpoint is only self-consistent when rollout data
    is durable through the same step. Once latched, SFT is permanently disabled
    for the run, so a complete rollout prefix is no longer part of resumability.

    The latch shortcut is load-bearing on VAGEN/verl: image capture is disabled
    after latching because no later SFT phase can consume those trajectories.
    """
    if bool(state.get("latched")):
        return 0
    return max(0, int(checkpoint_step))
