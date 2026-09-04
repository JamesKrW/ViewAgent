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

    Before the success threshold is latched, every trajectory may feed a later
    TrajToSFT phase, so the checkpoint is only self-consistent when rollout data
    is durable through the same step. Once latched, SFT is permanently disabled
    for the run, so a complete rollout prefix is no longer part of resumability.

    ★ The latch shortcut is load-bearing on the VAGEN/verl backend, not a
    micro-optimisation. ``rollout_step_is_complete`` accepts a step on either of
    two signals: a ``<step>.complete`` marker, or a non-empty ``image_<step>/``
    directory. verl writes no marker -- that was SLIME's -- so frames are the only
    signal, and the latch is precisely when the adaptive trainer stops writing
    them. Without this branch the first post-latch checkpoint can never commit:

        RuntimeError: regular checkpoint cannot commit before its rollout payload

    which lands at the single most important moment of the experiment.
    """
    if bool(state.get("latched")):
        return 0
    return max(0, int(checkpoint_step))
