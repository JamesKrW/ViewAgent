"""Completeness checks for per-step rollout data used by adaptive SFT."""

from __future__ import annotations

from pathlib import Path


def rollout_step_is_complete(rollout_dir: str | Path, step: int) -> bool:
    """Return whether one step has both non-empty JSONL and image payloads."""
    root = Path(rollout_dir)
    jsonl = root / f"{int(step)}.jsonl"
    images = root / f"image_{int(step)}"
    try:
        if not jsonl.is_file() or jsonl.stat().st_size <= 0 or not images.is_dir():
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
