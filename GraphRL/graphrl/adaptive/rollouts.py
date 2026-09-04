"""Completeness checks for per-step rollout data used by adaptive SFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def rollout_step_is_complete(rollout_dir: str | Path, step: int) -> bool:
    """Return whether one rollout step was published in full.

    The payload is ``<step>.jsonl``: it carries the trajectories, and it is what
    the later TrajToSFT phase reads. Frames are not part of it -- the view-graph
    generators take their images from the corpus directory
    (``traj_to_sft.viewsuite_15k_dir``), never from a rollout dump.

    ★ A non-empty JSONL is the whole signal on the VAGEN/verl backend, and an
    earlier revision that also demanded ``image_<step>/`` was wrong. That came
    from SLIME, which published asynchronously and so wrote a ``<step>.complete``
    marker once the JSONL and any enabled frames were durable. verl has no such
    race: ``fit_step`` runs ``_fit_dump_data`` to completion before
    ``_fit_save_checkpoint``, so the file is closed by the time anything asks. It
    also never writes the marker, and it only writes frames when
    ``trainer.log_image.enable`` is set -- which nothing here needs. Requiring
    them meant ``durable_rollout_prefix`` stayed at 0 forever and every run died
    at its first checkpoint:

        RuntimeError: regular checkpoint cannot commit before its rollout
        payload: checkpoint_step=20, required_rollout_step=20,
        durable_rollout_step=0

    The marker branch is kept, so a SLIME-era directory is still read with the
    stronger guarantee where one exists.
    """
    root = Path(rollout_dir)
    jsonl = root / f"{int(step)}.jsonl"
    marker = root / f"{int(step)}.complete"
    try:
        if not jsonl.is_file() or jsonl.stat().st_size <= 0:
            return False
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip() == "complete"
        return True
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

    The latch shortcut is no longer load-bearing: ``rollout_step_is_complete``
    now keys on the JSONL, which is written at every step whether or not frames
    are. It stays because it still states the rule correctly -- once SFT is off
    for good, no trajectory can reach it, so the prefix is not a resumability
    condition at all -- and because it keeps the post-latch path cheap.
    """
    if bool(state.get("latched")):
        return 0
    return max(0, int(checkpoint_step))
