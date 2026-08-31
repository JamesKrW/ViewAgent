"""
Iteration-level cleanup with two file-system triggers.

Configured via pipeline.yaml top-level (or per-iter) keys::

    delete_on_sft_model:     [rl_model, rollout_data]   # default
    delete_on_next_rl_model: [sft_model]                # default

Two trigger points, both fired by the controller at the end of each iter (and
re-checked at every subsequent iter end via :func:`process_pending_deletes`):

  * **on_sft_model** — fires once *this* iter has a *real* ``sft_model/``
    (a complete HF dir that is NOT just a symlink to this iter's
    ``rl_model/``). Used to drop SFT inputs (``rl_model``, ``rollout_data``,
    ``sft_data``, ``random_sft_stage``, ``slime_checkpoints``) once SFT has
    consumed them.

  * **on_next_rl_model** — fires once *the next* iter has a *real*
    ``rl_model/``. Used to drop this iter's ``sft_model`` once the next
    iter's RL has produced a fresh resume anchor.

Items that aren't ready yet are recorded in
``iter_XXX/.pending_delete`` (a JSON object with two keys for the two
triggers) and retried each subsequent iter.

The "real model" check is symlink-aware: if a phase was *skipped* and the
controller materialized the output dir as a symlink to another iter's
output, that symlink is **not** considered a real model — so the trigger
won't fire and we never delete the upstream the symlink points at.

Resource-name → relative-path map: short aliases map onto the canonical
sub-paths under ``iter_XXX/``. Anything not in the map is treated as a
literal relative path.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Short aliases users can put in either delete list. Anything else is
# resolved as a literal relative path under iter_XXX/.
RESOURCE_PATHS = {
    # Layout: iter_XXX/<phase>/... — every artefact lives under the phase
    # that produced it.
    "rl_model":         "rl/rl_model",
    "sft_model":        "sft/sft_model",
    "sft_data":         "traj_to_sft/sft_data",
    "rl":               "rl",                    # whole RL working dir
    "sft":              "sft",                   # whole SFT working dir
    "traj_to_sft":      "traj_to_sft",           # whole TrajToSFT scratch
    "rollout_data":     "rl/rollout_data",
    "slime_checkpoints": "rl/slime_checkpoints",
    "verl_checkpoints": "rl/slime_checkpoints",  # deprecated config alias
    "graph":            "traj_to_sft/graph",
    "random_sft_stage": "traj_to_sft/random_sft_stage",
}

# Eligible on the "same-iter sft_model became real" trigger — anything
# UPSTREAM of SFT (RL outputs, TrajToSFT scratch) is safe to drop here.
_TRIGGERED_ON_SFT_MODEL = {
    "rl_model",
    "rollout_data",
    "rl",
    "sft_data",
    "traj_to_sft",         # graph + sft_data_old + reasoning_dump
    "random_sft_stage",
    "slime_checkpoints",
    "verl_checkpoints",
}

# Eligible on the "next iter's rl_model became real" trigger — by then this
# iter has fully done its job and any of its remaining artefacts can go.
_TRIGGERED_ON_NEXT_RL_MODEL = {
    "sft_model",
    "sft",
    "sft_data",
}

_PENDING_FILENAME = ".pending_delete"


# ── helpers ───────────────────────────────────────────────────────────────


def _iter_dir(experiment_dir: Path, iter_num: int) -> Path:
    return experiment_dir / f"iter_{iter_num:03d}"


def _resolve_path(experiment_dir: Path, iter_num: int, resource: str) -> Path:
    rel = RESOURCE_PATHS.get(resource, resource)
    return _iter_dir(experiment_dir, iter_num) / rel


def _model_dir_complete(path: Path) -> bool:
    """A HuggingFace model dir is "complete" only when it has both
    ``config.json`` AND actual weight files (``*.safetensors`` or ``*.bin``).
    Symlinks are followed transparently.
    """
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return bool(any(path.glob("*.safetensors"))) or bool(any(path.glob("*.bin")))


def _is_symlink_aliased(child: Path, sibling: Path) -> bool:
    """``child`` is a symlink and resolves to ``sibling`` (or sibling's target).

    Used to detect "this dir is just a stand-in materialized by the
    controller for a skipped phase" — if so, it does NOT represent real work.
    """
    if not child.is_symlink():
        return False
    try:
        return child.resolve() == sibling.resolve()
    except FileNotFoundError:
        return False


def _real_sft_model(iter_dir: Path) -> bool:
    sft = iter_dir / "sft" / "sft_model"
    if not _model_dir_complete(sft):
        return False
    return not _is_symlink_aliased(sft, iter_dir / "rl" / "rl_model")


def _real_rl_model(iter_dir: Path) -> bool:
    rl = iter_dir / "rl" / "rl_model"
    if not _model_dir_complete(rl):
        return False
    if not rl.is_symlink():
        return True
    # If rl_model is a symlink, it was materialized for a skipped RL phase.
    # That is NOT a real RL output for trigger purposes.
    return False


def _delete(path: Path) -> bool:
    """Delete a file or dir (or dangling symlink). Returns True if anything was deleted."""
    if not path.exists() and not path.is_symlink():
        return False
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path, ignore_errors=True)
        return True
    except Exception as exc:
        logger.warning("[cleanup] failed to delete %s: %s", path, exc)
        return False


def _read_pending(iter_dir: Path) -> dict:
    """Return ``{"on_sft_model": [...], "on_next_rl_model": [...]}``."""
    sidecar = iter_dir / _PENDING_FILENAME
    if not sidecar.is_file():
        return {"on_sft_model": [], "on_next_rl_model": []}
    try:
        data = json.loads(sidecar.read_text())
        if isinstance(data, dict):
            return {
                "on_sft_model": list(data.get("on_sft_model") or []),
                "on_next_rl_model": list(data.get("on_next_rl_model") or []),
            }
        # Legacy: a flat list. Treat as on_sft_model bucket.
        if isinstance(data, list):
            return {"on_sft_model": [str(x) for x in data], "on_next_rl_model": []}
    except Exception as exc:
        logger.warning("[cleanup] %s unreadable (%s); treating as empty", sidecar, exc)
    return {"on_sft_model": [], "on_next_rl_model": []}


def _write_pending(iter_dir: Path, pending: dict) -> None:
    sidecar = iter_dir / _PENDING_FILENAME
    has_any = pending.get("on_sft_model") or pending.get("on_next_rl_model")
    if has_any:
        iter_dir.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(pending, indent=2))
    elif sidecar.is_file():
        sidecar.unlink(missing_ok=True)


# ── public API ────────────────────────────────────────────────────────────


def cleanup_iter(
    iter_num: int,
    experiment_dir: Path,
    delete_on_sft_model: Iterable[str],
    delete_on_next_rl_model: Iterable[str],
) -> None:
    """Schedule end-of-iter deletions per the two trigger lists.

    Items are deleted immediately if their trigger has already fired (i.e.,
    same-iter ``sft_model`` is already real, or next iter's ``rl_model`` is
    already real). Otherwise they're written into ``.pending_delete`` and
    retried by :func:`process_pending_deletes` at every subsequent iter.
    """
    on_sft = _filter_known(delete_on_sft_model, _TRIGGERED_ON_SFT_MODEL, "delete_on_sft_model")
    on_next_rl = _filter_known(
        delete_on_next_rl_model, _TRIGGERED_ON_NEXT_RL_MODEL, "delete_on_next_rl_model"
    )
    if not on_sft and not on_next_rl:
        return

    iter_dir = _iter_dir(experiment_dir, iter_num)
    pending = _read_pending(iter_dir)

    # Merge new requests with anything already pending (deduped, preserving order).
    pending["on_sft_model"] = list(dict.fromkeys(pending["on_sft_model"] + on_sft))
    pending["on_next_rl_model"] = list(dict.fromkeys(pending["on_next_rl_model"] + on_next_rl))

    _try_fire(iter_dir, iter_num, experiment_dir, pending)
    _write_pending(iter_dir, pending)


def process_pending_deletes(experiment_dir: Path) -> None:
    """Scan all ``iter_*/.pending_delete`` and fire any triggers now ready."""
    if not experiment_dir.is_dir():
        return

    for iter_dir in sorted(experiment_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        try:
            iter_num = int(iter_dir.name.split("_")[-1])
        except ValueError:
            continue

        pending = _read_pending(iter_dir)
        if not (pending["on_sft_model"] or pending["on_next_rl_model"]):
            continue

        _try_fire(iter_dir, iter_num, experiment_dir, pending)
        _write_pending(iter_dir, pending)


# ── internals ─────────────────────────────────────────────────────────────


def _filter_known(items: Iterable[str], allowed: set, list_name: str) -> List[str]:
    """Drop unknown / wrong-bucket entries with a warning. Preserves order, dedups."""
    seen = set()
    out: List[str] = []
    for r in items or []:
        if not r:
            continue
        if r not in allowed and r not in RESOURCE_PATHS:
            logger.warning(
                "[cleanup] %s: unknown resource %r — ignoring", list_name, r,
            )
            continue
        if r in RESOURCE_PATHS and r not in allowed:
            logger.warning(
                "[cleanup] %s: %r is not eligible for this trigger "
                "(allowed: %s) — ignoring", list_name, r, sorted(allowed),
            )
            continue
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _try_fire(iter_dir: Path, iter_num: int, experiment_dir: Path, pending: dict) -> None:
    """Mutates ``pending`` in place: removes anything that gets deleted now."""
    # Trigger 1: same-iter sft_model is real
    if _real_sft_model(iter_dir):
        for r in list(pending["on_sft_model"]):
            target = _resolve_path(experiment_dir, iter_num, r)
            if _delete(target):
                logger.info("[cleanup] iter %d: deleted %s (on_sft_model)", iter_num, target)
        pending["on_sft_model"] = []

    # Trigger 2: next iter's rl_model is real
    if _real_rl_model(_iter_dir(experiment_dir, iter_num + 1)):
        for r in list(pending["on_next_rl_model"]):
            target = _resolve_path(experiment_dir, iter_num, r)
            if _delete(target):
                logger.info(
                    "[cleanup] iter %d: deleted %s (on_next_rl_model)", iter_num, target,
                )
        pending["on_next_rl_model"] = []
