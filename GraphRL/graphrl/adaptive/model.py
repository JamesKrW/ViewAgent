"""Completeness checks for adaptive model snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

SNAPSHOT_MANIFEST = ".adaptive_snapshot_manifest.json"
CHECKPOINT_MANIFEST = ".adaptive_checkpoint_manifest.json"


def is_complete_hf_model(path: str | Path) -> bool:
    """Return whether all files referenced by a HuggingFace weight index exist."""
    model_dir = Path(path)
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        return False

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_dir / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files = set(index["weight_map"].values())
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return False
        return bool(weight_files) and all(
            (model_dir / name).is_file() and (model_dir / name).stat().st_size > 0
            for name in weight_files
        )

    weights = [*model_dir.glob("*.safetensors"), *model_dir.glob("*.bin")]
    return bool(weights) and all(path.stat().st_size > 0 for path in weights)


def write_snapshot_manifest(snapshot: str | Path) -> Path:
    """Record immutable file sizes so mirrored snapshots can be validated."""
    root = Path(snapshot)
    manifest_path = root / SNAPSHOT_MANIFEST
    files = {
        str(path.relative_to(root)): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(
        json.dumps({"files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_checkpoint_manifest(
    checkpoint: str | Path,
    related_paths: tuple[str | Path, ...] = (),
) -> Path:
    """Atomically record every file required to resume an RL checkpoint.

    SLIME stores the actor, critic and global rollout-dataset cursor in sibling
    locations.  ``related_paths`` lets one commit manifest cover that complete
    resume bundle instead of declaring the actor directory complete by itself.
    """
    root = Path(checkpoint)
    manifest_path = root / CHECKPOINT_MANIFEST
    candidates = [path for path in sorted(root.rglob("*")) if path.is_file()]
    for related in related_paths:
        path = Path(related)
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(item for item in sorted(path.rglob("*")) if item.is_file())
    files = {
        os.path.relpath(path, root): path.stat().st_size
        for path in candidates
        if path != manifest_path
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", dir=str(root)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"files": files}, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, manifest_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return manifest_path


def is_complete_checkpoint_manifest(checkpoint: str | Path) -> bool:
    """Verify every file captured after checkpoint commit still has its full size."""
    root = Path(checkpoint)
    manifest_path = root / CHECKPOINT_MANIFEST
    try:
        files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if not isinstance(files, dict) or not files:
        return False
    for relative, expected_size in files.items():
        path = root / relative
        try:
            if not path.is_file() or path.stat().st_size != int(expected_size):
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True


def is_complete_snapshot(snapshot: str | Path) -> bool:
    """Validate a best snapshot, including every file recorded in its manifest."""
    root = Path(snapshot)
    if not is_complete_hf_model(root / "actor" / "huggingface"):
        return False
    manifest_path = root / SNAPSHOT_MANIFEST
    try:
        files = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if not isinstance(files, dict) or not files:
        return False
    for relative, expected_size in files.items():
        path = root / relative
        try:
            if not path.is_file() or path.stat().st_size != int(expected_size):
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True
