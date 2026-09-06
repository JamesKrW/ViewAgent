#!/usr/bin/env python3
"""Build a clean, standalone proxy-task dataset tree.

The release layout intentionally matches ``data/viewsuite_15k``: one dataset
directory containing the three task JSONLs for train/validation/test and scene
directories containing every referenced image.  Generation-only files (shards,
logs, pre-filter JSONLs, verdicts, caches, and combined JSONLs) are not copied.

For ground-plane datasets, cached option renders are materialized back into the
usual ``<scene>/<sample>/option_NNN.png`` locations and JSONL paths are rewritten
accordingly.  Files are hard-linked where possible, including duplicate cached
renders, so staging does not duplicate their bytes and GNU tar can preserve the
deduplication inside the archive.
"""

from __future__ import annotations

import argparse
import copy
import filecmp
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any


TASKS = ("path_to_view", "view_to_path", "interactive_view_planning")
LABEL_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc


def _relative(value: str | Path) -> Path:
    path = Path(os.path.normpath(str(value)))
    if path.is_absolute() or path == Path("..") or ".." in path.parts:
        raise ValueError(f"path escapes dataset root: {value!r}")
    return path


def _source_file(root: Path, relative: str | Path) -> Path:
    lexical_path = root / _relative(relative)
    if os.path.commonpath((str(root), str(lexical_path))) != str(root):
        raise ValueError(f"source path escapes dataset root: {relative!r}")
    path = lexical_path.resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class Materializer:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self._source_by_destination: dict[Path, Path] = {}
        self._destination_by_inode: dict[tuple[int, int], Path] = {}
        self.requested = 0
        self.created = 0
        self.reused = 0
        self.hardlinked_duplicates = 0
        self.copied_cross_filesystem = 0

    def add(
        self,
        source_root: Path,
        source_relative: str | Path,
        destination_relative: str | Path | None = None,
    ) -> None:
        self.requested += 1
        source = _source_file(source_root, source_relative)
        destination_rel = _relative(
            source_relative if destination_relative is None else destination_relative
        )
        destination = self.root / destination_rel

        previous = self._source_by_destination.get(destination)
        if previous is not None:
            if previous != source and not filecmp.cmp(previous, source, shallow=False):
                raise RuntimeError(
                    f"destination collision at {destination}: {previous} != {source}"
                )
            self.reused += 1
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        stat = source.stat()
        inode = (stat.st_dev, stat.st_ino)
        first_destination = self._destination_by_inode.get(inode)
        try:
            os.link(first_destination or source, destination)
            if first_destination is not None:
                self.hardlinked_duplicates += 1
        except OSError:
            shutil.copy2(source, destination)
            self.copied_cross_filesystem += 1
        self._destination_by_inode.setdefault(inode, destination)
        self._source_by_destination[destination] = source
        self.created += 1


def _discover_jsonls(source: Path) -> tuple[str, list[Path]]:
    validation_split = "dev" if all(
        (source / f"{task}_dev.jsonl").is_file() for task in TASKS
    ) else "eval"
    splits = ("train", validation_split, "test")
    paths = [source / f"{task}_{split}.jsonl" for task in TASKS for split in splits]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing release JSONLs: {missing}")
    return validation_split, paths


def _copy_sidecars(
    materializer: Materializer,
    source: Path,
    base_source: Path,
    row: dict[str, Any],
) -> None:
    details = row.get("image_detail") or {}
    init = details.get("init_view") or {}
    init_path = init.get("path") or (row.get("image_path") or [None])[0]
    if not init_path:
        return
    sample_dir = _relative(init_path).parent
    scene_dir = Path(str(row["scene_id"]))
    for relative in (
        sample_dir / "meta.json",
        scene_dir / "meta.json",
        scene_dir / "top_down_meta.json",
    ):
        for candidate_root in (source, base_source):
            try:
                candidate = _source_file(candidate_root, relative)
            except FileNotFoundError:
                continue
            materializer.add(candidate_root, relative, relative)
            break


def _copy_row_images(
    materializer: Materializer,
    source: Path,
    row: dict[str, Any],
) -> None:
    paths = list(row.get("image_path") or [])
    for detail in (row.get("image_detail") or {}).values():
        if isinstance(detail, dict) and detail.get("path"):
            paths.append(str(detail["path"]))
    for relative in paths:
        materializer.add(source, relative)


def _ground_plane_row(
    materializer: Materializer,
    source: Path,
    row: dict[str, Any],
    task: str,
    target_indices: dict[str, int],
) -> dict[str, Any]:
    result = copy.deepcopy(row)
    details = result.get("image_detail") or {}
    original_details = row.get("image_detail") or {}
    image_paths = list(row.get("image_path") or [])
    if len(image_paths) < 3:
        raise ValueError(f"{row.get('sample_id')}: expected at least three images")

    init_path = str(original_details["init_view"]["path"])
    top_down_path = str(original_details["top_down_view"]["path"])
    sample_dir = _relative(init_path).parent
    option_paths = [str(sample_dir / f"option_{index:03d}.png") for index in range(4)]

    materializer.add(source, init_path, init_path)
    materializer.add(source, top_down_path, top_down_path)
    details["init_view"]["path"] = init_path
    details["top_down_view"]["path"] = top_down_path

    sample_id = str(row.get("sample_id"))
    label = str((row.get("meta") or {}).get("gt_label", "")).upper()
    if label in LABEL_TO_INDEX:
        target_index = LABEL_TO_INDEX[label]
        previous = target_indices.setdefault(sample_id, target_index)
        if previous != target_index:
            raise RuntimeError(f"{sample_id}: conflicting target option indices")
    elif sample_id in target_indices:
        target_index = target_indices[sample_id]
    else:
        raise ValueError(f"{sample_id}: target option cannot be determined")

    if task == "path_to_view":
        for index, destination in enumerate(option_paths):
            key = f"view_{index}"
            old_path = str(original_details[key]["path"])
            materializer.add(source, old_path, destination)
            details[key]["path"] = destination
        result["image_path"] = [init_path, top_down_path, *option_paths]
        if "target_view" in details:
            details["target_view"]["path"] = option_paths[target_index]
    else:
        old_target = str(original_details["target_view"]["path"])
        target_path = option_paths[target_index]
        materializer.add(source, old_target, target_path)
        details["target_view"]["path"] = target_path
        result["image_path"] = [init_path, top_down_path, target_path]

    result["image_detail"] = details
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate(root: Path, expected_jsonls: list[str], canonical: bool) -> dict[str, Any]:
    root_files = sorted(path.name for path in root.iterdir() if path.is_file())
    if root_files != sorted(expected_jsonls):
        raise RuntimeError(
            f"unexpected root files: expected {sorted(expected_jsonls)}, got {root_files}"
        )

    row_counts: dict[str, int] = {}
    sample_ids: dict[tuple[str, str], set[str]] = {}
    references = 0
    for name in expected_jsonls:
        task, split = next(
            (task, name[len(task) + 1 : -len(".jsonl")])
            for task in TASKS
            if name.startswith(f"{task}_")
        )
        rows = list(_iter_jsonl(root / name))
        row_counts[name] = len(rows)
        sample_ids[(task, split)] = {str(row["sample_id"]) for row in rows}
        for row in rows:
            paths = list(row.get("image_path") or [])
            for detail in (row.get("image_detail") or {}).values():
                if isinstance(detail, dict) and detail.get("path"):
                    paths.append(str(detail["path"]))
            for relative in paths:
                references += 1
                if not (root / _relative(relative)).is_file():
                    raise FileNotFoundError(f"{name}: {relative}")
                if canonical and str(relative).startswith("ground_plane_cache/"):
                    raise RuntimeError(f"cached path remains in {name}: {relative}")

    splits = {split for _, split in sample_ids}
    for split in splits:
        ids = [sample_ids[(task, split)] for task in TASKS]
        if not all(value == ids[0] for value in ids[1:]):
            raise RuntimeError(f"task sample IDs differ in split {split}")

    symlinks = [str(path) for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"standalone tree contains symlinks: {symlinks[:10]}")
    if canonical and (root / "ground_plane_cache").exists():
        raise RuntimeError("ground_plane_cache should not be present in release tree")
    return {
        "row_counts": row_counts,
        "references_checked": references,
        "files": sum(path.is_file() for path in root.rglob("*")),
        "symlinks": 0,
    }


def package(
    source: Path,
    destination: Path,
    *,
    base_source: Path | None = None,
    canonicalize_ground_plane: bool = False,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    base_source = (base_source or source).resolve(strict=True)
    validation_split, jsonls = _discover_jsonls(source)
    materializer = Materializer(destination)
    expected_names: list[str] = []
    target_indices: dict[str, int] = {}

    for jsonl_path in jsonls:
        task = next(task for task in TASKS if jsonl_path.name.startswith(f"{task}_"))
        expected_names.append(jsonl_path.name)
        if canonicalize_ground_plane:
            rows = []
            for row in _iter_jsonl(jsonl_path):
                converted = _ground_plane_row(
                    materializer, source, row, task, target_indices
                )
                rows.append(converted)
            _write_jsonl(destination / jsonl_path.name, rows)
        else:
            materializer.add(source, jsonl_path.name, jsonl_path.name)
            for row in _iter_jsonl(jsonl_path):
                _copy_row_images(materializer, source, row)
                _copy_sidecars(materializer, source, base_source, row)

    validation = _validate(destination, expected_names, canonicalize_ground_plane)
    return {
        "source": str(source),
        "destination": str(destination.resolve()),
        "validation_split": validation_split,
        "materialization": {
            "requested": materializer.requested,
            "created": materializer.created,
            "reused": materializer.reused,
            "hardlinked_duplicates": materializer.hardlinked_duplicates,
            "copied_cross_filesystem": materializer.copied_cross_filesystem,
        },
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--canonicalize-ground-plane", action="store_true")
    args = parser.parse_args()
    result = package(
        args.source,
        args.destination,
        base_source=args.base_source,
        canonicalize_ground_plane=args.canonicalize_ground_plane,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
