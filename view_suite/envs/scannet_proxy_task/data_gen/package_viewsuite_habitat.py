#!/usr/bin/env python3
"""Materialize the Habitat ViewSuite datasets as independent directory trees.

The generated datasets use a shared pose cache while they are being produced.  That
layout is space-efficient locally, but directly archiving it either stores broken
symlinks or (with tar --dereference) pulls the entire global cache into every archive.

This utility creates compact standalone trees:

* base: original ViewSuite-style scene/sample layout, without ``pose_cache``;
* P2V/V2P: only the base images and intermediate pose images referenced by that
  task's JSONLs, with no links outside the standalone tree.

Files are hard-linked while staging when possible.  GNU tar preserves those hardlink
relationships inside a tarball, so the resulting archive is independent without
storing duplicate image bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_SPLITS = ("train", "dev", "test")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _normalized_relative_path(value: str) -> Path:
    normalized = Path(os.path.normpath(value))
    if normalized.is_absolute() or normalized == Path("..") or ".." in normalized.parts:
        raise ValueError(f"path escapes dataset root: {value!r}")
    return normalized


def _safe_destination(root: Path, relative: str | Path) -> Path:
    rel = _normalized_relative_path(str(relative))
    destination = root / rel
    if os.path.commonpath((str(root), str(destination))) != str(root):
        raise ValueError(f"destination escapes dataset root: {relative!r}")
    return destination


def _flatten_intermediate_paths(value: Any) -> Iterator[str]:
    if isinstance(value, list):
        for item in value:
            yield str(item)
    elif isinstance(value, dict):
        for label in sorted(value):
            yield from _flatten_intermediate_paths(value[label])
    else:
        raise TypeError(f"unexpected intermediate path container: {type(value)}")


@dataclass
class MaterializeStats:
    requested: int = 0
    created: int = 0
    reused_destination: int = 0
    hardlinked_duplicate: int = 0
    copied_cross_filesystem: int = 0


class Materializer:
    def __init__(self, destination_root: Path):
        self.root = destination_root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.stats = MaterializeStats()
        self._source_by_destination: dict[Path, tuple[int, int]] = {}
        self._first_destination_by_inode: dict[tuple[int, int], Path] = {}

    def add(self, source: Path, relative_destination: str | Path) -> Path:
        self.stats.requested += 1
        source = source.resolve(strict=True)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = _safe_destination(self.root, relative_destination)
        source_stat = source.stat()
        source_key = (source_stat.st_dev, source_stat.st_ino)

        previous_key = self._source_by_destination.get(destination)
        if previous_key is not None:
            if previous_key != source_key:
                raise RuntimeError(
                    f"destination collision: {destination} maps to two source files"
                )
            self.stats.reused_destination += 1
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        first_destination = self._first_destination_by_inode.get(source_key)
        try:
            if first_destination is not None:
                os.link(first_destination, destination)
                self.stats.hardlinked_duplicate += 1
            else:
                os.link(source, destination)
                self._first_destination_by_inode[source_key] = destination
        except OSError:
            shutil.copy2(source, destination)
            self.stats.copied_cross_filesystem += 1
            if first_destination is None:
                self._first_destination_by_inode[source_key] = destination

        self._source_by_destination[destination] = source_key
        self.stats.created += 1
        return destination


def _tree_stats(root: Path) -> dict[str, int]:
    file_count = 0
    png_count = 0
    jsonl_count = 0
    logical_bytes = 0
    unique_inodes: dict[tuple[int, int], int] = {}
    symlink_count = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            symlink_count += 1
            continue
        if not path.is_file():
            continue
        stat = path.stat()
        file_count += 1
        logical_bytes += stat.st_size
        unique_inodes.setdefault((stat.st_dev, stat.st_ino), stat.st_size)
        png_count += path.suffix.lower() == ".png"
        jsonl_count += path.suffix.lower() == ".jsonl"
    return {
        "files": file_count,
        "pngs": png_count,
        "jsonls": jsonl_count,
        "logical_bytes": logical_bytes,
        "physical_bytes": sum(unique_inodes.values()),
        "unique_inodes": len(unique_inodes),
        "symlinks": symlink_count,
    }


def _write_readme(root: Path, kind: str) -> None:
    text = (
        "# Standalone ViewSuite15K Habitat dataset\n\n"
        f"Dataset kind: `{kind}`.\n\n"
        "This directory is self-contained. It has no symlinks to another dataset. "
        "Repeated files may be hard-linked internally to avoid duplicate bytes; tar "
        "preserves those links inside the archive.\n"
    )
    _atomic_write_text(root / "README.md", text)


def _write_manifest(
    root: Path,
    *,
    kind: str,
    source_root: Path,
    materializer: Materializer,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = _tree_stats(root)
    manifest = {
        "format_version": 1,
        "standalone": True,
        "dataset_kind": kind,
        "source_root": str(source_root.resolve()),
        "contains_external_symlinks": False,
        "tree": stats,
        "materialization": vars(materializer.stats),
        **(extra or {}),
    }
    _atomic_write_text(
        root / "dataset_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    # Recompute once so the manifest itself is included in the final counts.
    manifest["tree"] = _tree_stats(root)
    _atomic_write_text(
        root / "dataset_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest


def materialize_base(base_root: Path, destination: Path) -> dict[str, Any]:
    materializer = Materializer(destination)
    excluded_top_level = {
        "pose_cache",
        "pose_index_base.jsonl",
        "dataset_manifest.json",
        "validation_report.json",
    }
    for source in sorted(base_root.rglob("*")):
        relative = source.relative_to(base_root)
        if relative.parts and relative.parts[0] in excluded_top_level:
            continue
        if source.is_symlink():
            raise RuntimeError(f"unexpected symlink in base dataset: {source}")
        if source.is_file():
            materializer.add(source, relative)
    _write_readme(destination, "base")
    return _write_manifest(
        destination,
        kind="base",
        source_root=base_root,
        materializer=materializer,
        extra={"pose_cache_included": False},
    )


def _copy_sample_sidecars(
    materializer: Materializer,
    base_root: Path,
    row: dict[str, Any],
) -> None:
    first_image = _normalized_relative_path(row["image_path"][0])
    sample_dir = first_image.parent
    sample_meta = base_root / sample_dir / "meta.json"
    if sample_meta.is_file():
        materializer.add(sample_meta, sample_dir / "meta.json")
    scene_id = str(row["scene_id"])
    topdown_meta = base_root / scene_id / "top_down_meta.json"
    if topdown_meta.is_file():
        materializer.add(topdown_meta, Path(scene_id) / "top_down_meta.json")


def materialize_intermediate(
    *,
    task: str,
    base_root: Path,
    intermediate_root: Path,
    destination: Path,
) -> dict[str, Any]:
    materializer = Materializer(destination)
    row_count = 0
    base_image_references = 0
    intermediate_references = 0

    jsonl_names = [f"{task}_{split}.jsonl" for split in TASK_SPLITS]
    for name in jsonl_names:
        source_jsonl = intermediate_root / name
        materializer.add(source_jsonl, name)
        for row in _iter_jsonl(source_jsonl):
            row_count += 1
            for relative in row["image_path"]:
                normalized = _normalized_relative_path(relative)
                materializer.add(base_root / normalized, normalized)
                base_image_references += 1
            _copy_sample_sidecars(materializer, base_root, row)
            for relative in _flatten_intermediate_paths(row["intermediate_image_path"]):
                normalized = _normalized_relative_path(relative)
                materializer.add(intermediate_root / normalized, normalized)
                intermediate_references += 1

    pose_index_name = (
        "pose_index_p2v.jsonl" if task == "path_to_view" else "pose_index_v2p.jsonl"
    )
    pose_index = intermediate_root / pose_index_name
    if pose_index.is_file():
        materializer.add(pose_index, pose_index_name)

    base_promotion_manifest = base_root / "top_down_promotion.json"
    intermediate_promotion_manifest = intermediate_root / "top_down_promotion.json"
    if intermediate_promotion_manifest.is_file():
        # The primary report must describe the JSONLs in this intermediate
        # dataset.  Keep the base report as separate provenance for the
        # top-down images and sample sidecars copied from ``base_root``.
        materializer.add(
            intermediate_promotion_manifest,
            "top_down_promotion.json",
        )
        if base_promotion_manifest.is_file():
            materializer.add(
                base_promotion_manifest,
                "top_down_base_promotion.json",
            )
    elif base_promotion_manifest.is_file():
        # Older intermediate trees may not have their own promotion report.
        materializer.add(base_promotion_manifest, "top_down_promotion.json")

    kind = "p2v-intermediate" if task == "path_to_view" else "v2p-intermediate"
    _write_readme(destination, kind)
    return _write_manifest(
        destination,
        kind=kind,
        source_root=intermediate_root,
        materializer=materializer,
        extra={
            "task": task,
            "rows": row_count,
            "base_image_references": base_image_references,
            "intermediate_image_references": intermediate_references,
        },
    )


def validate_standalone(root: Path, task: str | None) -> dict[str, Any]:
    missing: list[str] = []
    rows = 0
    references = 0
    jsonl_paths: Iterable[Path]
    if task is None:
        jsonl_paths = sorted(
            path
            for path in root.glob("*.jsonl")
            if path.name.startswith(
                ("path_to_view_", "view_to_path_", "interactive_view_planning_")
            )
        )
    else:
        jsonl_paths = [root / f"{task}_{split}.jsonl" for split in TASK_SPLITS]

    for jsonl_path in jsonl_paths:
        for row in _iter_jsonl(jsonl_path):
            rows += 1
            for relative in row["image_path"]:
                references += 1
                if not (root / _normalized_relative_path(relative)).is_file():
                    missing.append(f"{jsonl_path.name}:{relative}")
            if task is not None:
                for relative in _flatten_intermediate_paths(
                    row["intermediate_image_path"]
                ):
                    references += 1
                    if not (root / _normalized_relative_path(relative)).is_file():
                        missing.append(f"{jsonl_path.name}:{relative}")

    tree = _tree_stats(root)
    result = {
        "ok": not missing and tree["symlinks"] == 0,
        "rows": rows,
        "references_checked": references,
        "missing": missing[:20],
        "tree": tree,
    }
    if not result["ok"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    _atomic_write_text(
        root / "validation_report.json",
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--p2v-root", required=True)
    parser.add_argument("--v2p-root", required=True)
    parser.add_argument("--standalone-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_root = Path(args.base_root).resolve()
    p2v_root = Path(args.p2v_root).resolve()
    v2p_root = Path(args.v2p_root).resolve()
    standalone_root = Path(args.standalone_root).resolve()
    standalone_root.mkdir(parents=True, exist_ok=False)

    outputs = {
        "base": standalone_root / "viewagent15k_scannet_habitat",
        "p2v": standalone_root / "viewagent15k_scannet_habitat_p2v_intermediate",
        "v2p": standalone_root / "viewagent15k_scannet_habitat_v2p_intermediate",
    }
    summaries = {
        "base": materialize_base(base_root, outputs["base"]),
        "p2v": materialize_intermediate(
            task="path_to_view",
            base_root=base_root,
            intermediate_root=p2v_root,
            destination=outputs["p2v"],
        ),
        "v2p": materialize_intermediate(
            task="view_to_path",
            base_root=base_root,
            intermediate_root=v2p_root,
            destination=outputs["v2p"],
        ),
    }
    validations = {
        "base": validate_standalone(outputs["base"], None),
        "p2v": validate_standalone(outputs["p2v"], "path_to_view"),
        "v2p": validate_standalone(outputs["v2p"], "view_to_path"),
    }
    print(json.dumps({"summaries": summaries, "validations": validations}, indent=2))


if __name__ == "__main__":
    main()
