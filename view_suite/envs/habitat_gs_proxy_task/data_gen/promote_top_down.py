#!/usr/bin/env python3
"""Promote reviewed top-down candidates into release dataset trees.

Only candidates explicitly marked ``keep`` are promoted.  Images are replaced
atomically and every JSONL/sample metadata reference for the affected scene is
updated with the candidate camera pose and intrinsics.  The command defaults to
a read-only dry run; pass ``--apply`` to write the selected dataset trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASKS = ("path_to_view", "view_to_path", "interactive_view_planning")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    relative = Path(os.path.normpath(value))
    if relative.is_absolute() or relative == Path("..") or ".." in relative.parts:
        raise ValueError(f"path escapes dataset root: {value!r}")
    return relative


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


@dataclass(frozen=True)
class Candidate:
    item_id: str
    corpus: str
    scene_id: str
    source: Path
    destination_relative: Path
    sha256: str
    parent_sha256: str
    camera_pose: list[list[float]]
    camera_intrinsics: list[list[float]]
    method: str


def load_reviewed_candidates(
    labels_path: Path,
    manifest_path: Path,
) -> tuple[list[Candidate], list[dict[str, str]]]:
    labels_path = labels_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review_round = int(labels.get("review_round", -1))
    if int(manifest.get("round", -2)) != review_round:
        raise ValueError("labels and candidate manifest belong to different rounds")

    manifest_items = {str(item["id"]): item for item in manifest["items"]}
    selected: list[Candidate] = []
    rejected: list[dict[str, str]] = []
    for label in labels["items"]:
        verdict = label.get("verdict")
        if verdict not in {"keep", "reject"}:
            raise ValueError(
                f"item {label.get('id')} has unresolved verdict {verdict!r}"
            )
        item_id = str(label["id"])
        if verdict == "reject":
            rejected.append(
                {
                    "id": item_id,
                    "corpus": str(label["corpus"]),
                    "scene_id": str(label["scene_id"]),
                    "sha256": str(label["sha256"]),
                }
            )
            continue

        item = manifest_items.get(item_id)
        if item is None:
            raise KeyError(f"kept item missing from candidate manifest: {item_id}")
        if item.get("corpus") != label.get("corpus") or item.get(
            "scene_id"
        ) != label.get("scene_id"):
            raise ValueError(f"labels/manifest identity mismatch for {item_id}")
        source = Path(item["image_path"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        source = source.resolve(strict=True)
        expected_hash = str(label["sha256"])
        actual_hash = _sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(
                f"candidate hash mismatch for {item_id}: {actual_hash} != {expected_hash}"
            )
        lineage = item.get("lineage") or {}
        selected.append(
            Candidate(
                item_id=item_id,
                corpus=str(item["corpus"]),
                scene_id=str(item["scene_id"]),
                source=source,
                destination_relative=_safe_relative(lineage["parent_image_rel"]),
                sha256=expected_hash,
                parent_sha256=str(lineage["parent_sha256"]),
                camera_pose=lineage["camera_pose"],
                camera_intrinsics=lineage["camera_intrinsics"],
                method=str(lineage.get("method", "unknown")),
            )
        )
    return selected, rejected


def _jsonl_paths(root: Path) -> Iterable[Path]:
    found: set[Path] = set()
    for task in TASKS:
        for path in root.glob(f"{task}*.jsonl*"):
            if path.is_file() and ".jsonl" in path.name:
                found.add(path)
    return sorted(found)


def _update_top_down_mapping(mapping: dict[str, Any], candidate: Candidate) -> bool:
    changed = False
    replacements = {
        "c2w_extrinsics": candidate.camera_pose,
        "c2w_intrinsics": candidate.camera_intrinsics,
    }
    for key, value in replacements.items():
        if mapping.get(key) != value:
            mapping[key] = value
            changed = True
    return changed


def _update_jsonl(path: Path, candidates: dict[str, Candidate], apply: bool) -> int:
    output: list[str] = []
    changed_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, original in enumerate(handle, 1):
            if not original.strip():
                output.append(original)
                continue
            try:
                row = json.loads(original)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            candidate = candidates.get(str(row.get("scene_id")))
            if candidate is None:
                output.append(original)
                continue
            top_down = (row.get("image_detail") or {}).get("top_down_view")
            if not isinstance(top_down, dict):
                raise TypeError(
                    f"{path}:{line_number}: missing image_detail.top_down_view"
                )
            if _update_top_down_mapping(top_down, candidate):
                changed_rows += 1
                output.append(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                output.append(original)
    if apply and changed_rows:
        _atomic_write_text(path, "".join(output))
    return changed_rows


def _update_sidecars(root: Path, candidates: dict[str, Candidate], apply: bool) -> int:
    changed_files = 0
    for scene_id, candidate in candidates.items():
        scene_root = root / scene_id
        paths = list(scene_root.glob("sample_*/meta.json"))
        top_down_meta = scene_root / "top_down_meta.json"
        if top_down_meta.is_file():
            paths.append(top_down_meta)
        for path in sorted(paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "top_down_meta.json":
                mapping = payload
                key_map = {
                    "pose_c2w": candidate.camera_pose,
                    "intrinsics": candidate.camera_intrinsics,
                }
                changed = False
                for key, value in key_map.items():
                    if mapping.get(key) != value:
                        mapping[key] = value
                        changed = True
            else:
                mapping = payload.get("top_down")
                if not isinstance(mapping, dict):
                    continue
                changed = False
                for key, value in {
                    "pose_c2w": candidate.camera_pose,
                    "intrinsics": candidate.camera_intrinsics,
                }.items():
                    if mapping.get(key) != value:
                        mapping[key] = value
                        changed = True
            if changed:
                changed_files += 1
                if apply:
                    _atomic_write_text(
                        path,
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                        + "\n",
                    )
    return changed_files


def promote_root(
    *,
    corpus: str,
    root: Path,
    candidates: list[Candidate],
    rejected: list[dict[str, str]],
    labels_path: Path,
    manifest_path: Path,
    apply: bool,
    metadata_only: bool = False,
    allow_diverged_images: bool = False,
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    selected = {item.scene_id: item for item in candidates if item.corpus == corpus}
    if not selected:
        raise ValueError(f"no kept candidates for corpus {corpus!r}")

    image_replaced = 0
    image_already_current = 0
    image_missing = 0
    diverged_images: list[dict[str, str]] = []
    if not metadata_only:
        for candidate in selected.values():
            destination = root / candidate.destination_relative
            if not destination.is_file():
                image_missing += 1
                raise FileNotFoundError(destination)
            current_hash = _sha256(destination)
            if current_hash == candidate.sha256:
                image_already_current += 1
                continue
            if current_hash != candidate.parent_sha256:
                diverged_images.append(
                    {
                        "scene_id": candidate.scene_id,
                        "existing_sha256": current_hash,
                        "review_parent_sha256": candidate.parent_sha256,
                    }
                )
                if not allow_diverged_images:
                    raise ValueError(
                        f"destination changed since review for {candidate.item_id}: "
                        f"{current_hash} != {candidate.parent_sha256}"
                    )
            image_replaced += 1
            if apply:
                _atomic_copy(candidate.source, destination)

    changed_jsonl_rows = sum(
        _update_jsonl(path, selected, apply) for path in _jsonl_paths(root)
    )
    changed_sidecars = _update_sidecars(root, selected, apply)
    rejected_for_corpus = [item for item in rejected if item["corpus"] == corpus]
    report = {
        "format_version": 1,
        "review_round": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": corpus,
        "dataset_root": str(root),
        "source_labels": str(labels_path.expanduser().resolve()),
        "source_manifest": str(manifest_path.expanduser().resolve()),
        "mode": "apply" if apply else "dry-run",
        "metadata_only": metadata_only,
        "kept_candidates": len(selected),
        "image_replaced": image_replaced,
        "image_already_current": image_already_current,
        "image_missing": image_missing,
        "jsonl_rows_updated": changed_jsonl_rows,
        "sidecar_files_updated": changed_sidecars,
        "diverged_images": diverged_images,
        "rejected_candidates_preserved": rejected_for_corpus,
        "promoted": [
            {
                **asdict(item),
                "source": str(item.source),
                "destination_relative": item.destination_relative.as_posix(),
            }
            for item in selected.values()
        ],
    }
    if apply:
        _atomic_write_text(
            root / "top_down_promotion.json",
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
    return report


def _parse_mapping(value: str) -> tuple[str, Path]:
    corpus, separator, path = value.partition("=")
    if not separator or not corpus or not path:
        raise argparse.ArgumentTypeError("expected CORPUS=/absolute/or/relative/path")
    return corpus, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_mapping,
        required=True,
        metavar="CORPUS=PATH",
        help="Dataset root whose images and metadata should be promoted.",
    )
    parser.add_argument(
        "--metadata-only",
        action="append",
        type=_parse_mapping,
        default=[],
        metavar="CORPUS=PATH",
        help="Additional root whose JSONL/sidecar metadata should be updated.",
    )
    parser.add_argument(
        "--allow-diverged-images",
        action="append",
        default=[],
        metavar="CORPUS",
        help="Allow replacing a destination that no longer has its reviewed parent hash.",
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    candidates, rejected = load_reviewed_candidates(args.labels, args.manifest)
    reports = []
    for corpus, root in args.dataset:
        reports.append(
            promote_root(
                corpus=corpus,
                root=root,
                candidates=candidates,
                rejected=rejected,
                labels_path=args.labels,
                manifest_path=args.manifest,
                apply=args.apply,
                allow_diverged_images=corpus in args.allow_diverged_images,
            )
        )
    for corpus, root in args.metadata_only:
        reports.append(
            promote_root(
                corpus=corpus,
                root=root,
                candidates=candidates,
                rejected=rejected,
                labels_path=args.labels,
                manifest_path=args.manifest,
                apply=args.apply,
                metadata_only=True,
            )
        )
    print(json.dumps({"applied": args.apply, "datasets": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
