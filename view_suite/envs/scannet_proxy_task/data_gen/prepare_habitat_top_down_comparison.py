"""Build a side-by-side browser inventory of ViewSuite and Habitat-Sim top-downs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _panel(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.pad(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
        color=(245, 245, 245),
    )


def _comparison_image(
    original: Path, habitat: Path, output: Path, *, panel_size: int = 512
) -> None:
    header_height = 42
    canvas = Image.new(
        "RGB", (panel_size * 2, panel_size + header_height), (24, 27, 32)
    )
    with Image.open(original) as image:
        canvas.paste(_panel(image, panel_size), (0, header_height))
    with Image.open(habitat) as image:
        canvas.paste(_panel(image, panel_size), (panel_size, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 13), "LEFT: current ViewSuite", fill=(235, 238, 243))
    draw.text(
        (panel_size + 14, 13),
        "RIGHT: Habitat-Sim rerender",
        fill=(235, 238, 243),
    )
    draw.line(
        (panel_size, 0, panel_size, panel_size + header_height),
        fill=(80, 86, 96),
        width=2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    original_scenes = {
        path.parent.name: path
        for path in args.original_root.glob("*/top_down_view.png")
    }
    habitat_scenes = {
        path.parent.name: path for path in args.habitat_root.glob("*/top_down_view.png")
    }
    scene_ids = sorted(set(original_scenes) & set(habitat_scenes))
    if args.only_round_1_rejects:
        labels = json.loads(args.round_1_labels.read_text(encoding="utf-8"))
        rejected = {
            str(item["scene_id"])
            for item in labels.get("items", [])
            if item.get("corpus") == "viewsuite" and item.get("verdict") == "reject"
        }
        scene_ids = [scene_id for scene_id in scene_ids if scene_id in rejected]

    items = []
    for index, scene_id in enumerate(scene_ids, 1):
        print(f"[{index}/{len(scene_ids)}] {scene_id}", flush=True)
        original = original_scenes[scene_id]
        habitat = habitat_scenes[scene_id]
        output = args.output_dir / f"{scene_id}.png"
        _comparison_image(original, habitat, output, panel_size=args.panel_size)
        items.append(
            {
                "id": f"viewsuite_habitat::{scene_id}",
                "corpus": "viewsuite_habitat_compare",
                "scene_id": scene_id,
                "image_path": str(output.resolve()),
                "lineage": {
                    "round": 2,
                    "kind": "comparison",
                    "method": "side_by_side",
                    "left_label": "current ViewSuite",
                    "left_path": str(original.resolve()),
                    "left_sha256": _sha256(original),
                    "right_label": "Habitat-Sim rerender",
                    "right_path": str(habitat.resolve()),
                    "right_sha256": _sha256(habitat),
                },
            }
        )

    payload = {
        "version": 1,
        "round": 2,
        "purpose": "viewsuite_habitat_top_down_comparison",
        "data_roots": {
            "viewsuite_original": str(args.original_root.resolve()),
            "viewsuite_habitat": str(args.habitat_root.resolve()),
            "comparison": str(args.output_dir.resolve()),
        },
        "items": items,
    }
    _atomic_json(args.manifest, payload)
    print(f"Wrote {len(items)} comparisons: {args.manifest}", flush=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    repo = _repo_root()
    review_root = repo / "data/topdown_review"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path(
            os.environ.get("VIEWSUITE_ORIGINAL_ROOT", repo / "data/viewsuite_15k")
        ),
    )
    parser.add_argument(
        "--habitat-root",
        type=Path,
        default=Path(
            os.environ.get("VIEWSUITE_HABITAT_ROOT", repo / "data/viewsuite15k-habitat")
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=review_root / "viewsuite_habitat_comparison",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=review_root / "viewsuite_habitat_comparison_manifest.json",
    )
    parser.add_argument(
        "--round-1-labels",
        type=Path,
        default=review_root / "round_1_labels.json",
    )
    parser.add_argument("--only-round-1-rejects", action="store_true")
    parser.add_argument("--panel-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in (
        "original_root",
        "habitat_root",
        "output_dir",
        "manifest",
        "round_1_labels",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
