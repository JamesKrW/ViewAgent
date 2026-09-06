"""Re-render only Habitat-GS Round-2 top-down candidates with true splatting.

The first replacement pass projected Gaussian centres directly. That made layout
legible but looked like a point cloud. This pass instead builds a temporary, filtered
3DGS asset per scene and lets Habitat-GS perform its native covariance/opacity
rasterization. It fixes the two important out-of-distribution failure modes first:

* remove ceiling/high-layer Gaussians and distant floaters;
* bake each Gaussian's colour from its nearest real task camera, then disable the
  view-dependent SH terms before rendering from overhead.

AI2-THOR and ViewSuite entries in the Round-2 manifest are preserved byte-for-byte.
Run with the separate ``habitat-gs`` Python environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from view_suite.envs.habitat_gs_proxy_task.data_gen.regenerate_top_down import (
    CameraMetadata,
    _fallback_habitat_camera,
    _find_scene_file,
    _read_ply_vertices,
    _scan_habitat_metadata,
)
from view_suite.habitat_gs.habitat_gs_render import HabitatGSRenderer
from view_suite.habitat_gs.pose_utils import fov_from_K

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = np.asarray(
    [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ],
    dtype=np.float32,
)
SH_C3 = np.asarray(
    [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ],
    dtype=np.float32,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sh_basis(directions: np.ndarray, degree: int) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float32)
    basis = np.zeros((len(directions), (degree + 1) ** 2), dtype=np.float32)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    basis[:, 0] = SH_C0
    if degree >= 1:
        basis[:, 1] = -SH_C1 * y
        basis[:, 2] = SH_C1 * z
        basis[:, 3] = -SH_C1 * x
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        basis[:, 4] = SH_C2[0] * xy
        basis[:, 5] = SH_C2[1] * yz
        basis[:, 6] = SH_C2[2] * (2.0 * zz - xx - yy)
        basis[:, 7] = SH_C2[3] * xz
        basis[:, 8] = SH_C2[4] * (xx - yy)
    if degree >= 3:
        xx, yy, zz = x * x, y * y, z * z
        basis[:, 9] = SH_C3[0] * y * (3.0 * xx - yy)
        basis[:, 10] = SH_C3[1] * x * y * z
        basis[:, 11] = SH_C3[2] * y * (4.0 * zz - xx - yy)
        basis[:, 12] = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
        basis[:, 13] = SH_C3[4] * x * (4.0 * zz - xx - yy)
        basis[:, 14] = SH_C3[5] * z * (xx - yy)
        basis[:, 15] = SH_C3[6] * x * (xx - 3.0 * yy)
    return basis


def _select_gaussians(
    vertices: np.memmap,
    camera: CameraMetadata,
    *,
    ceiling_margin: float,
    lower_percentile: float,
    opacity_min: float,
    max_log_scale: float,
) -> tuple[np.ndarray, dict[str, float | None]]:
    xyz = np.stack([np.asarray(vertices[name]) for name in ("x", "y", "z")], axis=1)
    opacity = np.asarray(vertices["opacity"])
    max_scale = np.maximum.reduce(
        [np.asarray(vertices[f"scale_{index}"]) for index in range(3)]
    )
    eye_height = float(np.median(camera.eye_heights))
    upper = eye_height + ceiling_margin
    mask = np.isfinite(xyz).all(axis=1) & ~np.isnan(opacity) & np.isfinite(max_scale)
    mask &= (xyz[:, 1] <= upper) & (opacity > opacity_min) & (max_scale < max_log_scale)

    positions = np.asarray(camera.view_positions, dtype=np.float64)
    tube_radius: float | None = None
    if len(positions) >= 2:
        sampled = positions[:, (0, 2)]
        sampled_span = float(np.ptp(sampled, axis=0).max())
        if sampled_span <= 40.0:
            tube_radius = max(4.0, 0.15 * sampled_span)
            candidate_indices = np.flatnonzero(mask)
            candidate_xz = xyz[mask][:, (0, 2)]
            distances = cKDTree(sampled).query(candidate_xz, workers=-1)[0]
            keep = candidate_indices[distances <= tube_radius]
            mask[:] = False
            mask[keep] = True

    candidate_indices = np.flatnonzero(mask)
    if not len(candidate_indices):
        raise ValueError("Gaussian quality filtering removed every point")
    lower = float(np.percentile(xyz[candidate_indices, 1], lower_percentile))
    selected = candidate_indices[xyz[candidate_indices, 1] >= lower]
    return selected, {
        "eye_height": eye_height,
        "lower_cutoff": lower,
        "upper_cutoff": upper,
        "tube_radius": tube_radius,
    }


def _bake_nearest_view_color(
    block: np.ndarray,
    positions: np.ndarray,
    view_positions: np.ndarray,
    *,
    chunk_size: int = 100_000,
) -> int:
    rest_names = sorted(
        [name for name in (block.dtype.names or ()) if name.startswith("f_rest_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if not rest_names:
        return 0
    coefficients_per_channel = len(rest_names) // 3
    degree = round(math.sqrt(coefficients_per_channel + 1) - 1)
    if degree < 1 or degree > 3 or len(view_positions) == 0:
        for name in rest_names:
            block[name] = 0.0
        return 0

    tree = cKDTree(np.asarray(view_positions, dtype=np.float64))
    for start in range(0, len(block), chunk_size):
        stop = min(start + chunk_size, len(block))
        points = np.asarray(positions[start:stop], dtype=np.float64)
        nearest = tree.query(points, workers=-1)[1]
        directions = points - view_positions[nearest]
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-8
        )
        basis = _sh_basis(directions, degree)
        dc = np.stack(
            [np.asarray(block[f"f_dc_{channel}"][start:stop]) for channel in range(3)],
            axis=1,
        )
        rest = np.stack(
            [np.asarray(block[name][start:stop]) for name in rest_names], axis=1
        ).reshape(stop - start, 3, coefficients_per_channel)
        rest = rest.transpose(0, 2, 1)
        coefficients = np.concatenate([dc[:, None, :], rest], axis=1)
        rgb = np.einsum("nk,nkc->nc", basis, coefficients) + 0.5
        rgb = np.clip(rgb, 0.0, 1.0)
        for channel in range(3):
            block[f"f_dc_{channel}"][start:stop] = (
                (rgb[:, channel] - 0.5) / SH_C0
            ).astype(np.float32)
    for name in rest_names:
        block[name] = 0.0
    return degree


def _write_filtered_ply(
    source: Path,
    vertices: np.memmap,
    selected: np.ndarray,
    camera: CameraMetadata,
    destination: Path,
) -> int:
    with source.open("rb") as handle:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY has no end_header: {source}")
            header += line
    header_text = header.decode("ascii")
    if re.search(r"^element (?!vertex\b)", header_text, flags=re.MULTILINE):
        raise ValueError(
            f"Gaussian PLY unexpectedly contains non-vertex elements: {source}"
        )
    header_text = re.sub(
        r"element vertex \d+",
        f"element vertex {len(selected)}",
        header_text,
        count=1,
    )

    block = vertices[selected].copy()
    positions = np.stack([block[name] for name in ("x", "y", "z")], axis=1)
    degree = _bake_nearest_view_color(block, positions, camera.view_positions)
    block["opacity"] = np.clip(block["opacity"], -10.0, 10.0)
    with destination.open("wb") as handle:
        handle.write(header_text.encode("ascii"))
        block.tofile(handle)
    return degree


def _crop_and_save(
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    output: Path,
    *,
    output_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.asarray(rgb, dtype=np.uint8)
    distance_from_white = np.linalg.norm(image.astype(np.int16) - 255, axis=2)
    foreground = distance_from_white > 18
    expanded = ndimage.binary_dilation(foreground, iterations=2)
    labels, count = ndimage.label(expanded)
    if count:
        sizes = np.bincount(labels.ravel())
        largest = int(sizes[1:].max(initial=0))
        keep = np.flatnonzero(sizes[1:] >= max(50, int(largest * 0.004))) + 1
        foreground &= np.isin(labels, keep)
    ys, xs = np.nonzero(foreground)
    height, width = image.shape[:2]
    if len(xs):
        x0, x1 = np.percentile(xs, (0.1, 99.9))
        y0, y1 = np.percentile(ys, (0.1, 99.9))
        side = min(max(x1 - x0 + 1, y1 - y0 + 1) * 1.12, min(width, height))
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        left = round(np.clip(center_x - side / 2, 0, width - side))
        top = round(np.clip(center_y - side / 2, 0, height - side))
        side = round(side)
    else:
        left, top, side = 0, 0, min(width, height)
    cropped = Image.fromarray(image).crop((left, top, left + side, top + side))
    cropped = cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(output)

    scale = output_size / float(side)
    adjusted = np.asarray(intrinsics, dtype=np.float64).copy()
    adjusted[0, 0] *= scale
    adjusted[1, 1] *= scale
    adjusted[0, 2] = (adjusted[0, 2] - left) * scale
    adjusted[1, 2] = (adjusted[1, 2] - top) * scale
    return adjusted, (left, top, side, side)


def _render_scene(
    scene_id: str,
    entry: dict[str, Any],
    camera: CameraMetadata,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source = _find_scene_file(args.source_root, scene_id, ".gs.ply")
    navmesh = _find_scene_file(args.source_root, scene_id, ".navmesh")
    vertices = _read_ply_vertices(source)
    selected, cutoffs = _select_gaussians(
        vertices,
        camera,
        ceiling_margin=args.ceiling_margin,
        lower_percentile=args.lower_percentile,
        opacity_min=args.opacity_min,
        max_log_scale=args.max_log_scale,
    )
    output = args.output_dir / f"{scene_id}.png"
    render_size = int(args.render_size)
    source_k = np.asarray(camera.intrinsics, dtype=np.float64)
    render_k = source_k.copy()
    render_k[:2] *= render_size / 512.0

    with tempfile.TemporaryDirectory(prefix=f"hgs_topdown_{scene_id}_") as temporary:
        filtered = Path(temporary) / f"{scene_id}.gs.ply"
        degree = _write_filtered_ply(source, vertices, selected, camera, filtered)
        renderer = HabitatGSRenderer(
            str(filtered),
            gpu_device_id=args.gpu,
            width=render_size,
            height=render_size,
            hfov_deg=fov_from_K(render_k, render_size),
            navmesh_path=str(navmesh),
            background=(1.0, 1.0, 1.0, 1.0),
        )
        try:
            rgb = renderer.render_image_from_cam_param(
                render_k, camera.c2w, render_size, render_size
            )
        finally:
            renderer.close()

    adjusted_k, crop = _crop_and_save(
        rgb, render_k, output, output_size=args.output_size
    )
    lineage = dict(entry.get("lineage") or {})
    lineage.update(
        {
            "method": "height_clipped_native_gaussian_splat",
            "failure_reason": "off_manifold_high_altitude_gaussian_render",
            "previous_candidate_path": entry.get("image_path"),
            "source_ply": str(source),
            "metadata_source": camera.source,
            "camera_pose": camera.c2w.tolist(),
            "camera_intrinsics": adjusted_k.tolist(),
            "selected_gaussians": len(selected),
            "source_gaussians": len(vertices),
            "baked_sh_degree": int(degree),
            "crop_xywh": list(crop),
            **cutoffs,
        }
    )
    updated = dict(entry)
    updated["image_path"] = str(output.resolve())
    updated["lineage"] = lineage
    return updated


def generate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if int(manifest.get("round", 0)) != 2:
        raise ValueError(f"Expected a Round-2 manifest: {args.manifest}")
    metadata = _scan_habitat_metadata(args.data_root)
    requested = set(args.scenes or ())
    habitat_entries = [
        entry
        for entry in manifest.get("items", [])
        if entry.get("corpus") == "habitat_gs"
        and (not requested or entry.get("scene_id") in requested)
    ]
    if not habitat_entries:
        raise ValueError("No Habitat-GS entries match the requested scenes")

    replacements: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(habitat_entries, 1):
        scene_id = str(entry["scene_id"])
        print(f"[{index}/{len(habitat_entries)}] {scene_id}", flush=True)
        source = _find_scene_file(args.source_root, scene_id, ".gs.ply")
        camera = metadata.get(scene_id)
        if camera is None:
            camera = _fallback_habitat_camera(_read_ply_vertices(source), str(source))
        replacements[str(entry["id"])] = _render_scene(scene_id, entry, camera, args)

    updated_manifest = dict(manifest)
    updated_manifest["items"] = [
        replacements.get(str(entry.get("id")), entry)
        for entry in manifest.get("items", [])
    ]
    updated_manifest["habitat_gs_rerender"] = {
        "method": "height_clipped_native_gaussian_splat",
        "count": len(replacements),
        "output_dir": str(args.output_dir.resolve()),
    }

    if requested and not args.commit_partial:
        output_manifest = args.manifest.with_name(
            args.manifest.stem + "_habitat_splat_preview.json"
        )
    else:
        backup = args.manifest.with_name(args.manifest.stem + "_point_projection.json")
        if not backup.exists():
            shutil.copy2(args.manifest, backup)
        output_manifest = args.manifest
    _atomic_json(output_manifest, updated_manifest)
    print(f"Wrote manifest: {output_manifest}", flush=True)
    return updated_manifest


def _parser() -> argparse.ArgumentParser:
    repo = _repo_root()
    review_root = repo / "data/topdown_review"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=review_root / "round_2_manifest.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=review_root / "round_2_candidates_habitat_splat" / "habitat_gs",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("HABITAT_GS_SOURCE_ROOT", repo / "data/gs_scenes")),
    )
    parser.add_argument("--data-root", type=Path, default=repo / "data/viewagent15k_habitat_gs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--ceiling-margin", type=float, default=0.25)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--opacity-min", type=float, default=-4.0)
    parser.add_argument("--max-log-scale", type=float, default=0.5)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--commit-partial",
        action="store_true",
        help="Update the main manifest even when --scenes selects a subset.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("manifest", "output_dir", "source_root", "data_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
