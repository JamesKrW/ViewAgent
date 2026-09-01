"""Regenerate rejected top-down references for the three proxy corpora.

Round one is a human filter. This module reads its saved labels and writes one
replacement image for every reject, together with the candidate manifest consumed by
``review_top_down --round 2``. Originals are never overwritten.

The three corpora need different repairs:

* AI2-THOR uses the canonical orthographic map camera with the ceiling hidden.
* ViewSuite projects the coloured ScanNet vertices after cutting geometry above the
  sampled eye level, which removes ceilings without changing the camera frame.
* Habitat-GS projects Gaussian centres using their view-independent SH colour. This
  avoids asking a 3DGS scene to synthesize a far-above-training-distribution view and
  rejects large/low-opacity floaters before rasterization.

Run this in a Python environment that contains Pillow, NumPy, SciPy, and
AI2-THOR::

    python -m \
      view_suite.envs.habitat_gs_proxy_task.data_gen.regenerate_top_down --round 2
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

SH_C0 = 0.28209479177387814
PLY_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class CameraMetadata:
    c2w: np.ndarray
    intrinsics: np.ndarray
    eye_heights: np.ndarray
    view_positions: np.ndarray
    source: str


@dataclass(frozen=True)
class ProjectionResult:
    intrinsics: np.ndarray
    camera_pose: np.ndarray
    height_cutoff: float
    retained_vertices: int
    projected_vertices: int
    crop_xywh: tuple[int, int, int, int]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_rejects(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    if int(summary.get("unlabeled", 0)):
        raise ValueError(f"Previous review is incomplete: {summary}")
    return [
        item for item in payload.get("items", []) if item.get("verdict") == "reject"
    ]


def _read_ply_vertices(path: Path) -> np.memmap:
    """Memory-map the fixed-width vertex element without loading later face data."""

    properties: list[tuple[str, str]] = []
    vertex_count: int | None = None
    in_vertices = False
    header_size = 0
    format_name: str | None = None
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY has no end_header: {path}")
            header_size += len(raw)
            line = raw.decode("ascii").strip()
            parts = line.split()
            if parts[:1] == ["format"]:
                format_name = parts[1]
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertices = True
            elif parts[:1] == ["element"]:
                in_vertices = False
            elif in_vertices and parts[:1] == ["property"]:
                if parts[1:2] == ["list"]:
                    raise ValueError(
                        f"Variable-width vertex property in {path}: {line}"
                    )
                if len(parts) != 3 or parts[1] not in PLY_SCALAR_DTYPES:
                    raise ValueError(f"Unsupported PLY property in {path}: {line}")
                properties.append((parts[2], PLY_SCALAR_DTYPES[parts[1]]))
            if line == "end_header":
                break

    if format_name != "binary_little_endian":
        raise ValueError(f"Only binary little-endian PLY is supported: {path}")
    if vertex_count is None or not {"x", "y", "z"}.issubset(
        name for name, _ in properties
    ):
        raise ValueError(f"PLY is missing a fixed-width xyz vertex element: {path}")
    return np.memmap(
        path,
        mode="r",
        dtype=np.dtype(properties),
        offset=header_size,
        shape=(vertex_count,),
    )


def _camera_from_viewsuite_meta(meta_root: Path, scene_id: str) -> CameraMetadata:
    metadata_paths = sorted((meta_root / scene_id).glob("sample_*/meta.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No ViewSuite metadata for {scene_id} under {meta_root}"
        )
    first = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    top_down = first["top_down"]
    eye_heights = []
    view_positions = []
    for path in metadata_paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        views = [sample["initial"], *sample.get("options", [])]
        for view in views:
            position = np.asarray(view["pose_c2w"], dtype=np.float64)[:3, 3]
            view_positions.append(position)
            eye_heights.append(float(position[2]))
    return CameraMetadata(
        c2w=np.asarray(top_down["pose_c2w"], dtype=np.float64),
        intrinsics=np.asarray(top_down["intrinsics"], dtype=np.float64),
        eye_heights=np.asarray(eye_heights, dtype=np.float64),
        view_positions=np.asarray(view_positions, dtype=np.float64),
        source=str(metadata_paths[0]),
    )


def _scan_habitat_metadata(data_root: Path) -> dict[str, CameraMetadata]:
    records: dict[str, dict[str, Any]] = {}
    # The unsuffixed file contains the union and avoids reading duplicated split rows.
    paths = [data_root / "interactive_view_planning.jsonl"]
    if not paths[0].is_file():
        paths = sorted(data_root.glob("interactive_view_planning*.jsonl"))
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                scene_id = str(row["scene_id"])
                detail = row.get("image_detail") or {}
                top_down = detail.get("top_down_view") or {}
                if "c2w_extrinsics" not in top_down:
                    continue
                record = records.setdefault(
                    scene_id,
                    {
                        "c2w": top_down["c2w_extrinsics"],
                        "intrinsics": top_down["c2w_intrinsics"],
                        "heights": [],
                        "positions": [],
                        "source": str(path),
                    },
                )
                for name in ("init_view", "target_view"):
                    view = detail.get(name) or {}
                    pose = view.get("c2w_extrinsics")
                    if pose is not None:
                        position = np.asarray(pose, dtype=np.float64)[:3, 3]
                        record["heights"].append(float(position[1]))
                        record["positions"].append(position)
    return {
        scene_id: CameraMetadata(
            c2w=np.asarray(record["c2w"], dtype=np.float64),
            intrinsics=np.asarray(record["intrinsics"], dtype=np.float64),
            eye_heights=np.asarray(record["heights"], dtype=np.float64),
            view_positions=np.asarray(record["positions"], dtype=np.float64),
            source=str(record["source"]),
        )
        for scene_id, record in records.items()
    }


def _find_scene_file(root: Path, scene_id: str, suffix: str) -> Path:
    matches = [
        root / split / scene_id / f"{scene_id}{suffix}" for split in ("train", "val")
    ]
    for path in matches:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {scene_id}{suffix} under {root}/{{train,val}}"
    )


def _fallback_habitat_camera(vertices: np.memmap, source: str) -> CameraMetadata:
    """Construct a robust camera only for orphan scenes absent from task JSONLs."""

    x = np.asarray(vertices["x"])
    y = np.asarray(vertices["y"])
    z = np.asarray(vertices["z"])
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x_lo, x_hi = np.percentile(x[finite], (1.0, 99.0))
    z_lo, z_hi = np.percentile(z[finite], (1.0, 99.0))
    y_lo, y_hi = np.percentile(y[finite], (5.0, 95.0))
    center = np.asarray([(x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2])
    extent = max(float(x_hi - x_lo), float(z_hi - z_lo))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    c2w[:3, 3] = [center[0], y_hi + 0.75 * extent, center[2]]
    intrinsics = np.asarray(
        [[256.0, 0.0, 256.0], [0.0, 256.0, 256.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # The orphan has no task viewpoints. The median vertical surface band gives a
    # stable cut while avoiding the extreme floaters common in these PLYs.
    eye_height = float(np.percentile(y[finite], 45.0))
    return CameraMetadata(
        c2w,
        intrinsics,
        np.asarray([eye_height]),
        np.empty((0, 3), dtype=np.float64),
        source,
    )


def _vertex_colors(
    vertices: np.memmap, mask: np.ndarray, *, gaussian: bool
) -> np.ndarray:
    names = set(vertices.dtype.names or ())
    if {"red", "green", "blue"}.issubset(names):
        return np.stack(
            [
                np.asarray(vertices[channel])[mask]
                for channel in ("red", "green", "blue")
            ],
            axis=1,
        ).astype(np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        colors = np.stack(
            [
                0.5 + SH_C0 * np.asarray(vertices[f"f_dc_{index}"])[mask]
                for index in range(3)
            ],
            axis=1,
        )
        return np.clip(colors * 255.0, 0.0, 255.0).astype(np.float32)
    shade = 150.0 if gaussian else 175.0
    return np.full((int(mask.sum()), 3), shade, dtype=np.float32)


def _largest_components(mask: np.ndarray) -> np.ndarray:
    expanded = ndimage.binary_dilation(mask, iterations=3)
    labels, count = ndimage.label(expanded)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max(initial=0))
    keep_labels = np.flatnonzero(sizes[1:] >= max(80, int(largest * 0.006))) + 1
    keep = np.isin(labels, keep_labels)
    return mask & ndimage.binary_dilation(keep, iterations=1)


def _crop_square(
    mask: np.ndarray, *, margin_fraction: float = 0.08
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if not len(xs):
        return (0, 0, width, height)
    # Percentiles discard a few isolated floaters without cutting real room geometry.
    x0, x1 = np.percentile(xs, (0.15, 99.85))
    y0, y1 = np.percentile(ys, (0.15, 99.85))
    side = max(float(x1 - x0 + 1), float(y1 - y0 + 1))
    side *= 1.0 + 2.0 * margin_fraction
    side = min(float(min(width, height)), max(32.0, side))
    center_x = (float(x0) + float(x1)) / 2.0
    center_y = (float(y0) + float(y1)) / 2.0
    left = round(np.clip(center_x - side / 2, 0, width - side))
    top = round(np.clip(center_y - side / 2, 0, height - side))
    size = round(side)
    return left, top, min(size, width - left), min(size, height - top)


def _project_vertices(
    vertices: np.memmap,
    camera: CameraMetadata,
    output: Path,
    *,
    vertical_axis: int,
    gaussian: bool,
    image_size: int = 512,
    oversample: int = 2,
    ceiling_margin: float = 0.25,
) -> ProjectionResult:
    coords = [np.asarray(vertices[name]) for name in ("x", "y", "z")]
    finite = np.isfinite(coords[0]) & np.isfinite(coords[1]) & np.isfinite(coords[2])
    eye_height = float(np.median(camera.eye_heights))
    height_cutoff = eye_height + float(ceiling_margin)
    lower_cutoff = eye_height - 2.0
    heights = coords[vertical_axis]
    mask = finite & (heights >= lower_cutoff) & (heights <= height_cutoff)

    alpha: np.ndarray | None = None
    if gaussian:
        opacity = np.asarray(vertices["opacity"])
        max_scale = np.maximum.reduce(
            [np.asarray(vertices[f"scale_{index}"]) for index in range(3)]
        )
        # +inf opacity is used for fully opaque Gaussians in part of this corpus.
        mask &= ~np.isnan(opacity) & np.isfinite(max_scale)
        mask &= (opacity > -4.0) & (max_scale < -1.2)

        # Aerial 3DGS renders are especially sensitive to floaters far from the
        # camera manifold. Keep a generous tube around the task's real viewpoints;
        # it retains room geometry while removing isolated streaks that otherwise
        # dominate the crop. Large scenes scale the radius with their sampled span.
        positions = np.asarray(camera.view_positions, dtype=np.float64)
        if len(positions) >= 2:
            from scipy.spatial import cKDTree

            horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
            sampled = positions[:, horizontal_axes]
            sampled_span = float(np.ptp(sampled, axis=0).max())
            # The ``sceneNN`` family can span hundreds of units and its sparse
            # samples do not trace every corridor. Restrict only compact rooms,
            # where the same test reliably separates nearby structure from floaters.
            if sampled_span <= 40.0:
                radius = max(4.0, 0.15 * sampled_span)
                candidate_indices = np.flatnonzero(mask)
                candidate_points = np.stack(
                    [coords[axis][mask] for axis in horizontal_axes], axis=1
                )
                distances = cKDTree(sampled).query(candidate_points, workers=-1)[0]
                keep_indices = candidate_indices[distances <= radius]
                mask[:] = False
                mask[keep_indices] = True
        alpha = 1.0 / (1.0 + np.exp(-np.clip(opacity[mask], -20.0, 20.0)))

    retained = int(mask.sum())
    if retained == 0:
        raise ValueError(f"Height/quality filtering removed every vertex for {output}")
    points = np.stack([coordinate[mask] for coordinate in coords], axis=1).astype(
        np.float32
    )
    colors = _vertex_colors(vertices, mask, gaussian=gaussian)
    c2w = np.asarray(camera.c2w, dtype=np.float32)
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float32)
    camera_points = (points - c2w[:3, 3]) @ c2w[:3, :3]
    visible = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0.05)
    points = points[visible]
    camera_points = camera_points[visible]
    colors = colors[visible]
    point_heights = points[:, vertical_axis]
    if alpha is not None:
        alpha = alpha[visible]
    else:
        alpha = np.ones(len(points), dtype=np.float32)

    canvas_size = int(image_size * oversample)
    scale = float(oversample)
    u = np.rint(
        (
            intrinsics[0, 0] * camera_points[:, 0] / camera_points[:, 2]
            + intrinsics[0, 2]
        )
        * scale
    ).astype(np.int32)
    v = np.rint(
        (
            intrinsics[1, 1] * camera_points[:, 1] / camera_points[:, 2]
            + intrinsics[1, 2]
        )
        * scale
    ).astype(np.int32)
    in_frame = (u >= 0) & (u < canvas_size) & (v >= 0) & (v < canvas_size)
    u, v = u[in_frame], v[in_frame]
    colors = colors[in_frame]
    point_heights = point_heights[in_frame]
    alpha = alpha[in_frame]
    if not len(u):
        raise ValueError(f"No projected vertices land inside the image for {output}")

    keys = v * canvas_size + u
    top_height = np.full(canvas_size * canvas_size, -np.inf, dtype=np.float32)
    np.maximum.at(top_height, keys, point_heights.astype(np.float32))
    surface = point_heights >= top_height[keys] - (0.08 if gaussian else 0.035)
    keys, colors, alpha = keys[surface], colors[surface], alpha[surface]

    color_sum = np.zeros((canvas_size * canvas_size, 3), dtype=np.float32)
    weight_sum = np.zeros(canvas_size * canvas_size, dtype=np.float32)
    for channel in range(3):
        np.add.at(color_sum[:, channel], keys, colors[:, channel] * alpha)
    np.add.at(weight_sum, keys, alpha)
    occupied = weight_sum.reshape(canvas_size, canvas_size) > 0
    occupied = _largest_components(occupied)

    rgb = np.full((canvas_size, canvas_size, 3), 246, dtype=np.uint8)
    valid_pixels = weight_sum > 0
    rgb.reshape(-1, 3)[valid_pixels] = np.clip(
        color_sum[valid_pixels] / weight_sum[valid_pixels, None], 0, 255
    ).astype(np.uint8)
    rgb[~occupied] = 246
    distance, nearest = ndimage.distance_transform_edt(~occupied, return_indices=True)
    fill = (~occupied) & (distance <= (5.0 if gaussian else 3.5))
    rgb[fill] = rgb[nearest[0][fill], nearest[1][fill]]
    display_mask = occupied | fill

    left, top, crop_width, crop_height = _crop_square(display_mask)
    side = min(crop_width, crop_height)
    image = Image.fromarray(rgb).crop((left, top, left + side, top + side))
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.08 if gaussian else 1.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)

    output_scale = image_size / float(side)
    adjusted = intrinsics.astype(np.float64).copy()
    adjusted[0, 0] *= scale * output_scale
    adjusted[1, 1] *= scale * output_scale
    adjusted[0, 2] = (adjusted[0, 2] * scale - left) * output_scale
    adjusted[1, 2] = (adjusted[1, 2] * scale - top) * output_scale
    return ProjectionResult(
        intrinsics=adjusted,
        camera_pose=np.asarray(camera.c2w, dtype=np.float64),
        height_cutoff=height_cutoff,
        retained_vertices=retained,
        projected_vertices=len(keys),
        crop_xywh=(left, top, side, side),
    )


def _render_ai2thor(scene_id: str, output: Path) -> tuple[np.ndarray, np.ndarray]:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    from view_suite.ai2thor.pose_utils import intrinsics_from_fov, unity_pose_to_c2w
    from view_suite.envs.ai2thor_proxy_task.data_gen.generate_data import (
        _render_topdown_mapview,
    )

    controller = Controller(
        platform=CloudRendering,
        agentMode="default",
        scene=scene_id,
        width=512,
        height=512,
        fieldOfView=90.0,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )
    try:
        rgb, pose = _render_topdown_mapview(controller)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(output)
        return unity_pose_to_c2w(pose), intrinsics_from_fov(512, 512, 90.0)
    finally:
        controller.stop()


def _lineage(
    item: dict[str, Any],
    method: str,
    failure_reason: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "round": 2,
        "kind": "regenerated",
        "parent_sha256": item.get("sha256"),
        "parent_image_rel": item.get("image_rel"),
        "failure_reason": failure_reason,
        "method": method,
        **details,
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    rejects = _load_rejects(args.parent_labels)
    selected_corpora = set(args.corpora)
    selected_scenes = set(args.scenes or ())
    rejects = [
        item
        for item in rejects
        if item["corpus"] in selected_corpora
        and (not selected_scenes or item["scene_id"] in selected_scenes)
    ]
    if not rejects:
        raise ValueError("No rejected items match the requested corpus/scene filters")

    habitat_metadata = (
        _scan_habitat_metadata(args.habitat_gs_data_root)
        if "habitat_gs" in selected_corpora
        else {}
    )
    output_items: list[dict[str, Any]] = []
    for index, item in enumerate(rejects, 1):
        corpus, scene_id = str(item["corpus"]), str(item["scene_id"])
        output = args.output_dir / corpus / f"{scene_id}.png"
        print(f"[{index}/{len(rejects)}] {corpus}/{scene_id}", flush=True)

        if corpus == "ai2thor":
            pose, intrinsics = _render_ai2thor(scene_id, output)
            lineage = _lineage(
                item,
                "ai2thor_map_camera_ceiling_hidden",
                "ceiling_occlusion",
                camera_pose=pose.tolist(),
                camera_intrinsics=np.asarray(intrinsics).tolist(),
            )
        elif corpus == "viewsuite":
            camera = _camera_from_viewsuite_meta(args.viewsuite_meta_root, scene_id)
            ply = args.scannet_root / scene_id / f"{scene_id}_vh_clean.ply"
            result = _project_vertices(
                _read_ply_vertices(ply),
                camera,
                output,
                vertical_axis=2,
                gaussian=False,
            )
            lineage = _lineage(
                item,
                "ceiling_clipped_colored_vertex_projection",
                "ceiling_or_high_surface_occlusion",
                source_ply=str(ply),
                metadata_source=camera.source,
                camera_pose=result.camera_pose.tolist(),
                camera_intrinsics=result.intrinsics.tolist(),
                height_cutoff=result.height_cutoff,
                retained_vertices=result.retained_vertices,
                projected_vertices=result.projected_vertices,
                crop_xywh=list(result.crop_xywh),
            )
        elif corpus == "habitat_gs":
            ply = _find_scene_file(args.habitat_gs_source_root, scene_id, ".gs.ply")
            vertices = _read_ply_vertices(ply)
            camera = habitat_metadata.get(scene_id)
            if camera is None:
                camera = _fallback_habitat_camera(vertices, str(ply))
            result = _project_vertices(
                vertices,
                camera,
                output,
                vertical_axis=1,
                gaussian=True,
            )
            lineage = _lineage(
                item,
                "ceiling_clipped_gaussian_dc_projection",
                "off_manifold_high_altitude_gaussian_render",
                source_ply=str(ply),
                metadata_source=camera.source,
                camera_pose=result.camera_pose.tolist(),
                camera_intrinsics=result.intrinsics.tolist(),
                height_cutoff=result.height_cutoff,
                retained_vertices=result.retained_vertices,
                projected_vertices=result.projected_vertices,
                crop_xywh=list(result.crop_xywh),
            )
        else:
            raise ValueError(f"Unsupported corpus: {corpus}")

        output_items.append(
            {
                "id": item["id"],
                "corpus": corpus,
                "scene_id": scene_id,
                "image_path": str(output.resolve()),
                "sources": item.get("sources", []),
                "lineage": lineage,
            }
        )

    manifest = {
        "version": 1,
        "round": int(args.round),
        "parent_labels": str(args.parent_labels.resolve()),
        "data_roots": {
            "round_candidates": str(args.output_dir.resolve()),
        },
        "items": output_items,
    }
    # A filtered diagnostic run should not replace the complete round manifest.
    if not selected_scenes and selected_corpora == {
        "ai2thor",
        "viewsuite",
        "habitat_gs",
    }:
        _atomic_json(args.manifest, manifest)
        print(f"Wrote manifest: {args.manifest}", flush=True)
    else:
        diagnostic = args.manifest.with_name(args.manifest.stem + "_partial.json")
        _atomic_json(diagnostic, manifest)
        print(f"Wrote partial manifest: {diagnostic}", flush=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    repo = _repo_root()
    review_root = repo / "data/topdown_review"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(2,), default=2)
    parser.add_argument(
        "--parent-labels",
        type=Path,
        default=review_root / "round_1_labels.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=review_root / "round_2_candidates",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=review_root / "round_2_manifest.json",
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=("ai2thor", "viewsuite", "habitat_gs"),
        default=("ai2thor", "viewsuite", "habitat_gs"),
    )
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--viewsuite-meta-root",
        type=Path,
        default=Path(
            os.environ.get("VIEWSUITE_HABITAT_ROOT", repo / "data/viewsuite15k-habitat")
        ),
    )
    parser.add_argument(
        "--scannet-root",
        type=Path,
        default=Path(os.environ.get("SCANNET_ROOT", repo / "data/scannet/scans")),
    )
    parser.add_argument(
        "--habitat-gs-source-root",
        type=Path,
        default=Path(os.environ.get("HABITAT_GS_SOURCE_ROOT", repo / "data/gs_scenes")),
    )
    parser.add_argument(
        "--habitat-gs-data-root",
        type=Path,
        default=repo / "data/habitat_gs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.parent_labels = args.parent_labels.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
