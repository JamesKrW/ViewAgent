#!/usr/bin/env python3
"""Re-render ViewSuite15K with Habitat-Sim and add action intermediates.

The generator intentionally runs in three resumable stages:

``base``
    Re-render every ViewSuite sample image (initial view, four options, and one
    top-down image per scene) through the ScanNet Habitat HTTP service.  A few
    source scene directories contain only a top-down PNG and no camera metadata;
    for those, a documented fallback camera is derived from the mesh AABB.  The
    output keeps the original ViewSuite directory layout and JSONL files.

``p2v``
    Replay the single Path-to-View action sequence from the initial camera.  The
    image after every action, including the final target view, is recorded in
    ``intermediate_image_path`` and ``intermediate_image_detail``.

``v2p``
    Replay all four View-to-Path option sequences from the initial camera and
    record the image after every action for every option.

All stages share ``<base_out>/pose_cache``.  A cache key contains the scene,
intrinsics, camera-to-world pose, and output size, so identical poses are rendered
once even when they occur in several splits/tasks.  Base dataset filenames are
hard-linked to the cache.  The intermediate datasets contain relative symlinks to
the base scene directories and pose cache, avoiding image duplication while
keeping all original ``image_path`` values valid.

Examples::

    python -m view_suite.envs.scannet_proxy_task.data_gen.regen_viewsuite_habitat \
      --stage base --client-url http://127.0.0.1:8813

    python -m view_suite.envs.scannet_proxy_task.data_gen.regen_viewsuite_habitat \
      --stage p2v --client-url http://127.0.0.1:8813

    python -m view_suite.envs.scannet_proxy_task.data_gen.regen_viewsuite_habitat \
      --stage v2p --client-url http://127.0.0.1:8813

The process is safe to re-run: valid cached PNGs and existing hard links are reused.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import httpx
import numpy as np

from view_suite.scannet.view_manipulator import ViewManipulator


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TASK_SPLITS = ("train", "dev", "test")
ACTION_NAMES = {
    "w": "move_forward",
    "s": "move_backward",
    "d": "move_right",
    "a": "move_left",
    "y": "move_up",
    "h": "move_down",
    "q": "turn_left",
    "e": "turn_right",
    "r": "look_up",
    "f": "look_down",
    "t": "rotate_ccw",
    "g": "rotate_cw",
}
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
FALLBACK_TOPDOWN_FX = 462.29648919753083
FALLBACK_TOPDOWN_FILL_RATIO = 0.85


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _valid_png(path: Path) -> bool:
    try:
        if path.stat().st_size < 32:
            return False
        with path.open("rb") as f:
            return f.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            if os.path.samefile(src, dst):
                return
            dst.unlink()
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _ensure_relative_symlink(link: Path, target: Path) -> None:
    """Create ``link`` pointing to ``target`` with a relative symlink."""
    relative = os.path.relpath(target, start=link.parent)
    if link.is_symlink() and os.readlink(link) == relative:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(relative, target_is_directory=True)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _matrix_list(value: Any) -> list[list[float]]:
    return np.asarray(value, dtype=np.float64).tolist()


def _read_binary_ply_xyz_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only a binary little-endian PLY's vertex block via numpy memmap."""
    header_lines: list[str] = []
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"unterminated PLY header: {path}")
            header_lines.append(raw.decode("ascii").strip())
            if raw == b"end_header\n" or raw == b"end_header\r\n":
                header_size = f.tell()
                break

    if "format binary_little_endian 1.0" not in header_lines:
        raise ValueError(f"only binary_little_endian PLY is supported: {path}")

    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if parts[:1] == ["element"] and parts[1:2] != ["vertex"]:
            in_vertex = False
        if in_vertex and parts[:1] == ["property"]:
            if parts[1:2] == ["list"]:
                raise ValueError(f"list property in PLY vertex element: {path}")
            if len(parts) != 3 or parts[1] not in PLY_SCALAR_DTYPES:
                raise ValueError(f"unsupported PLY property {line!r}: {path}")
            vertex_properties.append((parts[2], PLY_SCALAR_DTYPES[parts[1]]))

    if vertex_count is None or not {"x", "y", "z"}.issubset(
        name for name, _ in vertex_properties
    ):
        raise ValueError(f"missing vertex count or xyz properties: {path}")

    vertices = np.memmap(
        path,
        mode="r",
        dtype=np.dtype(vertex_properties),
        offset=header_size,
        shape=(vertex_count,),
    )
    minimum = np.asarray(
        [vertices[axis].min() for axis in ("x", "y", "z")], dtype=np.float64
    )
    maximum = np.asarray(
        [vertices[axis].max() for axis in ("x", "y", "z")], dtype=np.float64
    )
    return minimum, maximum


def _derive_top_down_from_mesh(scannet_root: Path, scene_id: str) -> dict[str, Any]:
    """Derive a reproducible fallback top-down camera for metadata-less scenes."""
    ply_path = scannet_root / scene_id / f"{scene_id}_vh_clean.ply"
    minimum, maximum = _read_binary_ply_xyz_bounds(ply_path)
    center = (minimum + maximum) / 2.0
    span = maximum - minimum
    depth = (
        max(float(span[0]), float(span[1]))
        * FALLBACK_TOPDOWN_FX
        / (512.0 * FALLBACK_TOPDOWN_FILL_RATIO)
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [center[0], center[1], center[2] + depth]
    intrinsics = np.asarray(
        [
            [FALLBACK_TOPDOWN_FX, 0.0, 256.0],
            [0.0, FALLBACK_TOPDOWN_FX, 256.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return {
        "scene_id": scene_id,
        "derivation": "mesh_aabb_center_and_fit",
        "mesh_path": str(ply_path),
        "mesh_aabb_min": minimum.tolist(),
        "mesh_aabb_max": maximum.tolist(),
        "fill_ratio": FALLBACK_TOPDOWN_FILL_RATIO,
        "pose_c2w": pose.tolist(),
        "intrinsics": intrinsics.tolist(),
        "image": "top_down_view.png",
    }


def _pose_digest(
    scene_id: str,
    intrinsics: Any,
    extrinsics: Any,
    size: tuple[int, int],
) -> str:
    # Eight decimals are much tighter than the dataset's camera tolerances while
    # collapsing harmless JSON/scipy floating-point noise (e.g. 1e-16 vs 0).
    values = np.concatenate(
        [
            np.asarray(intrinsics, dtype=np.float64).reshape(-1),
            np.asarray(extrinsics, dtype=np.float64).reshape(-1),
            np.asarray(size, dtype=np.float64),
        ]
    )
    payload = scene_id.encode("utf-8") + np.round(values, 8).astype("<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()[:24]


def _projection_digest(spec: "RenderSpec") -> str:
    values = np.concatenate(
        [
            np.asarray(spec.intrinsics, dtype=np.float64).reshape(-1),
            np.asarray(spec.size, dtype=np.float64),
        ]
    )
    return hashlib.sha256(np.round(values, 8).astype("<f8").tobytes()).hexdigest()[:16]


@dataclass
class RenderSpec:
    scene_id: str
    intrinsics: list[list[float]]
    extrinsics: list[list[float]]
    size: tuple[int, int]
    digest: str
    cache_rel: Path
    logical_outputs: set[Path] = field(default_factory=set)

    def task(self) -> dict[str, Any]:
        return {
            "mode": "cam_param",
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "size": list(self.size),
        }


class PoseRegistry:
    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self._specs: dict[tuple[str, str], RenderSpec] = {}

    def add(
        self,
        scene_id: str,
        intrinsics: Any,
        extrinsics: Any,
        *,
        size: tuple[int, int] = (512, 512),
        logical_output: Path | None = None,
    ) -> RenderSpec:
        digest = _pose_digest(scene_id, intrinsics, extrinsics, size)
        key = (scene_id, digest)
        spec = self._specs.get(key)
        if spec is None:
            cache_rel = Path(scene_id) / f"{digest}.png"
            spec = RenderSpec(
                scene_id=scene_id,
                intrinsics=_matrix_list(intrinsics),
                extrinsics=_matrix_list(extrinsics),
                size=(int(size[0]), int(size[1])),
                digest=digest,
                cache_rel=cache_rel,
            )
            self._specs[key] = spec
        if logical_output is not None:
            spec.logical_outputs.add(logical_output)
        return spec

    def by_scene(self) -> dict[str, list[RenderSpec]]:
        grouped: dict[str, list[RenderSpec]] = defaultdict(list)
        for spec in self._specs.values():
            grouped[spec.scene_id].append(spec)
        return dict(grouped)

    def __len__(self) -> int:
        return len(self._specs)

    def values(self) -> Iterable[RenderSpec]:
        return self._specs.values()


def _extract_boundary(content_type: str) -> bytes:
    for part in (content_type or "").split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            value = part.split("=", 1)[1].strip().strip('"')
            if value:
                return value.encode("utf-8")
    raise ValueError(f"missing multipart boundary: {content_type!r}")


def _decode_multipart_png_bytes(
    content_type: str, body: bytes
) -> tuple[dict[str, Any], list[bytes]]:
    """Decode the service response without PIL re-encoding the returned PNGs."""
    marker = b"--" + _extract_boundary(content_type)
    meta: dict[str, Any] = {}
    images: list[bytes] = []
    for chunk in body.split(marker):
        # Multipart framing contributes one leading and one trailing CRLF around
        # a part.  Remove only those exact bytes; bytes.strip() can corrupt a PNG
        # whose final CRC byte happens to be ASCII whitespace.
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        if not chunk or chunk == b"--":
            continue
        if chunk.endswith(b"--"):
            chunk = chunk[:-2]
            if chunk.endswith(b"\r\n"):
                chunk = chunk[:-2]
        header_blob, separator, payload = chunk.partition(b"\r\n\r\n")
        if not separator:
            continue
        part_type = ""
        for line in header_blob.decode("utf-8", errors="ignore").split("\r\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == "content-type":
                part_type = value.strip().lower()
        if "application/json" in part_type:
            parsed = json.loads(payload.decode("utf-8"))
            meta = parsed if isinstance(parsed, dict) else {"_meta": parsed}
        elif part_type.startswith("image/"):
            if not payload.startswith(PNG_SIGNATURE):
                raise ValueError("render service returned a non-PNG image payload")
            images.append(payload)
    return meta, images


class RawRenderClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float,
        max_connections: int,
        retries: int,
        verify_tls: bool,
    ):
        self.url = base_url.rstrip("/") + "/render"
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.client = httpx.AsyncClient(
            timeout=self.timeout_s,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            verify=verify_tls,
            trust_env=False,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def render(self, scene_id: str, specs: Sequence[RenderSpec]) -> list[bytes]:
        data = {"meta": _json_dump({"scene_id": scene_id, "tasks": [s.task() for s in specs]})}
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.post(self.url, data=data)
                if response.status_code == 503:
                    raise RuntimeError("render service busy")
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not content_type.lower().startswith("multipart/"):
                    raise RuntimeError(
                        f"unexpected response content type {content_type!r}"
                    )
                meta, images = _decode_multipart_png_bytes(content_type, response.content)
                if meta.get("error"):
                    raise RuntimeError(f"render service error: {meta['error']}")
                if len(images) != len(specs):
                    raise RuntimeError(
                        f"scene {scene_id}: requested {len(specs)} images, got "
                        f"{len(images)} (meta={meta})"
                    )
                return images
            except BaseException as exc:  # keep KeyboardInterrupt/CancelledError visible
                last_error = exc
                if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                    raise
                if attempt == self.retries:
                    break
                await asyncio.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(
            f"scene {scene_id}: render failed after {self.retries + 1} attempts"
        ) from last_error


@dataclass
class RenderProgress:
    total: int
    cached: int = 0
    rendered: int = 0
    materialized: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_report: int = 0

    def report(self, *, force: bool = False) -> None:
        done = self.cached + self.rendered
        if not force and done - self.last_report < 500:
            return
        elapsed = max(time.monotonic() - self.started_at, 1e-6)
        print(
            f"[render] poses={done}/{self.total} cached={self.cached} "
            f"new={self.rendered} links={self.materialized} "
            f"rate={done / elapsed:.1f} pose/s",
            flush=True,
        )
        self.last_report = done


def _materialize_spec(spec: RenderSpec, cache_root: Path, progress: RenderProgress) -> None:
    cache_path = cache_root / spec.cache_rel
    for output in sorted(spec.logical_outputs):
        _hardlink_or_copy(cache_path, output)
        progress.materialized += 1


async def _render_registry(
    registry: PoseRegistry,
    *,
    client_url: str,
    scene_concurrency: int,
    batch_size: int,
    timeout_s: float,
    retries: int,
    verify_tls: bool,
) -> RenderProgress:
    grouped = registry.by_scene()
    progress = RenderProgress(total=len(registry))
    semaphore = asyncio.Semaphore(max(1, int(scene_concurrency)))
    client = RawRenderClient(
        client_url,
        timeout_s=timeout_s,
        max_connections=max(8, scene_concurrency * 2),
        retries=retries,
        verify_tls=verify_tls,
    )

    async def render_scene(scene_id: str, specs: list[RenderSpec]) -> None:
        async with semaphore:
            missing: list[RenderSpec] = []
            for spec in specs:
                cache_path = registry.cache_root / spec.cache_rel
                if _valid_png(cache_path):
                    progress.cached += 1
                    _materialize_spec(spec, registry.cache_root, progress)
                    progress.report()
                else:
                    missing.append(spec)

            # Habitat bakes projection parameters into the simulator.  Keep equal
            # intrinsics/resolution together so a scene normally rebuilds only once
            # for top-down and once for perspective views.
            projection_groups: dict[str, list[RenderSpec]] = defaultdict(list)
            for spec in missing:
                projection_groups[_projection_digest(spec)].append(spec)

            for projection_key in sorted(projection_groups):
                projection_specs = projection_groups[projection_key]
                for start in range(0, len(projection_specs), batch_size):
                    chunk = projection_specs[start : start + batch_size]
                    payloads = await client.render(scene_id, chunk)
                    for spec, payload in zip(chunk, payloads):
                        cache_path = registry.cache_root / spec.cache_rel
                        _atomic_write_bytes(cache_path, payload)
                        progress.rendered += 1
                        _materialize_spec(spec, registry.cache_root, progress)
                        progress.report()

    try:
        await asyncio.gather(
            *(render_scene(scene_id, specs) for scene_id, specs in sorted(grouped.items()))
        )
    finally:
        await client.close()
    progress.report(force=True)
    return progress


def _copy_jsonls(src_root: Path, out_root: Path, task: str | None = None) -> list[str]:
    copied: list[str] = []
    pattern = f"{task}_*.jsonl" if task else "*.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_root.glob(pattern)):
        dst = out_root / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)
    return copied


def _write_pose_index(registry: PoseRegistry, path: Path) -> None:
    lines: list[str] = []
    for spec in sorted(registry.values(), key=lambda s: (s.scene_id, s.digest)):
        lines.append(
            _json_dump(
                {
                    "scene_id": spec.scene_id,
                    "pose_id": spec.digest,
                    "path": (Path("pose_cache") / spec.cache_rel).as_posix(),
                    "size": list(spec.size),
                    "intrinsics": spec.intrinsics,
                    "extrinsics_c2w": spec.extrinsics,
                }
            )
        )
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _write_manifest(path: Path, data: Mapping[str, Any]) -> None:
    payload = {
        "format_version": 1,
        "renderer": "habitat-sim",
        "image_size": [512, 512],
        **dict(data),
    }
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _build_base_registry(
    src_root: Path,
    base_out: Path,
    *,
    scannet_root: Path,
    limit_scenes: int,
) -> tuple[PoseRegistry, dict[str, Any]]:
    registry = PoseRegistry(base_out / "pose_cache")
    scene_to_metas: dict[str, list[Path]] = defaultdict(list)
    for meta_path in sorted(src_root.glob("scene*/sample_*/meta.json")):
        scene_to_metas[meta_path.parent.parent.name].append(meta_path)

    scene_ids = sorted(scene_to_metas)
    if limit_scenes > 0:
        scene_ids = scene_ids[:limit_scenes]

    logical_images = 0
    sample_count = 0
    for scene_id in scene_ids:
        top_down_added = False
        for meta_path in scene_to_metas[scene_id]:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sample_rel = meta_path.parent.relative_to(src_root)
            dst_meta = base_out / sample_rel / "meta.json"
            dst_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta_path, dst_meta)
            sample_count += 1

            initial = meta["initial"]
            registry.add(
                scene_id,
                initial["intrinsics"],
                initial["pose_c2w"],
                logical_output=base_out / sample_rel / initial["image"],
            )
            logical_images += 1
            for option in meta["options"]:
                registry.add(
                    scene_id,
                    option["intrinsics"],
                    option["pose_c2w"],
                    logical_output=base_out / sample_rel / option["image"],
                )
                logical_images += 1

            if not top_down_added:
                top_down = meta["top_down"]
                registry.add(
                    scene_id,
                    top_down["intrinsics"],
                    top_down["pose_c2w"],
                    logical_output=base_out / scene_id / "top_down_view.png",
                )
                logical_images += 1
                top_down_added = True

    all_source_scenes = sorted(p.name for p in src_root.glob("scene*") if p.is_dir())
    metadata_less_scenes = sorted(set(all_source_scenes) - set(scene_to_metas))
    derived_topdown_scenes: list[str] = []
    if limit_scenes <= 0:
        for scene_id in metadata_less_scenes:
            fallback = _derive_top_down_from_mesh(scannet_root, scene_id)
            scene_out = base_out / scene_id
            scene_out.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                scene_out / "top_down_meta.json",
                json.dumps(fallback, ensure_ascii=False, indent=2) + "\n",
            )
            registry.add(
                scene_id,
                fallback["intrinsics"],
                fallback["pose_c2w"],
                logical_output=scene_out / "top_down_view.png",
            )
            logical_images += 1
            derived_topdown_scenes.append(scene_id)

    summary = {
        "stage": "base",
        "source_samples": sample_count,
        "source_scenes": len(scene_ids) + len(derived_topdown_scenes),
        "logical_images": logical_images,
        "unique_poses": len(registry),
        "derived_topdown_scenes": derived_topdown_scenes,
        "skipped_source_scene_dirs": (
            metadata_less_scenes if limit_scenes > 0 else []
        ),
    }
    return registry, summary


def _action_names(letters: Sequence[str], supplied: Sequence[str] | None) -> list[str]:
    if supplied is not None and len(supplied) == len(letters):
        return [str(x) for x in supplied]
    return [ACTION_NAMES.get(letter, letter) for letter in letters]


def _replay_sequence(
    registry: PoseRegistry,
    *,
    scene_id: str,
    init_detail: Mapping[str, Any],
    letters: Sequence[str],
    names: Sequence[str] | None,
    step_translation: float,
    step_rotation_deg: float,
) -> list[dict[str, Any]]:
    vm = ViewManipulator(
        step_translation=step_translation,
        step_rotation_deg=step_rotation_deg,
        world_up_axis="Z",
        is_discrete=True,
        is_snap_every_step=True,
        image_y_down=True,
    )
    vm.reset(np.asarray(init_detail["c2w_extrinsics"], dtype=np.float64))
    intrinsics = init_detail["c2w_intrinsics"]
    action_names = _action_names(letters, names)
    steps: list[dict[str, Any]] = []
    for index, (letter, action_name) in enumerate(zip(letters, action_names), 1):
        pose = vm.step(str(letter))
        spec = registry.add(scene_id, intrinsics, pose)
        steps.append(
            {
                "step_index": index,
                "action_letter": str(letter),
                "action_name": action_name,
                "path": (Path("pose_cache") / spec.cache_rel).as_posix(),
                "pose_id": spec.digest,
                "c2w_extrinsics": pose.tolist(),
                "c2w_intrinsics": _matrix_list(intrinsics),
            }
        )
    return steps


def _validate_final_pose(row: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> float:
    if not steps:
        raise ValueError(f"empty action sequence for sample {row.get('sample_id')}")
    expected = np.asarray(
        row["image_detail"]["target_view"]["c2w_extrinsics"], dtype=np.float64
    )
    actual = np.asarray(steps[-1]["c2w_extrinsics"], dtype=np.float64)
    error = float(np.max(np.abs(expected - actual)))
    if error > 1e-6:
        raise ValueError(
            f"sample {row.get('sample_id')}: replayed final pose differs from target "
            f"(max_abs={error:.3e})"
        )
    return error


def _prepare_intermediate_links(base_out: Path, intermediate_out: Path) -> None:
    intermediate_out.mkdir(parents=True, exist_ok=True)
    _ensure_relative_symlink(intermediate_out / "pose_cache", base_out / "pose_cache")
    for scene_dir in sorted(p for p in base_out.glob("scene*") if p.is_dir()):
        _ensure_relative_symlink(intermediate_out / scene_dir.name, scene_dir)


def _build_intermediate_registry(
    *,
    task: str,
    src_root: Path,
    base_out: Path,
    intermediate_out: Path,
    limit_rows: int,
) -> tuple[PoseRegistry, dict[str, Any]]:
    if task not in {"path_to_view", "view_to_path"}:
        raise ValueError(task)
    _prepare_intermediate_links(base_out, intermediate_out)
    registry = PoseRegistry(base_out / "pose_cache")
    total_rows = 0
    total_steps = 0
    max_final_pose_error = 0.0
    split_counts: dict[str, int] = {}

    for split in TASK_SPLITS:
        src_jsonl = src_root / f"{task}_{split}.jsonl"
        if not src_jsonl.is_file():
            raise FileNotFoundError(src_jsonl)
        out_rows: list[str] = []
        split_count = 0
        for row in _iter_jsonl(src_jsonl):
            if limit_rows > 0 and total_rows >= limit_rows:
                break
            scene_id = str(row["scene_id"])
            init_detail = row["image_detail"]["init_view"]
            meta = row["meta"]
            step_translation = float(meta["step_translation_m"])
            step_rotation_deg = float(meta["step_rotation_deg"])

            if task == "path_to_view":
                letters = list(meta["gt_action_seq_letters"])
                names = meta.get("gt_action_seq_names")
                steps = _replay_sequence(
                    registry,
                    scene_id=scene_id,
                    init_detail=init_detail,
                    letters=letters,
                    names=names,
                    step_translation=step_translation,
                    step_rotation_deg=step_rotation_deg,
                )
                max_final_pose_error = max(max_final_pose_error, _validate_final_pose(row, steps))
                row["intermediate_image_path"] = [step["path"] for step in steps]
                row["intermediate_image_detail"] = steps
                row["meta"]["p2v_intermediate"] = {
                    "includes_initial_view": False,
                    "includes_final_target_view": True,
                    "num_steps": len(steps),
                    "image_paths": row["intermediate_image_path"],
                }
                total_steps += len(steps)
            else:
                option_letters = meta["option_action_seq_letters"]
                option_names = meta.get("option_action_seq_names", {})
                option_steps: dict[str, list[dict[str, Any]]] = {}
                for label in sorted(option_letters):
                    option_steps[label] = _replay_sequence(
                        registry,
                        scene_id=scene_id,
                        init_detail=init_detail,
                        letters=list(option_letters[label]),
                        names=option_names.get(label),
                        step_translation=step_translation,
                        step_rotation_deg=step_rotation_deg,
                    )
                    total_steps += len(option_steps[label])
                gold = str(row["gt_answer"])[0].upper()
                max_final_pose_error = max(
                    max_final_pose_error, _validate_final_pose(row, option_steps[gold])
                )
                row["intermediate_image_path"] = {
                    label: [step["path"] for step in steps]
                    for label, steps in option_steps.items()
                }
                row["intermediate_image_detail"] = option_steps
                row["meta"]["v2p_intermediate"] = {
                    "includes_initial_view": False,
                    "includes_each_option_final_view": True,
                    "num_steps_by_option": {
                        label: len(steps) for label, steps in option_steps.items()
                    },
                    "image_paths_by_option": row["intermediate_image_path"],
                }

            out_rows.append(_json_dump(row))
            total_rows += 1
            split_count += 1

        _atomic_write_text(
            intermediate_out / src_jsonl.name,
            "\n".join(out_rows) + ("\n" if out_rows else ""),
        )
        split_counts[split] = split_count
        if limit_rows > 0 and total_rows >= limit_rows:
            break

    stage_name = "p2v" if task == "path_to_view" else "v2p"
    summary = {
        "stage": stage_name,
        "task": task,
        "rows": total_rows,
        "rows_by_split": split_counts,
        "logical_intermediate_steps": total_steps,
        "unique_poses": len(registry),
        "max_final_pose_error": max_final_pose_error,
        "base_dataset": os.path.relpath(base_out, start=intermediate_out),
    }
    return registry, summary


def _default_paths() -> tuple[Path, Path, Path, Path]:
    root = Path(os.environ.get("VIEWSUITE_ROOT", Path.cwd())).resolve()
    src = root / "data" / "viewsuite_15k"
    shared = Path(os.environ.get("VIEWSUITE_DATA_ROOT", root / "data")).resolve()
    return (
        src,
        shared / "viewsuite15k-habitat",
        shared / "viewsuite15k-habitat-p2v-intermediate",
        shared / "viewsuite15k-habitat-v2p-intermediate",
    )


async def _run_stage(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    src_root = Path(args.src_root).resolve()
    scannet_root = Path(args.scannet_root).resolve()
    base_out = Path(args.base_out).resolve()
    p2v_out = Path(args.p2v_out).resolve()
    v2p_out = Path(args.v2p_out).resolve()

    if stage == "base":
        base_out.mkdir(parents=True, exist_ok=True)
        copied_jsonls = _copy_jsonls(src_root, base_out)
        registry, summary = _build_base_registry(
            src_root,
            base_out,
            scannet_root=scannet_root,
            limit_scenes=args.limit_scenes,
        )
        summary["copied_jsonls"] = copied_jsonls
        index_path = base_out / "pose_index_base.jsonl"
        manifest_path = base_out / "dataset_manifest.json"
    elif stage == "p2v":
        registry, summary = _build_intermediate_registry(
            task="path_to_view",
            src_root=src_root,
            base_out=base_out,
            intermediate_out=p2v_out,
            limit_rows=args.limit_rows,
        )
        index_path = p2v_out / "pose_index_p2v.jsonl"
        manifest_path = p2v_out / "dataset_manifest.json"
    elif stage == "v2p":
        registry, summary = _build_intermediate_registry(
            task="view_to_path",
            src_root=src_root,
            base_out=base_out,
            intermediate_out=v2p_out,
            limit_rows=args.limit_rows,
        )
        index_path = v2p_out / "pose_index_v2p.jsonl"
        manifest_path = v2p_out / "dataset_manifest.json"
    else:
        raise ValueError(stage)

    print(f"[prepare:{stage}] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    _write_pose_index(registry, index_path)
    progress = await _render_registry(
        registry,
        client_url=args.client_url,
        scene_concurrency=args.scene_concurrency,
        batch_size=args.batch_size,
        timeout_s=args.timeout,
        retries=args.retries,
        verify_tls=not args.insecure_tls,
    )
    summary.update(
        {
            "client_url": args.client_url,
            "renderer_appearance": args.renderer_appearance,
            "cached_poses": progress.cached,
            "rendered_poses": progress.rendered,
            "materialized_links": progress.materialized,
            "completed": True,
        }
    )
    _write_manifest(manifest_path, summary)
    print(f"[done:{stage}] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    src, base, p2v, v2p = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("base", "p2v", "v2p", "all"), default="all")
    parser.add_argument("--src-root", default=str(src))
    parser.add_argument(
        "--scannet-root",
        default=str(Path(os.environ.get("VIEWSUITE_ROOT", Path.cwd())).resolve() / "data" / "scannet" / "scans"),
    )
    parser.add_argument("--base-out", default=str(base))
    parser.add_argument("--p2v-out", default=str(p2v))
    parser.add_argument("--v2p-out", default=str(v2p))
    parser.add_argument("--client-url", default="http://127.0.0.1:8813")
    parser.add_argument(
        "--renderer-appearance",
        choices=("baked", "legacy_lit", "raw_unlit"),
        default=os.environ.get("SCANNET_HABITAT_APPEARANCE", "baked"),
        help="Appearance profile configured on the render service; recorded as provenance.",
    )
    parser.add_argument("--scene-concurrency", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help="Disable certificate verification for a trusted self-signed render service.",
    )
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stages = ("base", "p2v", "v2p") if args.stage == "all" else (args.stage,)
    for stage in stages:
        asyncio.run(_run_stage(args, stage))


if __name__ == "__main__":
    main()
