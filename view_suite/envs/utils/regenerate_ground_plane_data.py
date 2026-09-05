"""Rebuild an existing ViewSuite corpus for ``ground_plane_v1`` actions.

The script preserves the split of each retained sample, replays all four option
paths with the corpus-specific camera controller, renders only poses that changed,
applies cheap image-quality gates, and rewrites P2V/V2P/IVP rows consistently. Unchanged source images are reused
through read-only file symlinks; changed option renders live in a content-addressed
``ground_plane_cache`` directory.

ScanNet rows containing roll in any option are dropped by default because the
unified action space intentionally matches Habitat-GS's no-roll navigation set.

Examples:

    python -m view_suite.envs.utils.regenerate_ground_plane_data \
      --corpus scannet --src-root data/viewsuite_15k \
      --out-root data/viewagent15k_scannet_open3d_ground_plane

    python -m view_suite.envs.utils.regenerate_ground_plane_data \
      --corpus ai2thor --src-root data/ai2thor \
      --out-root data/viewagent15k_ai2thor_ground_plane
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TASKS = ("path_to_view", "view_to_path", "interactive_view_planning")
LABELS = ("A", "B", "C", "D")
UNIFIED_ACTIONS = frozenset(("w", "s", "a", "d", "y", "h", "q", "e", "r", "f"))


@dataclass(frozen=True)
class RenderSpec:
    scene_id: str
    c2w: np.ndarray
    intrinsics: np.ndarray
    relative_path: str
    width: int
    height: int


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _valid_png(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return path.stat().st_size > 32 and handle.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _movement_explanation(corpus: str) -> str:
    if corpus == "scannet":
        plane, up = "world XY", "world +Z/-Z"
    else:
        plane, up = "world XZ", "world +Y/-Y"
    return (
        "Movement mode = ground_plane_v1: forward/backward follow the yaw-only "
        f"heading on the horizontal {plane} plane; left/right strafe perpendicular "
        f"to that heading; up/down move along {up}; look up/down changes pitch only "
        "and never changes the direction of a later horizontal move."
    )


def _forward_prompt(meta: dict[str, Any], corpus: str) -> str:
    names = meta["gt_action_seq_names"]
    return (
        "Given the initial view <image> and a top-down reference <image>, after you "
        "execute the following action sequence "
        f"(translation step = {meta['step_translation_m']} m; rotation step = "
        f"{meta['step_rotation_deg']} degrees per step):\n"
        f"{_movement_explanation(corpus)}\n"
        f"[{', '.join(names)}]\n"
        "which of the following images corresponds to the result?\n"
        "A. <image>\nB. <image>\nC. <image>\nD. <image>\n"
    )


def _inverse_prompt(meta: dict[str, Any], corpus: str) -> str:
    names = meta["option_action_seq_names"]
    lines = [
        (
            "Given the initial view <image> and a top-down reference <image>, which "
            "action sequence will reach the target view <image>?"
        ),
        (
            f"(Action semantics: translation step = {meta['step_translation_m']} m; "
            f"rotation step = {meta['step_rotation_deg']} degrees per step.)"
        ),
        _movement_explanation(corpus),
    ]
    lines.extend(f"{label}. [{', '.join(names[label])}]" for label in LABELS)
    return "\n".join(lines) + "\n"


def _active_prompt(corpus: str) -> str:
    return (
        "Given the initial view <image> and a top-down reference <image>, estimate "
        "the target view's 6-DoF pose relative to the world. "
        + _movement_explanation(corpus)
    )


def _make_manipulator(
    corpus: str,
    init_c2w: np.ndarray,
    step_translation: float,
    step_rotation: float,
    pitch_limit: float,
):
    if corpus == "scannet":
        from view_suite.scannet.view_manipulator import ViewManipulator

        vm = ViewManipulator(
            step_translation=step_translation,
            step_rotation_deg=step_rotation,
            world_up_axis="Z",
            is_discrete=True,
            is_snap_every_step=True,
            image_y_down=True,
            ground_plane_movement=True,
        )
    elif corpus == "ai2thor":
        from view_suite.ai2thor.view_manipulator import ViewManipulator

        vm = ViewManipulator(
            step_translation=step_translation,
            step_rotation_deg=step_rotation,
            pitch_limit_deg=pitch_limit,
            is_discrete=True,
            is_snap_every_step=True,
            ground_plane_movement=True,
        )
    elif corpus == "habitat_gs":
        from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator

        vm = HabitatGSViewManipulator(
            step_translation=step_translation,
            step_rotation_deg=step_rotation,
            pitch_limit_deg=pitch_limit,
            discrete=True,
            ground_plane_movement=True,
        )
    else:
        raise ValueError(f"unknown corpus: {corpus}")
    vm.reset(init_c2w)
    return vm


def _replay(corpus: str, init_c2w: np.ndarray, actions: list[str], meta: dict[str, Any]):
    vm = _make_manipulator(
        corpus,
        init_c2w,
        float(meta.get("step_translation_m", 0.5)),
        float(meta.get("step_rotation_deg", 30.0)),
        float(meta.get("pitch_limit_deg", 60.0 if corpus == "habitat_gs" else 89.0)),
    )
    for action in actions:
        vm.step(action)
    return np.asarray(vm.get_pose(mode="c2w"), dtype=np.float64), vm


def _pose_digest(
    corpus: str, scene_id: str, c2w: np.ndarray, intrinsics: np.ndarray, size: int
) -> str:
    digest = hashlib.sha256()
    digest.update(corpus.encode("utf-8"))
    digest.update(b"\0")
    digest.update(scene_id.encode("utf-8"))
    digest.update(np.round(c2w, 12).astype("<f8").tobytes())
    digest.update(np.round(intrinsics, 12).astype("<f8").tobytes())
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()[:24]


def _cache_path(
    corpus: str, scene_id: str, c2w: np.ndarray, intrinsics: np.ndarray, size: int
) -> str:
    digest = _pose_digest(corpus, scene_id, c2w, intrinsics, size)
    return f"ground_plane_cache/{scene_id}/{digest}.png"


def _tag_meta(meta: dict[str, Any], corpus: str) -> None:
    meta["ground_plane_movement"] = True
    meta["action_space_version"] = "ground_plane_v1"
    meta["world_up_axis"] = "Z" if corpus == "scannet" else "Y"


def _replace_target(row: dict[str, Any], detail: dict[str, Any], corpus: str) -> None:
    row["image_detail"]["target_view"] = copy.deepcopy(detail)
    prefix = list(row.get("image_path") or [])[:2]
    row["image_path"] = prefix + [detail["path"]]
    answer = row.get("gt_answer")
    c2w = np.asarray(detail["c2w_extrinsics"], dtype=np.float64)
    if isinstance(answer, dict):
        answer["pose_c2w"] = c2w.tolist()
        if corpus == "ai2thor":
            from view_suite.ai2thor.pose_utils import c2w_to_unity_pose

            answer["unity_pose"] = c2w_to_unity_pose(c2w)
        elif corpus == "habitat_gs":
            from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator

            vm = HabitatGSViewManipulator(ground_plane_movement=True)
            vm.reset(c2w)
            answer["gs_pose"] = vm.get_state()
    elif isinstance(answer, (list, tuple)) and len(answer) == 6:
        from view_suite.scannet.utils.pose_utils import c2w_extrinsic_to_se3

        row["gt_answer"] = c2w_extrinsic_to_se3(c2w, degrees=True).tolist()


def _build_rows(
    corpus: str,
    src_root: Path,
    size: int,
    drop_roll: bool,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, RenderSpec], dict[str, Any]]:
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    render_specs: dict[str, RenderSpec] = {}
    stats: dict[str, Any] = defaultdict(int)
    dropped_roll_ids: list[str] = []

    split_order = [
        split
        for split in ("train", "dev", "eval", "test")
        if (src_root / f"path_to_view_{split}.jsonl").is_file()
    ]
    if not split_order:
        raise FileNotFoundError(f"no split path_to_view JSONLs under {src_root}")

    # A new endpoint may already have been rendered elsewhere in the accepted
    # source corpus. Reuse that exact image before asking a live renderer. This is
    # especially effective for grid-based AI2-THOR scenes.
    source_pose_paths: dict[tuple[str, str], str] = {}
    for split in split_order:
        for row in _iter_jsonl(src_root / f"path_to_view_{split}.jsonl"):
            scene_id = str(row["scene_id"])
            for detail in (row.get("image_detail") or {}).values():
                if not isinstance(detail, dict):
                    continue
                if not detail.get("path") or detail.get("c2w_extrinsics") is None:
                    continue
                c2w = np.asarray(detail["c2w_extrinsics"], dtype=np.float64)
                intrinsics = np.asarray(detail["c2w_intrinsics"], dtype=np.float64)
                digest = _pose_digest(corpus, scene_id, c2w, intrinsics, size)
                source_pose_paths.setdefault((scene_id, digest), str(detail["path"]))

    for split in split_order:
        paths = {task: src_root / f"{task}_{split}.jsonl" for task in TASKS}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing matching task JSONLs: {missing}")
        source = {task: {row["sample_id"]: row for row in _iter_jsonl(path)}
                  for task, path in paths.items()}
        outputs.update({(task, split): [] for task in TASKS})

        for p2v in source["path_to_view"].values():
            sample_id = p2v["sample_id"]
            meta = p2v.get("meta") or {}
            sequences = meta.get("option_action_seq_letters") or {}
            if set(sequences) != set(LABELS):
                raise ValueError(f"{sample_id}: expected options A-D, got {sorted(sequences)}")
            actions = [str(a).lower() for seq in sequences.values() for a in seq]
            unsupported = sorted(set(actions) - UNIFIED_ACTIONS)
            if unsupported:
                if drop_roll and set(unsupported) <= {"t", "g"}:
                    stats["dropped_roll_samples"] += 1
                    dropped_roll_ids.append(str(sample_id))
                    continue
                raise ValueError(f"{sample_id}: unsupported actions {unsupported}")

            init_detail = p2v["image_detail"]["init_view"]
            init_c2w = np.asarray(init_detail["c2w_extrinsics"], dtype=np.float64)
            option_details: dict[str, dict[str, Any]] = {}
            for index, label in enumerate(LABELS):
                old_detail = p2v["image_detail"][f"view_{index}"]
                new_c2w, _ = _replay(corpus, init_c2w, list(sequences[label]), meta)
                old_c2w = np.asarray(old_detail["c2w_extrinsics"], dtype=np.float64)
                intrinsics = np.asarray(old_detail["c2w_intrinsics"], dtype=np.float64)
                detail = copy.deepcopy(old_detail)
                detail["c2w_extrinsics"] = new_c2w.tolist()
                if np.allclose(new_c2w, old_c2w, atol=1e-7, rtol=0.0):
                    stats["reused_option_images"] += 1
                else:
                    stats["changed_option_poses"] += 1
                    scene_id = str(p2v["scene_id"])
                    digest = _pose_digest(corpus, scene_id, new_c2w,
                                          intrinsics, size)
                    source_rel = source_pose_paths.get((scene_id, digest))
                    if source_rel is not None:
                        detail["path"] = source_rel
                        stats["reused_source_pose_images"] += 1
                    else:
                        rel = _cache_path(corpus, scene_id, new_c2w,
                                          intrinsics, size)
                        detail["path"] = rel
                        render_specs.setdefault(
                            rel,
                            RenderSpec(scene_id, new_c2w, intrinsics,
                                       rel, size, size),
                        )
                option_details[label] = detail

            gt_label = str(meta["gt_label"]).upper()
            if gt_label not in option_details:
                raise ValueError(f"{sample_id}: invalid gt_label={gt_label!r}")

            p2v_out = copy.deepcopy(p2v)
            _tag_meta(p2v_out.setdefault("meta", {}), corpus)
            for index, label in enumerate(LABELS):
                p2v_out["image_detail"][f"view_{index}"] = copy.deepcopy(
                    option_details[label]
                )
            if "target_view" in p2v_out["image_detail"]:
                p2v_out["image_detail"]["target_view"] = copy.deepcopy(
                    option_details[gt_label]
                )
            p2v_out["image_path"] = [
                p2v_out["image_detail"]["init_view"]["path"],
                p2v_out["image_detail"]["top_down_view"]["path"],
                *(option_details[label]["path"] for label in LABELS),
            ]
            p2v_out["prompt"] = _forward_prompt(p2v_out["meta"], corpus)

            try:
                v2p_out = copy.deepcopy(source["view_to_path"][sample_id])
                ivp_out = copy.deepcopy(source["interactive_view_planning"][sample_id])
            except KeyError as exc:
                raise ValueError(f"{sample_id}: missing matching task row") from exc
            for row in (v2p_out, ivp_out):
                _tag_meta(row.setdefault("meta", {}), corpus)
                _replace_target(row, option_details[gt_label], corpus)
            v2p_out["prompt"] = _inverse_prompt(p2v_out["meta"], corpus)
            ivp_out["prompt"] = _active_prompt(corpus)

            outputs[("path_to_view", split)].append(p2v_out)
            outputs[("view_to_path", split)].append(v2p_out)
            outputs[("interactive_view_planning", split)].append(ivp_out)
            stats["kept_samples"] += 1

    stats["unique_changed_renders"] = len(render_specs)
    if dropped_roll_ids:
        stats["dropped_roll_sample_ids"] = sorted(dropped_roll_ids)
    return outputs, render_specs, dict(stats)


def _default_client_url(corpus: str, root: Path) -> str:
    filename = {
        "scannet": "client_url.txt",
        "ai2thor": "client_url_ai2thor.txt",
        "habitat_gs": "client_url_habitat_gs.txt",
    }[corpus]
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"client URL not supplied and {path} does not exist")
    return path.read_text(encoding="utf-8").strip()


def _make_renderer(corpus: str, scene_id: str, client_url: str):
    if corpus == "scannet":
        from view_suite.scannet.unified_renderer import UnifiedRender

        return UnifiedRender("client", None, client_url, None, scene_id)
    if corpus == "ai2thor":
        from view_suite.ai2thor.ai2thor_unified_renderer import AI2ThorUnifiedRender

        return AI2ThorUnifiedRender(client_url=client_url, scene_id=scene_id)
    from view_suite.habitat_gs.habitat_gs_unified_renderer import HabitatGSUnifiedRender

    return HabitatGSUnifiedRender(client_url=client_url, scene_id=scene_id)


def _render_task(corpus: str, spec: RenderSpec) -> dict[str, Any]:
    if corpus == "ai2thor":
        from view_suite.ai2thor.pose_utils import build_render_task

        return build_render_task(spec.c2w, spec.intrinsics,
                                 width=spec.width, height=spec.height)
    K = spec.intrinsics[:3, :3] if spec.intrinsics.shape == (4, 4) else spec.intrinsics
    return {
        "mode": "cam_param",
        "intrinsics": K.tolist(),
        "extrinsics": spec.c2w.tolist(),
        "size": [spec.width, spec.height],
    }


async def _render_all(
    corpus: str,
    specs: dict[str, RenderSpec],
    out_root: Path,
    client_url: str,
    batch_size: int,
    max_scene_concurrency: int,
) -> tuple[int, list[str]]:
    pending: dict[str, list[RenderSpec]] = defaultdict(list)
    cached = 0
    for spec in specs.values():
        if _valid_png(out_root / spec.relative_path):
            cached += 1
        else:
            pending[spec.scene_id].append(spec)

    semaphore = asyncio.Semaphore(max_scene_concurrency)
    rendered = 0
    counter_lock = asyncio.Lock()

    async def render_scene(scene_id: str, scene_specs: list[RenderSpec]) -> None:
        nonlocal rendered
        async with semaphore:
            renderer = _make_renderer(corpus, scene_id, client_url)
            try:
                for start in range(0, len(scene_specs), batch_size):
                    batch = scene_specs[start:start + batch_size]
                    images = await renderer.render_tasks(
                        [_render_task(corpus, spec) for spec in batch]
                    )
                    if len(images) != len(batch):
                        raise RuntimeError(
                            f"{scene_id}: renderer returned {len(images)} images for "
                            f"{len(batch)} requests"
                        )
                    for spec, image in zip(batch, images):
                        target = out_root / spec.relative_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tmp = target.with_name(f".{target.stem}.tmp-{os.getpid()}.png")
                        image.save(tmp, format="PNG")
                        os.replace(tmp, target)
                    async with counter_lock:
                        rendered += len(batch)
                        print(
                            f"[render] {rendered}/{sum(len(v) for v in pending.values())} "
                            f"new images ({scene_id})",
                            flush=True,
                        )
            finally:
                await renderer.close()

    await asyncio.gather(*(render_scene(scene, values)
                           for scene, values in sorted(pending.items())))
    return cached, rendered


def _render_ai2thor_local(
    specs: dict[str, RenderSpec], out_root: Path, gpu_id: int
) -> tuple[int, int]:
    """Render AI2-THOR poses directly, avoiding a busy shared HTTP fleet."""
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    from view_suite.ai2thor.pose_utils import c2w_to_unity_pose, fov_from_K

    pending: dict[str, list[RenderSpec]] = defaultdict(list)
    cached = 0
    for spec in specs.values():
        if _valid_png(out_root / spec.relative_path):
            cached += 1
        else:
            pending[spec.scene_id].append(spec)
    if not pending:
        return cached, 0

    # CloudRendering interprets the visible device as index zero.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    first_scene = next(iter(sorted(pending)))
    first_spec = pending[first_scene][0]
    first_fov = fov_from_K(first_spec.intrinsics, first_spec.width)
    controller = Controller(
        platform=CloudRendering,
        agentMode="default",
        scene=first_scene,
        width=first_spec.width,
        height=first_spec.height,
        fieldOfView=first_fov,
    )
    rendered = 0
    total = sum(len(values) for values in pending.values())
    try:
        for scene_id, scene_specs in sorted(pending.items()):
            controller.reset(scene=scene_id)
            seed_spec = scene_specs[0]
            seed_pose = c2w_to_unity_pose(seed_spec.c2w)
            controller.step(
                action="AddThirdPartyCamera",
                position=seed_pose["position"],
                rotation=seed_pose["rotation"],
                fieldOfView=fov_from_K(seed_spec.intrinsics, seed_spec.width),
            )
            for spec in scene_specs:
                pose = c2w_to_unity_pose(spec.c2w)
                event = controller.step(
                    action="UpdateThirdPartyCamera",
                    thirdPartyCameraId=0,
                    position=pose["position"],
                    rotation=pose["rotation"],
                    fieldOfView=fov_from_K(spec.intrinsics, spec.width),
                )
                if not event.metadata.get("lastActionSuccess", False):
                    raise RuntimeError(
                        f"{scene_id}: UpdateThirdPartyCamera failed: "
                        f"{event.metadata.get('errorMessage', 'unknown error')}"
                    )
                frames = event.third_party_camera_frames
                if not frames:
                    raise RuntimeError(f"{scene_id}: no third-party camera frame")
                target = out_root / spec.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(f".{target.stem}.tmp-{os.getpid()}.png")
                Image.fromarray(np.asarray(frames[0])).save(tmp, format="PNG")
                os.replace(tmp, target)
                rendered += 1
                if rendered % 50 == 0 or rendered == total:
                    print(f"[render-local] {rendered}/{total} new images", flush=True)
    finally:
        with suppress(Exception):
            controller.stop()
    return cached, rendered


def _link_source_images(
    outputs: dict[tuple[str, str], list[dict[str, Any]]],
    src_root: Path,
    out_root: Path,
) -> int:
    """Materialize only referenced legacy images as symlinks.

    Earlier versions of this converter linked whole scene directories. Replace
    only those exact links on resume, then create real directories so stale
    source ``meta.json`` files are not exposed as ground-plane metadata.
    """
    relative_paths: set[str] = set()
    for rows in outputs.values():
        for row in rows:
            for detail in (row.get("image_detail") or {}).values():
                if not isinstance(detail, dict) or not detail.get("path"):
                    continue
                rel = os.path.normpath(str(detail["path"]))
                if rel.startswith("../") or os.path.isabs(rel):
                    raise ValueError(f"unsafe image path in source row: {rel!r}")
                if not rel.startswith("ground_plane_cache/"):
                    relative_paths.add(rel)

    # Clean up whole-scene symlinks created by converter versions before the
    # per-file layout. Only links resolving directly under this exact source root
    # are touched.
    for path in out_root.iterdir():
        if path.is_symlink() and path.resolve().parent == src_root:
            path.unlink()

    top_levels = {Path(rel).parts[0] for rel in relative_paths}
    for name in top_levels:
        path = out_root / name
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"expected a directory at {path}")

    count = 0
    for rel in sorted(relative_paths):
        source = (src_root / rel).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = out_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() == source:
            count += 1
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to replace {target}")
        target.symlink_to(source)
        count += 1
    return count


def _write_outputs(
    outputs: dict[tuple[str, str], list[dict[str, Any]]], out_root: Path
) -> dict[str, int]:
    counts: dict[str, int] = {}
    splits = [split for split in ("train", "dev", "eval", "test")
              if ("path_to_view", split) in outputs]
    for (task, split), rows in outputs.items():
        path = out_root / f"{task}_{split}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        _atomic_write_text(path, payload)
        counts[path.name] = len(rows)
    for task in TASKS:
        rows = [row for split in splits for row in outputs[(task, split)]]
        path = out_root / f"{task}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        _atomic_write_text(path, payload)
        counts[path.name] = len(rows)
    return counts


def _dominant_pixel_fraction(image: Image.Image, bits: int = 3) -> float:
    array = np.asarray(image.convert("RGB"), dtype=np.uint16) >> bits
    keys = ((array[..., 0].astype(np.int32) << 12)
            | (array[..., 1].astype(np.int32) << 6)
            | array[..., 2].astype(np.int32))
    counts = np.bincount(keys.ravel())
    return float(counts.max()) / float(keys.size)


def _filter_changed_image_quality(
    outputs: dict[tuple[str, str], list[dict[str, Any]]],
    out_root: Path,
    *,
    min_image_std: float,
    min_image_mean: float,
    max_dominant_pixel_fraction: float,
) -> tuple[int, list[str]]:
    """Drop a sample if one of its newly rendered options fails cheap gates.

    Source images have already passed their corpus's original filtering pass, so
    only files in ``ground_plane_cache`` need to be checked here.
    """
    quality: dict[str, bool] = {}
    bad_ids: set[str] = set()
    checked = 0
    for (task, _split), rows in outputs.items():
        if task != "path_to_view":
            continue
        for row in rows:
            for index in range(4):
                rel = str(row["image_detail"][f"view_{index}"]["path"])
                if not rel.startswith("ground_plane_cache/"):
                    continue
                if rel not in quality:
                    path = out_root / rel
                    with Image.open(path) as image:
                        array = np.asarray(image.convert("RGB"), dtype=np.float32)
                        quality[rel] = (
                            float(array.std()) >= min_image_std
                            and float(array.mean()) >= min_image_mean
                            and _dominant_pixel_fraction(image)
                            <= max_dominant_pixel_fraction
                        )
                    checked += 1
                if not quality[rel]:
                    bad_ids.add(str(row["sample_id"]))
                    break

    if bad_ids:
        for key, rows in outputs.items():
            outputs[key] = [row for row in rows
                            if str(row["sample_id"]) not in bad_ids]
    return checked, sorted(bad_ids)


def run(args: argparse.Namespace) -> None:
    repo_root = Path(os.environ.get("VIEWSUITE_ROOT", Path.cwd())).resolve()
    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    if src_root == out_root:
        raise ValueError("--out-root must differ from --src-root")

    outputs, specs, stats = _build_rows(
        args.corpus, src_root, args.size, args.drop_roll
    )
    planned_stats = {key: value for key, value in stats.items()
                     if not key.endswith("_sample_ids")}
    print(json.dumps({"phase": "planned", **planned_stats}, indent=2), flush=True)
    if args.dry_run:
        return

    out_root.mkdir(parents=True, exist_ok=True)
    linked_images = _link_source_images(outputs, src_root, out_root)
    if args.render_backend == "local":
        if args.corpus != "ai2thor":
            raise ValueError("--render-backend=local is currently supported for ai2thor only")
        cached, rendered = _render_ai2thor_local(specs, out_root, args.gpu_id)
        client_url = None
    else:
        client_url = args.client_url or _default_client_url(args.corpus, repo_root)
        cached, rendered = asyncio.run(
            _render_all(
                args.corpus,
                specs,
                out_root,
                client_url,
                args.batch_size,
                args.max_scene_concurrency,
            )
        )
    quality_checked, quality_dropped_ids = _filter_changed_image_quality(
        outputs,
        out_root,
        min_image_std=args.min_image_std,
        min_image_mean=args.min_image_mean,
        max_dominant_pixel_fraction=args.max_dominant_pixel_fraction,
    )
    stats["quality_checked_changed_images"] = quality_checked
    stats["dropped_quality_samples"] = len(quality_dropped_ids)
    stats["dropped_quality_sample_ids"] = quality_dropped_ids
    stats["kept_samples"] -= len(quality_dropped_ids)
    counts = _write_outputs(outputs, out_root)
    manifest = {
        "action_space_version": "ground_plane_v1",
        "ground_plane_movement": True,
        "world_up_axis": "Z" if args.corpus == "scannet" else "Y",
        "corpus": args.corpus,
        "render_backend": args.render_backend,
        "source_root": str(src_root),
        "source_images_are_symlinked": True,
        "linked_source_images": linked_images,
        "drop_roll": bool(args.drop_roll),
        "quality_thresholds": {
            "min_image_std": args.min_image_std,
            "min_image_mean": args.min_image_mean,
            "max_dominant_pixel_fraction": args.max_dominant_pixel_fraction,
        },
        "cached_changed_renders": cached,
        "new_changed_renders": rendered,
        **stats,
        "jsonl_counts": counts,
    }
    _atomic_write_text(
        out_root / "dataset_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps({"phase": "complete", **manifest}, indent=2), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        choices=("scannet", "ai2thor", "habitat_gs"))
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--client-url", default="")
    parser.add_argument("--render-backend", choices=("client", "local"),
                        default="client")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-scene-concurrency", type=int, default=8)
    parser.add_argument("--min-image-std", type=float, default=8.0)
    parser.add_argument("--min-image-mean", type=float, default=12.0)
    parser.add_argument("--max-dominant-pixel-fraction", type=float, default=0.8)
    parser.add_argument("--drop-roll", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
