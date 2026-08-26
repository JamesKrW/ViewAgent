"""Idea-1: atomize the view-graph into a SINGLE-ACTION-only graph via rendering.

The default builder splits multi-action turns into virtual-node chains and then
``_collapse_virtual_nodes`` re-joins them into ``"a | b | c"`` edges. That leaves
multi-action edges in the SFT data and blocks clean cross-trajectory merging of
the intermediate poses (they have no image, so they stay virtual).

This pass instead makes the graph fully atomic:

  1. For every multi-action edge  u --[a1|a2|...|aN]--> v, compute the TRUE
     intermediate camera poses by replaying each single action from u's pose
     with ``ViewManipulator`` (0.5 m / 30 deg steps).
  2. RENDER each intermediate pose (same render fleet / intrinsics the rollout
     used) so it becomes a REAL node (pose + image).
  3. Replace the multi-action edge with the single-action chain
     u --[a1]--> n1 --[a2]--> ... --[aN]--> v.
  4. If ANY intermediate render in a chain fails, the whole multi-action edge is
     DROPPED -- the final graph is guaranteed 100% single-action.
  5. Re-run the full node merge + dedup (same pose-similarity the builder uses)
     so the freshly rendered intermediate nodes merge with existing / each other
     across trajectories.

Runs once, after graph build, before path sampling.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# action name -> ViewManipulator key code (inverse of the env _KEYMAP)
_ACTION_TO_CODE = {
    "move_forward": "w", "move_backward": "s",
    "move_right": "d",   "move_left": "a",
    "move_up": "y",      "move_down": "h",
    "turn_left": "q",    "turn_right": "e",
    "look_up": "r",      "look_down": "f",
    "rotate_ccw": "t",   "rotate_cw": "g",
}


# Per-corpus adapters. Each corpus has its own camera convention, render task
# schema and intrinsics; mismatches render successfully but wrongly.
#   scannet     Z-up, cam_param tasks
#   habitat_gs  Y-up, yaw-only body + pitching sensor, per-scene step, cam_param
#   ai2thor     Unity pose dicts, pose{position,rotation} tasks


class _ScanNetAdapter:
    corpus = "scannet"

    @staticmethod
    def manipulator(src_se3, step_t, step_r):
        from view_suite.scannet.view_manipulator import ViewManipulator
        # Match gym_scannet_tool_env.
        vm = ViewManipulator(
            step_translation=step_t, step_rotation_deg=step_r,
            world_up_axis="Z", is_discrete=True,
            is_snap_every_step=True, image_y_down=True,
        )
        vm.set_se3(np.asarray(src_se3, dtype=np.float64), degrees=True)
        return vm

    @staticmethod
    def snapshot(vm):
        return (vm.get_se3(degrees=True).tolist(),
                np.asarray(vm.get_pose(mode="c2w"), dtype=np.float64))

    @staticmethod
    def renderer(scene_id, cfg):
        from view_suite.scannet.unified_renderer import UnifiedRender
        return UnifiedRender(
            render_backend="client", scannet_root=cfg.get("scannet_root"),
            client_url=cfg["client_url"], client_origin=cfg.get("client_origin"),
            scene_id=scene_id)

    @staticmethod
    def tasks(c2w_list, K, size):
        return [{"mode": "cam_param", "intrinsics": K,
                 "extrinsics": E.tolist(), "size": [int(size), int(size)]}
                for E in c2w_list]

    @staticmethod
    def intrinsics():
        from view_suite.envs.utils.scannet_utils import default_intrinsics
        return _K3(default_intrinsics())


class _HabitatGSAdapter(_ScanNetAdapter):
    corpus = "habitat_gs"

    @staticmethod
    def manipulator(src_se3, step_t, step_r):
        from view_suite.habitat_gs.view_manipulator import HabitatGSViewManipulator
        vm = HabitatGSViewManipulator(
            step_translation=step_t, step_rotation_deg=step_r, discrete=True)
        vm.set_se3(np.asarray(src_se3, dtype=np.float64), degrees=True)
        return vm

    @staticmethod
    def intrinsics():
        # 90 deg FOV / 512 px, not ScanNet's calibration.
        from view_suite.ai2thor.pose_utils import intrinsics_from_fov
        return _K3(intrinsics_from_fov(512, 512, 90.0))

    @staticmethod
    def renderer(scene_id, cfg):
        from view_suite.habitat_gs.habitat_gs_unified_renderer import (
            HabitatGSUnifiedRender)
        return HabitatGSUnifiedRender(
            client_url=cfg["client_url"],
            client_origin=cfg.get("client_origin"), scene_id=scene_id)


class _Ai2ThorAdapter:
    corpus = "ai2thor"

    @staticmethod
    def manipulator(src_se3, step_t, step_r):
        from view_suite.ai2thor.view_manipulator import ViewManipulator
        from view_suite.ai2thor.pose_utils import c2w_to_unity_pose
        # Unity pose dicts in and out; no se3 accessors.
        c2w = _se3_to_c2w(src_se3)
        vm = ViewManipulator(
            init_pose=c2w_to_unity_pose(c2w),
            step_translation=step_t, step_rotation_deg=step_r,
            is_discrete=True)
        return vm

    @staticmethod
    def snapshot(vm):
        from view_suite.ai2thor.pose_utils import unity_pose_to_c2w
        c2w = np.asarray(unity_pose_to_c2w(vm.get_pose_thor()), dtype=np.float64)
        return (_c2w_to_se3(c2w), c2w)

    @staticmethod
    def renderer(scene_id, cfg):
        from view_suite.ai2thor.ai2thor_unified_renderer import AI2ThorUnifiedRender
        return AI2ThorUnifiedRender(
            client_url=cfg["client_url"],
            client_origin=cfg.get("client_origin"), scene_id=scene_id)

    @staticmethod
    def tasks(c2w_list, K, size):
        from view_suite.ai2thor.pose_utils import build_render_task
        return [build_render_task(np.asarray(E, dtype=np.float64),
                                  K=np.asarray(K, dtype=np.float64),
                                  width=int(size), height=int(size))
                for E in c2w_list]

    @staticmethod
    def intrinsics():
        from view_suite.ai2thor.pose_utils import intrinsics_from_fov
        return _K3(intrinsics_from_fov(512, 512, 90.0))


_ADAPTERS = {"scannet": _ScanNetAdapter,
             "habitat_gs": _HabitatGSAdapter,
             "ai2thor": _Ai2ThorAdapter}


def _se3_to_c2w(se3):
    """se3 (tx,ty,tz,rx,ry,rz degrees) -> 4x4 c2w."""
    from view_suite.scannet.view_manipulator import ViewManipulator
    vm = ViewManipulator(world_up_axis="Z", is_discrete=True)
    vm.set_se3(np.asarray(se3, dtype=np.float64), degrees=True)
    return np.asarray(vm.get_pose(mode="c2w"), dtype=np.float64)


def _c2w_to_se3(c2w):
    from view_suite.scannet.view_manipulator import ViewManipulator
    vm = ViewManipulator(world_up_axis="Z", is_discrete=True)
    vm.reset(np.asarray(c2w, dtype=np.float64))
    return vm.get_se3(degrees=True).tolist()


def _scene_step_map(corpus_dir: Optional[str]) -> Dict[str, Tuple[float, float]]:
    """scene_id -> (step_translation_m, step_rotation_deg), for per-scene steps."""
    out: Dict[str, Tuple[float, float]] = {}
    if not corpus_dir:
        return out
    import json as _json
    for split in ("train", "eval", "test", ""):
        name = f"interactive_view_planning_{split}.jsonl" if split else \
               "interactive_view_planning.jsonl"
        p = Path(corpus_dir) / name
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = _json.loads(line)
                m = d.get("meta") or {}
                st = m.get("step_translation_m")
                if st is None:
                    continue
                out.setdefault(d["scene_id"],
                               (float(st), float(m.get("step_rotation_deg", 30.0))))
    return out


def _default_client_url(corpus: str = "scannet") -> Optional[str]:
    """Resolve the render fleet URL from $VIEWSUITE_ROOT, mirroring the env.

    Corpus-first: habitat_gs and ai2thor each run their own service on their own
    port, and falling through to the generic client_url.txt would silently point
    one corpus' atomize pass at the other corpus' renderer -- which answers, with
    images, of the wrong scenes.
    """
    import os
    root = os.environ.get("VIEWSUITE_ROOT", ".")
    names = [f"client_url_{corpus}.txt"] if corpus != "scannet" else []
    names += ["client_url_2.txt", "client_url.txt"]
    for name in names:
        for p in (os.path.join(root, name), name):
            if os.path.isfile(p):
                with open(p) as fh:
                    for line in fh:
                        if line.strip():
                            return line.strip()
    return None


def _split_actions(obs_str: str) -> List[str]:
    return [a.strip().lower() for a in (obs_str or "").split("|") if a.strip()]


def _pose_to_se3(pose: Dict[str, float]) -> List[float]:
    return [pose["tx"], pose["ty"], pose["tz"], pose["rx"], pose["ry"], pose["rz"]]


def _se3_to_pose(se3) -> Dict[str, float]:
    return {"tx": float(se3[0]), "ty": float(se3[1]), "tz": float(se3[2]),
            "rx": float(se3[3]), "ry": float(se3[4]), "rz": float(se3[5])}


def _K3(K) -> List[List[float]]:
    K = np.asarray(K, dtype=np.float64)
    if K.shape == (4, 4):
        K = K[:3, :3]
    return K.tolist()


def _intermediate_chain(src_se3, actions, step_t, step_r, adapter):
    """Replay ``actions`` from ``src_se3``; return, for each intermediate pose
    (all but the last, which lands on the existing dst), ``(se3, c2w_4x4)``.
    Returns ``None`` if any action is not renderable (unknown code)."""
    vm = adapter.manipulator(src_se3, step_t, step_r)
    out = []
    n = len(actions)
    for i, a in enumerate(actions):
        code = _ACTION_TO_CODE.get(a)
        if code is None:
            return None
        vm.step(code)
        if i < n - 1:  # intermediate; last action lands on the existing dst node
            out.append(adapter.snapshot(vm))
    return out


async def _render_scene(scene_id, c2w_list, cfg, K, size, chunk, adapter):
    """Render a list of c2w poses for one scene. Returns list of PIL.Image|None
    aligned with ``c2w_list`` (None on failure)."""
    r = adapter.renderer(scene_id, cfg)
    results: List[Any] = [None] * len(c2w_list)
    try:
        for start in range(0, len(c2w_list), chunk):
            sub = c2w_list[start:start + chunk]
            try:
                imgs = await r.render_tasks(adapter.tasks(sub, K, size))
                for j, im in enumerate(imgs):
                    results[start + j] = im
            except Exception as e:  # whole chunk failed -> those stay None
                logger.warning("[atomize] render chunk failed scene=%s: %s", scene_id, e)
    finally:
        try:
            await r.close()
        except Exception:
            pass
    return results


def _pose_dedup(builder, graph) -> int:
    """Re-run the builder's node merge + dedup over ALL nodes (bucket_key +
    is_similar_to + unique_key). Real nodes (with image_paths) win when merging.
    Returns number of nodes merged away."""
    g = graph._g
    buckets: Dict[str, List[str]] = defaultdict(list)  # bucket_key -> [rep uid]
    key_map: Dict[str, str] = {}
    rep_attrs: Dict[str, Dict] = {}

    for nid, ndata in list(g.nodes(data=True)):
        node = builder._make_node_data(ndata)
        bk = node.bucket_key()
        rep = None
        for uid in buckets[bk]:
            if node.is_similar_to(builder._make_node_data(rep_attrs[uid])):
                rep = uid
                break
        if rep is None:
            rep = node.unique_key()
            buckets[bk].append(rep)
            rep_attrs[rep] = dict(ndata)
        else:
            # prefer attrs that carry an image
            if ndata.get("image_paths") and not rep_attrs[rep].get("image_paths"):
                rep_attrs[rep] = dict(ndata)
        key_map[nid] = rep

    new_g = g.__class__()
    for rep, attrs in rep_attrs.items():
        new_g.add_node(rep, **attrs)
    seen: set = set()
    for u, v, _eid, data in g.edges(data=True, keys=True):
        ru, rv = key_map[u], key_map[v]
        if ru == rv:
            continue
        k = (ru, rv, data["obs_str"])
        if k in seen:
            continue
        seen.add(k)
        new_g.add_edge(ru, rv, key=repr(data["obs_str"]),
                       obs_str=data["obs_str"],
                       image_paths=data.get("image_paths", []),
                       extra=data.get("extra", {}))
    merged = g.number_of_nodes() - new_g.number_of_nodes()
    graph._g = new_g
    return merged


def atomize_graph(builder, graph, images_dir, cfg: Dict[str, Any]) -> Dict[str, int]:
    """Make ``graph`` single-action-only by rendering intermediate views.

    cfg keys: corpus(scannet), client_url (required), scannet_root, client_origin,
    corpus_dir, step_translation(0.5), step_rotation(30.0), size(512),
    render_chunk(32).
    """
    g = graph._g
    corpus = str(cfg.get("corpus", "scannet"))
    if corpus not in _ADAPTERS:
        raise ValueError(f"[atomize] unknown corpus {corpus!r}; "
                         f"expected one of {sorted(_ADAPTERS)}")
    adapter = _ADAPTERS[corpus]
    step_t = float(cfg.get("step_translation", 0.5))
    step_r = float(cfg.get("step_rotation", 30.0))
    scene_steps = _scene_step_map(cfg.get("corpus_dir"))
    size = int(cfg.get("size", 512))
    chunk = int(cfg.get("render_chunk", 32))
    K = adapter.intrinsics()
    if not cfg.get("client_url"):
        cfg = dict(cfg)
        cfg["client_url"] = _default_client_url(corpus)
    if not cfg.get("client_url"):
        raise ValueError("[atomize] no render client_url (set atomize.client_url "
                         "or provide client_url_2.txt / client_url.txt in VIEWSUITE_ROOT)")
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[atomize] corpus=%s per-scene steps for %d scenes "
                "(global fallback %.2f m / %.1f deg)",
                corpus, len(scene_steps), step_t, step_r)

    # ── 1) collect multi-action edges + compute intermediate poses ──
    # jobs: list of dict(u, v, eid, scene, actions, inter_se3, req_idx[list])
    jobs: List[Dict[str, Any]] = []
    scene_reqs: Dict[str, List[np.ndarray]] = defaultdict(list)  # scene -> [c2w]
    n_multi = 0
    for u, v, eid, data in list(g.edges(data=True, keys=True)):
        actions = _split_actions(data.get("obs_str", ""))
        if len(actions) <= 1:
            continue
        n_multi += 1
        su = g.nodes[u].get("state") or {}
        pose = su.get("pose")
        scene = su.get("scene_id") or g.nodes[u].get("extra", {}).get("scene_id")
        if not pose or not scene:
            jobs.append({"u": u, "v": v, "eid": eid, "drop": True})
            continue
        s_t, s_r = scene_steps.get(scene, (step_t, step_r))
        chain = _intermediate_chain(_pose_to_se3(pose), actions, s_t, s_r, adapter)
        if chain is None:
            jobs.append({"u": u, "v": v, "eid": eid, "drop": True})
            continue
        req_idx = []
        for se3, c2w in chain:
            req_idx.append(len(scene_reqs[scene]))
            scene_reqs[scene].append(c2w)
        jobs.append({"u": u, "v": v, "eid": eid, "scene": scene,
                     "actions": actions,
                     "inter_se3": [se3 for se3, _ in chain],
                     "req_idx": req_idx, "drop": False})

    if n_multi == 0:
        logger.info("[atomize] no multi-action edges; graph already atomic")
        return {"multi_edges": 0, "rendered": 0, "dropped": 0, "merged": 0}

    # ── 2) render all intermediate poses (async, per-scene batched) ──
    async def _run_all():
        sem = asyncio.Semaphore(int(cfg.get("max_scene_concurrency", 8)))
        out: Dict[str, List[Any]] = {}

        async def _one(scene, poses):
            async with sem:
                out[scene] = await _render_scene(scene, poses, cfg, K, size, chunk,
                                                 adapter)

        await asyncio.gather(*[_one(s, p) for s, p in scene_reqs.items()])
        return out

    rendered = asyncio.run(_run_all())

    # ── 3) rebuild edges: chain on success, drop on any render failure ──
    n_drop = n_rendered = 0
    for job in jobs:
        # remove the original multi-action edge
        if g.has_edge(job["u"], job["v"], key=job["eid"]):
            g.remove_edge(job["u"], job["v"], key=job["eid"])
        if job.get("drop"):
            n_drop += 1
            continue
        scene = job["scene"]
        imgs = [rendered[scene][i] for i in job["req_idx"]]
        if any(im is None for im in imgs):
            n_drop += 1  # any intermediate render failed -> drop whole edge
            continue
        # materialize intermediate real nodes + single-action chain
        prev = job["u"]
        ok = True
        node_ids = []
        for k, (se3, im) in enumerate(zip(job["inter_se3"], imgs)):
            pose = _se3_to_pose(se3)
            raw = f"{scene}|{pose['tx']:.4f}_{pose['ty']:.4f}_{pose['tz']:.4f}_" \
                  f"{pose['rx']:.4f}_{pose['ry']:.4f}_{pose['rz']:.4f}"
            nid = f"{scene}_" + hashlib.md5(raw.encode()).hexdigest()[:12]
            img_path = images_dir / f"atom_{nid}.png"
            try:
                if not img_path.exists():
                    im.save(str(img_path))
            except Exception as e:
                logger.warning("[atomize] image save failed %s: %s", img_path, e)
                ok = False
                break
            if nid not in g:
                g.add_node(nid, state={"scene_id": scene, "pose": pose},
                           obs_str="", image_paths=[str(img_path)],
                           extra={"scene_id": scene, "virtual": False, "atomized": True})
            node_ids.append(nid)
            n_rendered += 1
        if not ok:
            n_drop += 1
            continue
        chain_nodes = [job["u"]] + node_ids + [job["v"]]
        for k, act in enumerate(job["actions"]):
            a, b = chain_nodes[k], chain_nodes[k + 1]
            if a == b:
                continue
            if not g.has_edge(a, b, key=repr(act)):
                g.add_edge(a, b, key=repr(act), obs_str=act, image_paths=[], extra={})

    # ── 4) re-run node merge + dedup so new nodes merge across trajectories ──
    n_merged = _pose_dedup(builder, graph)

    # ── 5) safety: guarantee no multi-action edges survive ──
    leftover = 0
    for u, v, eid, data in list(graph._g.edges(data=True, keys=True)):
        if len(_split_actions(data.get("obs_str", ""))) > 1:
            graph._g.remove_edge(u, v, key=eid)
            leftover += 1

    logger.info(
        "[atomize] multi_edges=%d rendered_nodes=%d dropped_edges=%d merged=%d "
        "leftover_removed=%d -> %d nodes / %d edges",
        n_multi, n_rendered, n_drop, n_merged, leftover,
        graph.num_nodes, graph.num_edges,
    )
    return {"multi_edges": n_multi, "rendered": n_rendered,
            "dropped": n_drop, "merged": n_merged, "leftover_removed": leftover}
