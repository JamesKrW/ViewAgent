"""Shared utility functions for SFT generators."""

import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar

from graphrl.traj_to_sft.utils.base_graph import BaseGraph

from .constants import _ACTION_VOCAB, _ACTION_DISP

logger = logging.getLogger(__name__)


# ── universal helpers (used by all generators) ────────────────────────────

def _pick_prompt(pool: List[str], rng: random.Random) -> str:
    return rng.choice(pool)


def _clean_action(raw: str) -> str:
    """Strip trailing pipe characters from a raw obs_str action."""
    return raw.strip().rstrip("|").strip()


def _concat_actions(path: List[Dict[str, Any]]) -> str:
    """Join per-step action strings with ' | '."""
    return " | ".join(_clean_action(step["action"]) for step in path)


def _copy_image(
    src: Path,
    output_dir: Path,
    dataset_name: str,
    filename: str,
    copied: Set[str],
) -> Optional[str]:
    """Copy *src* to output_dir/dataset_name/filename.  Returns relative path."""
    if not src.exists():
        return None
    rel = f"{dataset_name}/{filename}"
    if filename not in copied:
        dst_dir = output_dir / dataset_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / filename
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.add(filename)
    return rel


def _resolve_node_image(
    graph: BaseGraph,
    node_id: str,
    images_dir: Path,
    output_dir: Path,
    dataset_name: str,
    copied: Set[str],
) -> Optional[str]:
    """Copy a node's first image to output dir.  Returns relative path or None."""
    image_rels = graph.nodes[node_id].get("image_paths", [])
    if not image_rels:
        return None
    # rel_path is like "images/abc123_0.jpg", relative to graph_dir
    filename = Path(image_rels[0]).name
    src = images_dir / filename  # images_dir = graph_dir/images
    return _copy_image(src, output_dir, dataset_name, filename, copied)


def _resolve_top_down(
    scene_id: str,
    viewsuite_15k_dir: Optional[str],
    output_dir: Path,
    dataset_name: str,
    copied: Set[str],
) -> Optional[str]:
    """Copy top-down view and return relative path, or None."""
    if not viewsuite_15k_dir:
        return None
    src = Path(viewsuite_15k_dir) / scene_id / "top_down_view.png"
    filename = f"{scene_id}_top_down.png"
    return _copy_image(src, output_dir, dataset_name, filename, copied)


def _get_scene_id(graph: BaseGraph, node_id: str) -> str:
    return graph.nodes[node_id].get("extra", {}).get("scene_id", "unknown")


# ── multi-turn helpers ────────────────────────────────────────────────────

def _format_pose(text: str, label: str = "Current") -> str:
    """Build the camera-pose block from node text (which IS the pose string)."""
    return f"{label} camera 6-DoF (c2w, Euler XYZ, DEGREES):\n{text}"


def _answer_pose(graph: BaseGraph, node_id: str) -> str:
    """Build answer(tx, ty, tz, rx, ry, rz) from node state."""
    pose = graph.nodes[node_id].get("state", {}).get("pose", {})
    if pose:
        return (
            f"answer({pose['tx']:.1f}, {pose['ty']:.1f}, {pose['tz']:.1f}, "
            f"{pose['rx']:.1f}, {pose['ry']:.1f}, {pose['rz']:.1f})"
        )
    return "answer"


# ── MCQ helpers ───────────────────────────────────────────────────────────

def _parse_individual_actions(action_str: str) -> List[str]:
    """Split a concatenated action string (pipe-separated) into individual actions."""
    return [a.strip() for a in action_str.split("|") if a.strip()]


def _net_displacement(actions: List[str]) -> tuple:
    """Compute approximate 6-D net displacement for equivalence checking."""
    disp = [0, 0, 0, 0, 0, 0]
    for a in actions:
        d = _ACTION_DISP.get(a)
        if d:
            for i in range(6):
                disp[i] += d[i]
    return tuple(disp)


def _generate_negative_action_seqs(
    gt_actions: List[str],
    n: int,
    rng: random.Random,
    max_attempts: int = 200,
) -> List[List[str]]:
    """Generate *n* negative action sequences of the same length as *gt_actions*.

    Each negative is guaranteed to have a different net displacement from the
    ground truth and from every other negative, so they cannot lead to the
    same destination.
    """
    length = len(gt_actions)
    gt_disp = _net_displacement(gt_actions)

    negatives: List[List[str]] = []
    seen_disps = {gt_disp}

    for _ in range(max_attempts):
        if len(negatives) >= n:
            break
        candidate = [rng.choice(_ACTION_VOCAB) for _ in range(length)]
        cand_disp = _net_displacement(candidate)
        if cand_disp not in seen_disps:
            seen_disps.add(cand_disp)
            negatives.append(candidate)

    return negatives[:n]


# ── specialized helpers ───────────────────────────────────────────────────

OVERSAMPLE_FACTOR = 5
DEFAULT_MAX_ACTION_RATIO = 0.15

T = TypeVar("T")


def _path_action_counter(path: List[Dict[str, Any]]) -> Counter:
    """Count individual actions across all edges of a path."""
    counts: Counter = Counter()
    for step in path:
        raw = step["action"].strip().rstrip("|").strip()
        for a in raw.split("|"):
            a = a.strip()
            if a:
                counts[a] += 1
    return counts


def _select_balanced_paths(
    candidates: List[Tuple[Counter, T]],
    num_samples: int,
    max_action_ratio: float,
    rng: random.Random,
) -> List[T]:
    """Select paths with balanced action distribution.

    Algorithm:
      1. Compute global action frequency across all candidates.
      2. Score each candidate by "rarity" — paths containing globally
         under-represented actions score higher.
      3. Greedily pick candidates in rarity order.  Skip any candidate
         that would push a single action above *max_action_ratio* of the
         running total.
      4. If the quota is not met, fill from remaining candidates
         (ignoring the cap).

    Args:
        candidates: list of (action_counter, payload) tuples.
        num_samples: target number of samples.
        max_action_ratio: max fraction any single action can occupy
            (e.g. 0.15 → no action exceeds 15 % of all actions).
        rng: random number generator.

    Returns:
        Selected payloads with balanced action distribution.
    """
    if not candidates:
        return []

    # Global action frequencies for inverse-frequency scoring
    global_counts: Counter = Counter()
    for counter, _ in candidates:
        global_counts += counter
    total_global = max(sum(global_counts.values()), 1)

    # Score each candidate: rarer actions → higher score
    indexed: List[Tuple[float, int, Counter, T]] = []
    for i, (counter, payload) in enumerate(candidates):
        rarity = sum(
            cnt * (total_global / max(global_counts[act], 1))
            for act, cnt in counter.items()
        )
        indexed.append((rarity, i, counter, payload))

    # Shuffle first for random tiebreak, then sort by rarity descending
    rng.shuffle(indexed)
    indexed.sort(key=lambda x: x[0], reverse=True)

    # Greedy selection with cap constraint
    selected: List[T] = []
    used: Set[int] = set()
    action_totals: Counter = Counter()
    total_actions = 0

    for rarity, i, counter, payload in indexed:
        if len(selected) >= num_samples:
            break
        path_total = sum(counter.values())
        new_total = total_actions + path_total
        ok = True
        for act, cnt in counter.items():
            if (action_totals[act] + cnt) / new_total > max_action_ratio:
                ok = False
                break
        if ok:
            selected.append(payload)
            used.add(i)
            action_totals += counter
            total_actions = new_total

    # Fill remainder if needed (relax constraint)
    if len(selected) < num_samples:
        for _, i, _, payload in indexed:
            if i in used:
                continue
            if len(selected) >= num_samples:
                break
            selected.append(payload)

    selected = selected[:num_samples]
    rng.shuffle(selected)
    return selected


def _generate_negative_numbers(
    gt: int,
    n: int,
    rng: random.Random,
    lo: int = 1,
    hi: int = 10,
) -> List[int]:
    """Generate *n* distinct negative numbers != *gt* in [lo, hi]."""
    candidates = [x for x in range(lo, hi + 1) if x != gt]
    rng.shuffle(candidates)
    return candidates[:n]


# ── per-scene sampling helpers ──────────────────────────────────────────

def _parse_sample_per_scene(val: Any) -> int:
    """Parse sample_per_scene from config.

    Accepts:
        15     → 15
        "15"   → 15
    """
    return int(val)


def _get_scene_node_ids(graph: BaseGraph) -> Dict[str, List[str]]:
    """Group node IDs by scene_id.

    Returns:
        dict mapping scene_id → list of node_ids belonging to that scene.
    """
    scene_nodes: Dict[str, List[str]] = defaultdict(list)
    for nid in graph._g.nodes():
        sid = graph._g.nodes[nid].get("extra", {}).get("scene_id", "unknown")
        scene_nodes[sid].append(nid)
    return dict(scene_nodes)


def _step_record(graph: BaseGraph, u: str, v: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """One path step, in the shape every generator expects."""
    return {
        "from_id": u,
        "from_state": graph._g.nodes[u]["obs_str"],
        "action": data["obs_str"],
        "to_id": v,
        "to_state": graph._g.nodes[v]["obs_str"],
    }


def _walk_paths_for_scene(
    graph: BaseGraph,
    scene_node_ids: List[str],
    min_len: int,
    max_len: int,
    cap: Optional[int] = None,
) -> Tuple[List[List[Dict[str, Any]]], Dict[int, int], bool]:
    """Enumerate every distinct path of length ``min_len``..``max_len``.

    Exhaustive counterpart of the random walk below, and deliberately identical in
    what counts as a path: nodes may not repeat, and two paths are distinct when
    their *edge-key* sequences differ (the graph is a multigraph, so parallel edges
    between the same pair are different paths).

    Returns ``(paths, count_by_length, truncated)``. Enumeration stops once ``cap``
    paths exist, in which case ``truncated`` is True and the counts are lower bounds
    — a dense graph has combinatorially many paths and counting them all is neither
    possible nor useful.
    """
    counts: Dict[int, int] = {L: 0 for L in range(min_len, max_len + 1)}
    out: List[List[Dict[str, Any]]] = []
    if not scene_node_ids or graph._g.number_of_edges() == 0:
        return out, counts, False

    def walk(cur: str, visited: Set[str], steps: List[Dict[str, Any]]) -> bool:
        """Depth-first extend. Returns False once the cap is hit."""
        depth = len(steps)
        if depth >= min_len:
            counts[depth] += 1
            out.append(list(steps))
            if cap is not None and len(out) >= cap:
                return False
        if depth >= max_len:
            return True
        for u, v, _eid, data in graph._g.out_edges(cur, data=True, keys=True):
            if v in visited:
                continue
            steps.append(_step_record(graph, u, v, data))
            visited.add(v)
            keep_going = walk(v, visited, steps)
            visited.discard(v)
            steps.pop()
            if not keep_going:
                return False
        return True

    for start in scene_node_ids:
        if not walk(start, {start}, []):
            return out, counts, True
    return out, counts, False


def _count_paths_by_length(
    graph: BaseGraph,
    scene_node_ids: List[str],
    min_len: int,
    max_len: int,
    cap: Optional[int] = None,
) -> Tuple[Dict[int, int], bool]:
    """How many paths of each length this scene can yield.

    Counts without materialising the paths — the same traversal as
    _walk_paths_for_scene, but only the tally is kept. Building step records for
    every path is what makes the exhaustive version expensive at longer lengths,
    where a single scene can hold millions of them.

    ``cap`` bounds the number counted per scene; ``truncated`` says whether it bit,
    in which case the counts are lower bounds.
    """
    counts: Dict[int, int] = {L: 0 for L in range(min_len, max_len + 1)}
    if not scene_node_ids or graph._g.number_of_edges() == 0:
        return counts, False

    g = graph._g
    total = 0
    truncated = False

    def walk(cur: str, visited: Set[str], depth: int) -> bool:
        nonlocal total, truncated
        if depth >= min_len:
            counts[depth] += 1
            total += 1
            if cap is not None and total >= cap:
                truncated = True
                return False
        if depth >= max_len:
            return True
        for _u, v, _eid in g.out_edges(cur, keys=True):
            if v in visited:
                continue
            visited.add(v)
            keep_going = walk(v, visited, depth + 1)
            visited.discard(v)
            if not keep_going:
                return False
        return True

    for start in scene_node_ids:
        if not walk(start, {start}, 0):
            break
    return counts, truncated


def _sample_paths_for_scene(
    graph: BaseGraph,
    scene_node_ids: List[str],
    min_len: int,
    max_len: int,
    num_samples: int,
    rng: random.Random,
    *,
    sample_fraction: Optional[float] = None,
    attempts_multiplier: int = 30,
    pool_cap: Optional[int] = None,
    telemetry: Optional[List[Dict[str, Any]]] = None,
    scene_id: str = "",
) -> List[List[Dict[str, Any]]]:
    """Sample paths starting from nodes of a specific scene.

    Start nodes are restricted to *scene_node_ids*; walks may cross into other
    scenes (the graph is shared).

    Two knobs beyond the original absolute ``num_samples``:

    ``sample_fraction``
        Take this fraction of the paths the scene can actually produce, instead of a
        fixed count. Comparing two methods at a fixed count is not a fair control
        when one of them builds a much richer graph: the richer method is throttled
        and the sparser one is asked for more than it has. A fraction holds the
        *rate* equal and lets the volume follow from the graph, which is the thing
        being compared.

    ``attempts_multiplier``
        Retry budget per requested path. Previously the budget was
        ``num_samples * 30``, so asking for fewer paths also allowed fewer attempts
        to find them — on a sparse graph, where unique paths are hardest to hit,
        the arm that needed more attempts got fewer. It is now independent of how
        the target was derived, and the exhaustive branch below removes the need for
        it entirely in the sparse case.

    When the pool is no larger than the target we enumerate and return everything:
    random-walking for paths you could just list is what made the old sampler
    undershoot its own budget on early, sparse graphs.
    """
    if not scene_node_ids or graph._g.number_of_edges() == 0:
        return []

    target = int(num_samples)
    pool_counts: Optional[Dict[int, int]] = None
    truncated = False

    if sample_fraction is not None or telemetry is not None:
        cap = pool_cap if pool_cap is not None else max(4 * max(target, 1), 5000)
        enumerated, pool_counts, truncated = _walk_paths_for_scene(
            graph, scene_node_ids, min_len, max_len, cap
        )
        pool_total = sum(pool_counts.values())
        if sample_fraction is not None:
            target = max(1, int(round(float(sample_fraction) * pool_total)))
        if telemetry is not None:
            telemetry.append({
                "scene_id": scene_id,
                "min_len": min_len,
                "max_len": max_len,
                "pool_by_length": dict(pool_counts),
                "pool_total": pool_total,
                "pool_truncated": truncated,
                "target": target,
            })
        # Small pool: return it outright rather than trying to rediscover it by
        # random walk (and failing, which is what the attempt cap used to cause).
        if not truncated and pool_total <= target:
            rng.shuffle(enumerated)
            return enumerated

    seen: Set[Tuple] = set()
    paths: List[List[Dict[str, Any]]] = []
    max_attempts = target * attempts_multiplier
    attempts = 0

    while len(paths) < target and attempts < max_attempts:
        attempts += 1
        cur = rng.choice(scene_node_ids)
        target_len = rng.randint(min_len, max_len)
        steps: List[Dict[str, Any]] = []
        ekey_seq: List[str] = []
        visited: Set[str] = {cur}
        ok = True

        for _ in range(target_len):
            out = list(graph._g.out_edges(cur, data=True, keys=True))
            if not out:
                ok = False
                break
            unvisited_out = [e for e in out if e[1] not in visited]
            chosen_pool = unvisited_out if unvisited_out else None
            if chosen_pool is None:
                break
            u, v, eid, data = rng.choice(chosen_pool)
            steps.append({
                "from_id": u,
                "from_state": graph._g.nodes[u]["obs_str"],
                "action": data["obs_str"],
                "to_id": v,
                "to_state": graph._g.nodes[v]["obs_str"],
            })
            ekey_seq.append(eid)
            visited.add(v)
            cur = v

        if not ok or len(steps) < min_len:
            continue

        path_key = tuple(ekey_seq)
        if path_key in seen:
            continue
        seen.add(path_key)
        paths.append(steps)

    return paths
