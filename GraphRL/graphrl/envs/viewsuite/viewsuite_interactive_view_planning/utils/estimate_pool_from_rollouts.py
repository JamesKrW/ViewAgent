#!/usr/bin/env python3
"""Estimate each arm's path pool from rollout dumps, without re-running traj_to_sft.

``count_paths`` needs a built graph, and a graph is only built when an iteration's RL
phase completes. That is a slow way to answer "how many paths can each arm sample",
because the answer is already latent in the rollouts: poses and actions are parsed out
of the conversation text, and only the *quality filter* needs the rendered images.

So this rebuilds the graph from ``iter_00N/rl/rollout_data/*.jsonl`` with each arm's own
builder — which is where the structural difference lives (cross-trajectory node merging
vs per-trajectory scoping) — and skips the image-quality filter.

**The result is an upper bound.** The real pipeline also:
  * drops nodes whose renders are blank or near-uniform (needs the images, which are not
    kept in the network storage mirror), and
  * runs atomize, which re-renders intermediate views and splits multi-action edges.
Both arms are affected the same way, so the *ratio* travels further than the absolute
counts — but treat both as estimates, not measurements.

    python -m graphrl.envs.viewsuite.viewsuite_interactive_view_planning.utils.\
estimate_pool_from_rollouts \
        --arm graph:<exp>/iter_003/rl/rollout_data \
        --arm nograph:<exp>/iter_003/rl/rollout_data \
        [--min-len 1 --max-len 5] [--cap 20000] [--max-files N]
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

from .count_paths import measure  # noqa: F401  (kept importable together)
from .sft_generators.helpers import _count_paths_by_length, _get_scene_node_ids

logger = logging.getLogger(__name__)

BUILDERS = {
    "graph": (
        "graphrl.envs.viewsuite.viewsuite_interactive_view_planning."
        "interactive_view_planning_graph_builder",
        "InteractiveViewPlanningGraphBuilder",
    ),
    "nograph": (
        "graphrl.envs.viewsuite.viewsuite_interactive_view_planning_nograph."
        "nograph_graph_builder",
        None,  # resolved below: first Builder subclass in the module
    ),
}


def _load_builder(kind: str):
    import importlib
    mod_name, cls_name = BUILDERS[kind]
    mod = importlib.import_module(mod_name)
    if cls_name:
        return getattr(mod, cls_name)
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.endswith("GraphBuilder"):
            return obj
    raise SystemExit(f"no *GraphBuilder class in {mod_name}")


def build_graph(kind: str, rollout_dir: Path, max_files: int | None,
                merge_pos: float, merge_ang: float, dedup: bool = True):
    cls = _load_builder(kind)
    # Same merge tolerance the runs used (run.sh overrides the 1e-3 default). This is
    # what lets grid-identical poses from different trajectories merge, so it is the
    # single most important setting for the graph arm's pool -- leaving it at the
    # config default would understate graph by construction.
    config = {
        "num_workers": 1,
        "merge_tol": {"position": merge_pos, "angle": merge_ang},
        # atomize re-renders intermediate views; off here, and reported as such.
        "atomize": {"enabled": False},
        "filter": {"void_threshold": 0.7, "std_threshold": 10.0},
    }
    builder = cls(config)

    files = sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if max_files:
        files = files[:max_files]
    with tempfile.TemporaryDirectory() as tmp:
        images_dir = Path(tmp) / "images"
        images_dir.mkdir()
        graph = builder._build_sequential(files, rollout_dir, images_dir)
    n_before = graph._g.number_of_nodes()
    merged = 0
    if dedup:
        # Insertion-time dedup is EXACT: unique_key md5s the pose at 4dp, and
        # _upsert_node is a plain key lookup, so merge_tol plays no part there and
        # float drift keeps grid-identical poses in different trajectories apart.
        # The tolerance only bites in the pose-similarity pass that atomize runs
        # (bucket_key + is_similar_to). Skipping it measures the graph *before* the
        # merge that defines the method -- which is why both arms then come out with
        # near-identical node counts. It needs no renderer, so run it here.
        from .graph_atomize import _pose_dedup
        merged = _pose_dedup(builder, graph)
    return graph, len(files), n_before, merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="kind:path_to_rollout_data, kind in {graph,nograph}")
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--merge-pos", type=float, default=0.2,
                    help="graph_builder.merge_tol.position as used by run.sh")
    ap.add_argument("--merge-ang", type=float, default=10.0,
                    help="graph_builder.merge_tol.angle as used by run.sh")
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip the pose-similarity merge (measures the raw parse)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    lens = list(range(a.min_len, a.max_len + 1))
    rows = []
    for spec in a.arm:
        kind, _, path = spec.partition(":")
        graph, n_files, n_before, merged = build_graph(
            kind, Path(path), a.max_files, a.merge_pos, a.merge_ang,
            dedup=not a.no_dedup)
        scenes = _get_scene_node_ids(graph)
        per_len = {L: 0 for L in lens}
        capped = 0
        for _sid, node_ids in scenes.items():
            counts, truncated = _count_paths_by_length(
                graph, node_ids, a.min_len, a.max_len, cap=a.cap
            )
            for L, n in counts.items():
                per_len[L] += n
            capped += bool(truncated)
        rows.append({
            "arm": kind, "rollout_files": n_files,
            "nodes": graph._g.number_of_nodes(), "edges": graph._g.number_of_edges(),
            "scenes": len(scenes), "scenes_capped": capped,
            "nodes_before_merge": n_before, "nodes_merged_away": merged,
            "paths_by_length": per_len, "paths_total": sum(per_len.values()),
        })

    w = max(len(r["arm"]) for r in rows) + 2
    print(f"{'arm':<{w}}{'files':>7}{'nodes':>8}{'edges':>8}{'scenes':>8}"
          + "".join(f"{'len=' + str(L):>10}" for L in lens) + f"{'total':>11}{'capped':>8}")
    for r in rows:
        print(f"{r['arm']:<{w}}{r['rollout_files']:>7}{r['nodes']:>8}{r['edges']:>8}"
              f"{r['scenes']:>8}"
              + "".join(f"{r['paths_by_length'][L]:>10}" for L in lens)
              + f"{r['paths_total']:>11}{r['scenes_capped']:>8}")

    if len(rows) == 2 and rows[1]["paths_total"]:
        print(f"\npool ratio ({rows[0]['arm']} / {rows[1]['arm']}) = "
              f"{rows[0]['paths_total'] / rows[1]['paths_total']:.3f}")
    print("\nUPPER BOUND: image-quality filtering and atomize are both skipped "
          "(the mirror keeps no rollout images). Compare the ratio, not the absolutes.")
    if any(r["scenes_capped"] for r in rows):
        print("Some scenes hit --cap; those counts are lower bounds, so the ratio is "
              "unreliable. Raise --cap or use a shorter --max-len.")

    if a.json:
        a.json.write_text(json.dumps(rows, indent=2))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
