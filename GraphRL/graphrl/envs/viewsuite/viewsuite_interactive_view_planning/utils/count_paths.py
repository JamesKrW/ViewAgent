#!/usr/bin/env python3
"""How many paths can each arm actually sample, per scene and per path length?

The sampler takes an absolute ``sample_per_scene`` count, so a graph-vs-no-graph
comparison at a fixed count is not a controlled one: the arm that builds the richer
graph is throttled, and the sparser arm is asked for more paths than it has. To hold
the *rate* equal instead you need the denominator, and nothing in the pipeline ever
computed it. This does.

    python -m graphrl.envs.viewsuite.viewsuite_interactive_view_planning.utils.count_paths \
        --graph-dir <exp>/iter_00N/traj_to_sft/graph [--graph-dir <other arm>] \
        [--min-len 1 --max-len 5] [--cap 20000]

Prints, per length, the number of distinct paths available — where "distinct" is
exactly what the sampler dedups on (edge-key sequence, no repeated nodes) — so the
numbers are directly comparable to what a run would draw from.

``--cap`` bounds the enumeration per scene; a dense graph has combinatorially many
paths. Capped scenes are reported separately, and their counts are lower bounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrl.traj_to_sft.utils.base_graph import BaseGraph

from .sft_generators.helpers import _count_paths_by_length, _get_scene_node_ids


def measure(graph_dir: Path, min_len: int, max_len: int, cap: int) -> dict:
    graph = BaseGraph.load(graph_dir)
    scenes = _get_scene_node_ids(graph)
    per_len = {L: 0 for L in range(min_len, max_len + 1)}
    capped, empty = [], []
    for scene_id, node_ids in scenes.items():
        counts, truncated = _count_paths_by_length(
            graph, node_ids, min_len, max_len, cap=cap
        )
        for L, n in counts.items():
            per_len[L] += n
        if truncated:
            capped.append(scene_id)
        elif not sum(counts.values()):
            empty.append(scene_id)
    return {
        "graph_dir": str(graph_dir),
        "nodes": graph._g.number_of_nodes(),
        "edges": graph._g.number_of_edges(),
        "scenes": len(scenes),
        "scenes_capped": len(capped),
        "scenes_with_no_path": len(empty),
        "paths_by_length": per_len,
        "paths_total": sum(per_len.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", action="append", required=True, type=Path,
                    help="repeat to compare arms")
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=5)
    ap.add_argument("--cap", type=int, default=20000, help="per-scene enumeration cap")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    results = [measure(d, a.min_len, a.max_len, a.cap) for d in a.graph_dir]

    lens = list(range(a.min_len, a.max_len + 1))
    name_w = max(len(Path(r["graph_dir"]).parts[-4]) for r in results) + 2
    print(f"{'arm':<{name_w}}{'nodes':>8}{'edges':>8}{'scenes':>8}"
          + "".join(f"{'len=' + str(L):>10}" for L in lens) + f"{'total':>10}{'capped':>8}")
    for r in results:
        print(f"{Path(r['graph_dir']).parts[-4]:<{name_w}}{r['nodes']:>8}{r['edges']:>8}"
              f"{r['scenes']:>8}"
              + "".join(f"{r['paths_by_length'][L]:>10}" for L in lens)
              + f"{r['paths_total']:>10}{r['scenes_capped']:>8}")

    if len(results) == 2 and results[1]["paths_total"]:
        ratio = results[0]["paths_total"] / results[1]["paths_total"]
        print(f"\npool ratio (first / second) = {ratio:.3f}")
        print("Equal-rate sampling means one sample_fraction for both arms; the volume "
              "difference should then be a consequence of the graphs, not of the config.")
    if any(r["scenes_capped"] for r in results):
        print("\nNOTE: some scenes hit --cap; their counts are lower bounds and the "
              "ratio above is not trustworthy. Raise --cap or compare at shorter lengths.")

    if a.json:
        a.json.write_text(json.dumps(results, indent=2))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
