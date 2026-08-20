"""Break an eval run down by sample property: what does the model actually get right?

An aggregate success rate says a model is bad at a task. It does not say whether it is
bad everywhere or bad at one identifiable slice, and those call for different responses
-- more training versus a broken or degenerate slice of the corpus.

Joins each episode's `sample_id` back to its JSONL row and reports success against the
things the generator varied: ground-truth path length, which actions the path contains,
scene family (this corpus is half outdoor), and the per-scene translation step.

    ~/miniconda3/envs/habitat-gs/bin/python -m \
      view_suite.envs.habitat_gs_proxy_task.data_gen.analyse_eval \
      --rollout_dir=rollouts/default_habitat_gs --data_root=data/habitat_gs --split=test
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import fire

TASKS = {"path_to_view": "P2V", "view_to_path": "V2P",
         "interactive_view_planning": "IVP"}
LOOK = {"look_up", "look_down"}
TURN = {"turn_left", "turn_right"}


def _rows(data_root: str, task: str, split: str) -> Dict[str, dict]:
    p = os.path.join(data_root, f"{task}_{split}.jsonl")
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["sample_id"]] = d
    return out


def _episodes(rollout_dir: str, task: str) -> List[dict]:
    p = os.path.join(rollout_dir, f"tag_{task}", "summary.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p))["episodes"]


def _sample_id(ep: dict) -> Optional[str]:
    for t in ep.get("per_turn") or []:
        sid = (t.get("info") or {}).get("sample_id")
        if sid:
            return sid
    return None


def _bucket(v: float, edges) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return f"{lo:g}-{hi:g}"
    return f">={edges[-1]:g}"


def _report(name: str, groups: Dict[str, List[bool]], min_n: int = 5) -> None:
    print(f"  {name}")
    for k in sorted(groups, key=lambda x: (len(x), x)):
        v = groups[k]
        if len(v) < min_n:
            continue
        print(f"    {k:>14s}  n={len(v):>4d}  {100.0 * sum(v) / len(v):5.1f}%")


def run(rollout_dir: str, data_root: str = "data/habitat_gs", split: str = "test") -> None:
    for task, short in TASKS.items():
        eps = _episodes(rollout_dir, task)
        if not eps:
            print(f"\n=== {short}: no rollouts ===")
            continue
        rows = _rows(data_root, task, split)
        n_ok = sum(1 for e in eps if e["success"])
        print(f"\n=== {short}  {n_ok}/{len(eps)} = {100.0 * n_ok / len(eps):.1f}% ===")

        by_len, by_fam, by_step, by_look, by_turn, by_nuniq = (defaultdict(list) for _ in range(6))
        unmatched = 0
        for e in eps:
            sid = _sample_id(e)
            row = rows.get(sid)
            if row is None:
                unmatched += 1
                continue
            ok = bool(e["success"])
            m = row["meta"]
            names = m["gt_action_seq_names"]
            by_len[f"len={len(names)}"].append(ok)
            by_fam["interior" if row["scene_id"].startswith("interior_") else "sceneNN"].append(ok)
            by_step[_bucket(float(m["step_translation_m"]), [0, 0.5, 1, 2, 4, 8])].append(ok)
            n_look = sum(1 for a in names if a in LOOK)
            by_look[f"look={min(n_look, 3)}" + ("+" if n_look >= 3 else "")].append(ok)
            n_turn = sum(1 for a in names if a in TURN)
            by_turn[f"turn={min(n_turn, 3)}" + ("+" if n_turn >= 3 else "")].append(ok)
            by_nuniq[f"uniq={len(set(names))}"].append(ok)

        if unmatched:
            # Not cosmetic: a silent mismatch would quietly shrink every slice below.
            print(f"  [warn] {unmatched} episodes had no matching JSONL row")
        _report("by ground-truth path length", by_len)
        _report("by scene family", by_fam)
        _report("by translation step (m)", by_step)
        _report("by look actions in the path", by_look)
        _report("by turn actions in the path", by_turn)
        _report("by distinct actions in the path", by_nuniq)


if __name__ == "__main__":
    fire.Fire(run)
