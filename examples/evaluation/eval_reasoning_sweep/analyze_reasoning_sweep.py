#!/usr/bin/env python3
"""
Reasoning-effort sweep analysis: Short/Long/All success per task per model per
reasoning mode (nothink / medium / high).

"Short" vs "Long" follows view_suite/analysis/proxy_analysis/easy_hard_analysis.py:
unified view distance d = sqrt((d_pos/0.5)^2 + (d_rot/30)^2); d < 3 => Short
(easy), d >= 3 => Long (hard). IVP success uses the 0.5 m / 30 deg pose
threshold; P2V/V2P use exact-choice success.

The "medium" mode is NOT re-run — it is the published eval_all_openrouter
baseline (eval_mode prompt + reasoning.effort=medium). Its per-task Short/Long/
All numbers are embedded below (from the paper table). nothink/high are read
from rollouts/reasoning_sweep/<mode>/<model>/tag_*/metrics.json.

CAVEAT: medium is on the full N=530 test set; nothink/high are on a fixed,
deterministic 150-sample subset (identical across nothink & high, so those two
are perfectly controlled). Medium is a population-rate reference, not a
same-sample comparison.

Usage:
  python analyze_reasoning_sweep.py \
    --rollouts_root /path/to/rollouts/reasoning_sweep \
    --data_dir /path/to/data/viewsuite_15k
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# ── Config ──────────────────────────────────────────────────────────────────

EASY_THRESHOLD = 3.0
STEP_TRANSLATION_M = 0.5
STEP_ROTATION_DEG = 30.0
AE_POS_THRESHOLD = 0.5
AE_ANG_THRESHOLD = 30.0

TASKS = ["tag_path_to_view", "tag_view_to_path", "tag_interactive_view_planning"]
TASK_DISPLAY = {
    "tag_path_to_view": "P2V",
    "tag_view_to_path": "V2P",
    "tag_interactive_view_planning": "IVP",
}
TASK_JSONL = {
    "tag_path_to_view": "path_to_view_test_filter.jsonl",
    "tag_view_to_path": "view_to_path_test_filter.jsonl",
    "tag_interactive_view_planning": "interactive_view_planning_test_filter.jsonl",
}

MODELS = ["gpt_5_4", "gemini_3_1_pro", "claude_opus_4_6", "grok_4_20_beta"]
MODEL_DISPLAY = {
    "gpt_5_4": "GPT-5.4",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "claude_opus_4_6": "Claude Opus 4.6",
    "grok_4_20_beta": "Grok 4.20 Beta",
}
MODES = ["nothink", "medium", "high"]

# Published medium (eval_mode + effort=medium) numbers from the paper table,
# N=530. Format: {model: {tag: {"easy": %, "hard": %, "all": %}}}
MEDIUM_PAPER: Dict[str, Dict[str, Dict[str, float]]] = {
    "gpt_5_4": {
        "tag_path_to_view": {"easy": 57.3, "hard": 42.9, "all": 47.9},
        "tag_view_to_path": {"easy": 60.5, "hard": 37.4, "all": 45.5},
        "tag_interactive_view_planning": {"easy": 33.5, "hard": 7.5, "all": 16.6},
    },
    "gemini_3_1_pro": {
        "tag_path_to_view": {"easy": 63.8, "hard": 40.9, "all": 48.9},
        "tag_view_to_path": {"easy": 53.0, "hard": 47.5, "all": 49.4},
        "tag_interactive_view_planning": {"easy": 28.6, "hard": 17.4, "all": 21.3},
    },
    "claude_opus_4_6": {
        "tag_path_to_view": {"easy": 46.5, "hard": 28.4, "all": 34.7},
        "tag_view_to_path": {"easy": 47.6, "hard": 38.3, "all": 41.5},
        "tag_interactive_view_planning": {"easy": 23.8, "hard": 3.8, "all": 10.8},
    },
    "grok_4_20_beta": {
        "tag_path_to_view": {"easy": 61.6, "hard": 38.0, "all": 46.2},
        "tag_view_to_path": {"easy": 44.9, "hard": 44.3, "all": 44.5},
        "tag_interactive_view_planning": {"easy": 17.3, "hard": 2.9, "all": 7.9},
    },
}


# ── Pose / distance utilities (from easy_hard_analysis.py) ────────────────────

def c2w_to_se3(c2w: list) -> np.ndarray:
    M = np.array(c2w, dtype=np.float64)
    eul = R.from_matrix(M[:3, :3]).as_euler("xyz", degrees=True)
    return np.concatenate([M[:3, 3], eul])


def geodesic_angle_deg(euler_a, euler_b) -> float:
    Ra = R.from_euler("xyz", np.asarray(euler_a, float), degrees=True).as_matrix()
    Rb = R.from_euler("xyz", np.asarray(euler_b, float), degrees=True).as_matrix()
    cos_theta = float(np.clip((np.trace(Ra @ Rb.T) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def unified_view_distance(init_c2w, target_c2w) -> float:
    a, b = c2w_to_se3(init_c2w), c2w_to_se3(target_c2w)
    d_pos = float(np.linalg.norm(a[:3] - b[:3]))
    d_rot = geodesic_angle_deg(a[3:], b[3:])
    return float(np.sqrt((d_pos / STEP_TRANSLATION_M) ** 2 + (d_rot / STEP_ROTATION_DEG) ** 2))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_view_distances(jsonl_path: Path) -> Dict[str, float]:
    dist: Dict[str, float] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item.get("sample_id", "")
            det = item.get("image_detail", {})
            iv = det.get("init_view", {}).get("c2w_extrinsics")
            tv = det.get("target_view", {}).get("c2w_extrinsics")
            if sid and iv is not None and tv is not None:
                try:
                    dist[sid] = unified_view_distance(iv, tv)
                except Exception:
                    pass
    return dist


def is_success(m: dict, task: str) -> bool:
    if task == "tag_interactive_view_planning":
        pos, ang = m.get("pos_err_m"), m.get("ang_err_deg")
        if pos is None or ang is None:
            infos = m.get("infos", [])
            if infos and isinstance(infos[-1], dict):
                pos = infos[-1].get("pos_err_m")
                ang = infos[-1].get("ang_err_deg")
        if pos is not None and ang is not None:
            return pos <= AE_POS_THRESHOLD + 1e-9 and ang <= AE_ANG_THRESHOLD + 1e-9
        return bool(m.get("success", False))
    return bool(m.get("success", False))


def read_metrics_by_sid(task_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not task_dir.is_dir():
        return out
    for rd in task_dir.iterdir():
        mp = rd / "metrics.json"
        if not (rd.is_dir() and mp.exists()):
            continue
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        sid = m.get("sample_id", "")
        if not sid:
            for info in m.get("infos", []):
                if isinstance(info, dict) and info.get("sample_id"):
                    sid = str(info["sample_id"])
                    break
        if sid:
            out[sid] = m
    return out


def rate_from_rollouts(task_dir: Path, distances: Dict[str, float]) -> Dict[str, float]:
    et = es = ht = hs = 0
    for sid, m in read_metrics_by_sid(task_dir).items():
        if sid not in distances:
            continue
        ok = is_success(m, task_dir.name)
        if distances[sid] < EASY_THRESHOLD:
            et += 1; es += int(ok)
        else:
            ht += 1; hs += int(ok)
    at = et + ht
    return {
        "easy": es / et * 100 if et else float("nan"),
        "hard": hs / ht * 100 if ht else float("nan"),
        "all": (es + hs) / at * 100 if at else float("nan"),
        "easy_n": et, "hard_n": ht, "n": at,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def fmt(x) -> str:
    return "  -  " if x != x else f"{x:.1f}"  # NaN check


def run(rollouts_root: str, data_dir: str, out_dir: Optional[str] = None):
    root = Path(rollouts_root)
    data = Path(data_dir)
    out = Path(out_dir) if out_dir else root / "_analysis"
    out.mkdir(parents=True, exist_ok=True)

    distances = {t: load_view_distances(data / TASK_JSONL[t]) for t in TASKS}
    for t in TASKS:
        d = distances[t]
        ne = sum(v < EASY_THRESHOLD for v in d.values())
        print(f"{TASK_DISPLAY[t]}: {len(d)} samples ({ne} short / {len(d)-ne} long)")

    # results[model][mode][task] = {"easy","hard","all",...}
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    for model in MODELS:
        for mode in ("nothink", "high"):
            for t in TASKS:
                td = root / mode / model / t
                if td.is_dir():
                    results[model][mode][t] = rate_from_rollouts(td, distances[t])
        # medium from paper
        if model in MEDIUM_PAPER:
            for t in TASKS:
                results[model]["medium"][t] = dict(MEDIUM_PAPER[model][t])

    # ── Build table ──
    header = "| Model | Mode | " + " | ".join(
        f"{TASK_DISPLAY[t]} S | {TASK_DISPLAY[t]} L | {TASK_DISPLAY[t]} All" for t in TASKS
    ) + " | Overall | (N/task) |"
    sep = "|" + "|".join(["---"] * (2 + 3 * len(TASKS) + 2)) + "|"
    lines = [header, sep]

    def overall(model, mode):
        vals = [results[model][mode][t]["all"] for t in TASKS if t in results[model][mode]]
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    for model in MODELS:
        for mode in MODES:
            if mode not in results[model]:
                continue
            row = f"| {MODEL_DISPLAY[model]} | {mode} "
            ns = []
            for t in TASKS:
                r = results[model][mode].get(t)
                if r is None:
                    row += "| - | - | - "
                else:
                    row += f"| {fmt(r['easy'])} | {fmt(r['hard'])} | **{fmt(r['all'])}** "
                    if "n" in r:
                        ns.append(r["n"])
            ncol = str(max(ns)) if ns else "paper(530)"
            row += f"| **{fmt(overall(model, mode))}** | {ncol} |"
            lines.append(row)
        lines.append(sep)

    table = "\n".join(lines)
    print("\n" + table + "\n")

    md = out / "reasoning_sweep_table.md"
    with open(md, "w") as f:
        f.write("# Reasoning-effort sweep: Short/Long/All success\n\n")
        f.write(f"Short: d < {EASY_THRESHOLD}, Long: d >= {EASY_THRESHOLD} "
                f"(unified view distance).\n\n")
        f.write("Modes: **nothink** (no-think prompt + reasoning off; Gemini 3.1 Pro "
                "uses effort=low, its minimum), **medium** (paper baseline, N=530, "
                "eval_mode + effort=medium), **high** (eval_mode + effort=high).\n\n")
        f.write("nothink & high are run on this setup (see the N/task column); "
                "medium is the published baseline (N=530). When nothink/high N=530 "
                "they cover the same full test set as medium.\n\n")
        f.write(table + "\n")

    # CSV
    csv = out / "reasoning_sweep_table.csv"
    with open(csv, "w") as f:
        cols = ["Model", "Mode"]
        for t in TASKS:
            cols += [f"{TASK_DISPLAY[t]}_S", f"{TASK_DISPLAY[t]}_L", f"{TASK_DISPLAY[t]}_All"]
        cols += ["Overall", "N"]
        f.write(",".join(cols) + "\n")
        for model in MODELS:
            for mode in MODES:
                if mode not in results[model]:
                    continue
                cells = [MODEL_DISPLAY[model], mode]
                ns = []
                for t in TASKS:
                    r = results[model][mode].get(t, {})
                    cells += [fmt(r.get("easy", float('nan'))), fmt(r.get("hard", float('nan'))), fmt(r.get("all", float('nan')))]
                    if "n" in r:
                        ns.append(r["n"])
                cells += [fmt(overall(model, mode)), str(max(ns)) if ns else "530"]
                f.write(",".join(cells) + "\n")

    json.dump(
        {MODEL_DISPLAY[m]: {mode: results[m].get(mode, {}) for mode in MODES} for m in MODELS},
        open(out / "reasoning_sweep_results.json", "w"), indent=2,
    )
    print(f"Wrote {md}\n      {csv}\n      {out/'reasoning_sweep_results.json'}")


if __name__ == "__main__":
    import fire
    fire.Fire(run)
