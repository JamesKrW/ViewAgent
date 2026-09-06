#!/usr/bin/env bash
# =============================================================================
# Preflight the corpus and the renderer, then start GraphRL -- in that order,
# in one process, so training cannot begin against a setup that is already broken.
#
# A broken setup does not announce itself: the service answers 200 on internal
# errors, and IVP reading 0.000 looks the same whether the renderer is dead or the
# model is bad. Each stage checks one thing that is otherwise invisible until the
# metrics come back wrong.
#
#   1. data      -- splits, row counts, scene-disjoint, no do-nothing wins,
#                   referenced images on disk, n_envs matching the row count
#   2. transport -- the client_url file resolves; a wrong scheme is env_error on
#                   every episode
#   3. pixels    -- real renders, checked for variance
#   4. episodes  -- the real env class through multi-turn episodes; turns must
#                   advance past the first
#   5. atomize   -- the SFT-side render path, which uses a different renderer class
#
# Usage:
#   bash scripts/graphrl_preflight_and_run.sh habitat_gs
#   bash scripts/graphrl_preflight_and_run.sh ai2thor
#   PREFLIGHT_ONLY=1 bash scripts/graphrl_preflight_and_run.sh ai2thor   # gate only
#   bash scripts/graphrl_preflight_and_run.sh ai2thor iterations=2       # passed through
#
# Env:
#   VIEWSUITE_ROOT       repo root (required)
#   PREFLIGHT_ONLY=1     run the gate, skip training
#   PREFLIGHT_EPISODES   episodes for stage 4 (default 3)
#   RENDER_TLS_NO_VERIFY=1  accept a self-signed render cert
#   PY                   interpreter for the env smoke (default: the habitat-gs env)
# =============================================================================
set -euo pipefail

CORPUS="${1:-}"
[ -n "$CORPUS" ] || { echo "usage: $0 <habitat_gs|ai2thor> [extra run.sh args]"; exit 2; }
shift || true

# ViewSuite's data dir and validation split name differ from the other two.
case "$CORPUS" in
  habitat_gs) EXAMPLE=habitat_gs_interactive_view_planning; ENV_CLASS=HabitatGSInteractiveViewPlanning
              DATA_SUBDIR=habitat_gs; VAL_SPLIT=eval; URL_NAME=client_url_habitat_gs.txt ;;
  ai2thor)    EXAMPLE=ai2thor_interactive_view_planning;    ENV_CLASS=Ai2ThorInteractiveViewPlanning
              DATA_SUBDIR=ai2thor;    VAL_SPLIT=eval; URL_NAME=client_url_ai2thor.txt ;;
  viewsuite)  EXAMPLE=viewsuite_interactive_view_planning;  ENV_CLASS=InteractiveViewPlanning
              DATA_SUBDIR=viewagent15k_scannet_open3d; VAL_SPLIT=dev;  URL_NAME=client_url.txt ;;
  *) echo "unknown corpus: $CORPUS (expected habitat_gs, ai2thor or viewsuite)"; exit 2 ;;
esac

: "${VIEWSUITE_ROOT:?VIEWSUITE_ROOT must be exported}"
DATA_DIR="$VIEWSUITE_ROOT/data/$DATA_SUBDIR"
EXAMPLE_DIR="$VIEWSUITE_ROOT/GraphRL/examples/viewsuite/$EXAMPLE"
URL_FILE="$VIEWSUITE_ROOT/$URL_NAME"
EPISODES="${PREFLIGHT_EPISODES:-3}"
PY="${PY:-$HOME/miniconda3/envs/habitat-gs/bin/python}"

# Check the interpreter before failing inside a heredoc.
"$PY" -c 'import numpy, PIL' 2>/dev/null || {
    echo "[preflight][FATAL] PY=$PY cannot import numpy/PIL."
    echo "  Point PY at the env that owns this corpus' renderer and env classes,"
    echo "  e.g. PY=\$HOME/miniconda3/envs/habitat-gs/bin/python (habitat_gs)"
    echo "    or PY=\$HOME/miniconda3/envs/viewsuite/bin/python  (ai2thor)."
    exit 1
}

echo "=============================================================="
echo " preflight: $CORPUS"
echo "   data     $DATA_DIR"
echo "   example  $EXAMPLE_DIR"
echo "   url file $URL_FILE"
echo "=============================================================="

# ---- 1) Data ------------------------------------------------------------------
# Row counts, scene disjointness, and whether a do-nothing agent can win.
"$PY" - "$DATA_DIR" "$EXAMPLE_DIR" "$VAL_SPLIT" "$CORPUS" <<'PY' || { echo "[preflight][FATAL] data check failed"; exit 1; }
import json, os, sys
import numpy as np

data_dir, example_dir, val_split, corpus = (sys.argv[1], sys.argv[2],
                                            sys.argv[3], sys.argv[4])
TASKS = ("interactive_view_planning", "path_to_view", "view_to_path")
SPLITS = ("train", val_split, "test")
bad = []

def pose_delta(a, b):
    a, b = np.array(a), np.array(b)
    R = a[:3, :3].T @ b[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(ang)

counts, scenes = {}, {}
for task in TASKS:
    for split in SPLITS:
        p = os.path.join(data_dir, f"{task}_{split}.jsonl")
        if not os.path.isfile(p):
            bad.append(f"missing split: {task}_{split}.jsonl"); continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        counts[(task, split)] = len(rows)
        scenes.setdefault(task, {})[split] = {r["scene_id"] for r in rows}
        if not rows:
            bad.append(f"empty split: {task}_{split}.jsonl")

# A scene in both train and test makes every number meaningless.
for task, per_split in scenes.items():
    for a in SPLITS:
        for b in SPLITS:
            if a < b and per_split.get(a) and per_split.get(b):
                overlap = per_split[a] & per_split[b]
                if overlap:
                    bad.append(f"{task}: {len(overlap)} scenes in both {a} and {b}")

# The three tasks share samples; a mismatch means a partial regen.
for split in SPLITS:
    sizes = {t: counts.get((t, split)) for t in TASKS}
    if len({v for v in sizes.values() if v is not None}) > 1:
        bad.append(f"{split}: task row counts disagree {sizes}")

# Do-nothing wins, judged by each sample's own tolerance.
for split in SPLITS:
    p = os.path.join(data_dir, f"interactive_view_planning_{split}.jsonl")
    if not os.path.isfile(p):
        continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    noop = 0
    for r in rows:
        m = r.get("meta") or {}
        step_t = m.get("step_translation_m")
        tol_p = float(step_t) if step_t else 0.5
        tol_a = float(m.get("step_rotation_deg", 30.0)) if step_t else 30.0
        d = r["image_detail"]
        dp, da = pose_delta(d["init_view"]["c2w_extrinsics"], d["target_view"]["c2w_extrinsics"])
        if dp <= tol_p and da <= tol_a + 1e-6:
            noop += 1
    print(f"  {'IVP ' + split:20s} {len(rows):5d} rows   do-nothing wins: {noop}")
    if noop:
        # ViewSuite is the published corpus; its known rate is left alone.
        # The regenerated corpora must stay at zero.
        if corpus == "viewsuite":
            print(f"  [warn] {split}: {noop} samples solvable without moving "
                  f"({100.0*noop/max(1,len(rows)):.1f}%) -- known, not regenerated")
        else:
            bad.append(f"{split}: {noop} samples solvable without moving")

# resolve_rel_image returns None on a miss, which fails one frame later.
missing = 0
checked = 0
for task in TASKS:
    for split in SPLITS:
        p = os.path.join(data_dir, f"{task}_{split}.jsonl")
        if not os.path.isfile(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            for rel in json.loads(line).get("image_path", []):
                checked += 1
                if not os.path.isfile(os.path.normpath(os.path.join(data_dir, rel))):
                    missing += 1
print(f"  {'images':20s} {checked:5d} referenced, {missing} missing")
if missing:
    bad.append(f"{missing} referenced images not on disk")

# reset() does idx = seed % total_lines, so a mismatch silently re-weights or
# skips rows rather than erroring.
import re
for yml, split in (("train.yaml", "train"), ("val.yaml", val_split)):
    path = os.path.join(example_dir, yml)
    if not os.path.isfile(path):
        bad.append(f"missing {yml}"); continue
    for n in (int(x) for x in re.findall(r"n_envs:\s*(\d+)", open(path).read())):
        rows = counts.get(("interactive_view_planning", split))
        if rows is not None and n != rows:
            bad.append(f"{yml}: n_envs={n} but {split} split has {rows} rows")

if bad:
    print("\n[preflight] data problems:")
    for b in bad:
        print(f"  - {b}")
    sys.exit(1)
print("  data OK")
PY

# ---- 2) Transport --------------------------------------------------------------
[ -f "$URL_FILE" ] || { echo "[preflight][FATAL] $URL_FILE missing -- start the render service first"; exit 1; }
RENDER_URL="$(tr -d '[:space:]' < "$URL_FILE" | cut -d';' -f1)"
[ -n "$RENDER_URL" ] || { echo "[preflight][FATAL] $URL_FILE is empty"; exit 1; }
echo "  render url: $RENDER_URL"

# ---- 3) Pixels -----------------------------------------------------------------
# Ask for real images over the URL the trainer will use, and look at them. A service
# that answers 200 with a black frame is the failure mode that costs days.
PYTHONPATH="$VIEWSUITE_ROOT:${PYTHONPATH:-}" \
"$PY" - "$RENDER_URL" "$DATA_DIR" "$CORPUS" <<'PY' || { echo "[preflight][FATAL] render check failed"; exit 1; }
import io, json, os, ssl, sys, time, urllib.request, uuid, glob
import numpy as np
from PIL import Image

base, data_dir, corpus = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]

ctx = None
if os.environ.get("RENDER_TLS_NO_VERIFY", "0").strip().lower() in ("1", "true", "yes"):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

# A pose the corpus contains: a made-up camera can legitimately see nothing.
row = None
for split in ("eval", "test", "train"):
    p = os.path.join(data_dir, f"interactive_view_planning_{split}.jsonl")
    if os.path.isfile(p):
        with open(p) as f:
            row = json.loads(f.readline())
        break
if row is None:
    print("  no jsonl to source a pose from"); sys.exit(1)

scene = row["scene_id"]
view = row["image_detail"]["init_view"]
# Each service has its own task schema and neither rejects the other's.
if corpus == "ai2thor":
    from view_suite.ai2thor.pose_utils import build_render_task
    task = build_render_task(np.array(view["c2w_extrinsics"], dtype=np.float64),
                             width=512, height=512,
                             K=np.array(view["c2w_intrinsics"], dtype=np.float64))
else:
    task = {"mode": "cam_param",
            "intrinsics": view["c2w_intrinsics"],
            "extrinsics": view["c2w_extrinsics"],
            "size": [512, 512]}
n_ask = 3
meta = json.dumps({"scene_id": scene, "tasks": [task] * n_ask})
b = "----p" + uuid.uuid4().hex
body = (f"--{b}\r\nContent-Disposition: form-data; name=\"meta\"\r\n\r\n{meta}\r\n--{b}--\r\n").encode()
req = urllib.request.Request(base + "/render", data=body,
                             headers={"Content-Type": f"multipart/form-data; boundary={b}"})
t = time.time()
try:
    with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
        payload = r.read()
except Exception as e:
    print(f"  render request failed: {type(e).__name__}: {str(e)[:160]}")
    print("  (an http:// url against an https-only service fails exactly here)")
    sys.exit(1)

n = payload.count(b"image/png")
print(f"  render: {len(payload)}B, {n}/{n_ask} images, {time.time() - t:.1f}s, scene={scene}")
if n != n_ask:
    sys.exit(1)

# Variance, not just presence.
stds = []
for part in payload.split(b"\x89PNG\r\n\x1a\n")[1:]:
    try:
        img = Image.open(io.BytesIO(b"\x89PNG\r\n\x1a\n" + part)).convert("RGB")
    except Exception:
        continue
    stds.append(float(np.asarray(img, dtype=np.float32).std()))
if not stds:
    print("  could not decode any returned image"); sys.exit(1)
print(f"  pixel std: {['%.1f' % s for s in stds]}")
if max(stds) < 5.0:
    print("  every frame is near-uniform -- the renderer is answering but not drawing")
    sys.exit(1)
print("  pixels OK")
PY

# ---- 4) Episodes ---------------------------------------------------------------
# Turns must advance past the first.
PYTHONPATH="$VIEWSUITE_ROOT:${PYTHONPATH:-}" \
"$PY" - "$DATA_DIR" "$URL_FILE" "$ENV_CLASS" "$EPISODES" "$VAL_SPLIT" <<'PY' || { echo "[preflight][FATAL] env smoke failed"; exit 1; }
import asyncio, importlib, os, sys

data_dir, url_file, env_class, n_ep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

MODULES = {
    "HabitatGSInteractiveViewPlanning":
        "view_suite.envs.habitat_gs_proxy_task.interactive_view_planning",
    "Ai2ThorInteractiveViewPlanning":
        "view_suite.envs.ai2thor_proxy_task.interactive_view_planning",
    "InteractiveViewPlanning":
        "view_suite.envs.scannet_proxy_task.interactive_view_planning",
}
cls = getattr(importlib.import_module(MODULES[env_class]), env_class)

cfg = {
    "jsonl_path": os.path.join(data_dir, f"interactive_view_planning_{sys.argv[5]}.jsonl"),
    "image_size": [512, 512],
    "format": "eval_mode",
    "use_example_in_sys_prompt": False,
    "client_url_file_path": url_file,
    "render_backend": "client",
    "max_turns": 10,
    "action_only_mode": True,
    "allow_rotate": False,
}

async def one(seed):
    env = cls(dict(cfg))
    obs, info = await env.reset(seed)
    if not obs:
        return seed, 0, "reset returned no observation"
    turns = 0
    # Enough to prove the render loop survives past turn 1.
    for action in ("<answer>move_forward</answer>",
                   "<answer>turn_right</answer>",
                   "<answer>move_forward</answer>"):
        try:
            obs, reward, done, info = await env.step(action)
        except Exception as e:
            return seed, turns, f"{type(e).__name__}: {str(e)[:120]}"
        turns += 1
        if info.get("env_error"):
            return seed, turns, f"env_error: {str(info['env_error'])[:120]}"
        if done:
            break
    return seed, turns, None

async def main():
    results = await asyncio.gather(*(one(s) for s in range(n_ep)))
    worst = 0
    for seed, turns, err in results:
        status = "ok" if err is None else f"FAIL {err}"
        print(f"  episode seed={seed}: {turns} turns  {status}")
        worst = max(worst, turns)
    if any(err for _, _, err in results):
        return 1
    if worst < 2:
        print("  no episode got past its first turn -- this is the num_turns/max == 1 failure")
        return 1
    print(f"  episodes OK (max turns reached: {worst})")
    return 0

sys.exit(asyncio.run(main()))
PY

# ---- 5) Atomize render path ----------------------------------------------------
# Separate from stage 3: that one posts hand-built multipart, this one goes through
# the corpus' UnifiedRender class. Agreeing with the service is not agreeing with
# each other.
PYTHONPATH="$VIEWSUITE_ROOT:$VIEWSUITE_ROOT/GraphRL:${PYTHONPATH:-}" \
"$PY" - "$CORPUS" "$DATA_DIR" "$URL_FILE" <<'PYATOM' \
  || { echo "[preflight][FATAL] atomize render path failed"; exit 1; }
import asyncio, importlib.util, json, sys
import numpy as np

corpus, data_dir, url_file = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location(
    "ga", "GraphRL/graphrl/envs/viewsuite/viewsuite_interactive_view_planning/"
          "utils/graph_atomize.py")
GA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GA)

adapter = GA._ADAPTERS["scannet" if corpus == "viewsuite" else corpus]
with open(f"{data_dir}/interactive_view_planning_test.jsonl") as f:
    row = json.loads(f.readline())
c2w = np.array(row["image_detail"]["init_view"]["c2w_extrinsics"], dtype=np.float64)
cfg = {"client_url": open(url_file).read().strip()}
try:
    imgs = asyncio.run(GA._render_scene(row["scene_id"], [c2w, c2w], cfg,
                                        adapter.intrinsics(), 512, 32, adapter))
except Exception as e:
    print(f"  atomize renderer raised: {type(e).__name__}: {str(e)[:160]}")
    sys.exit(1)
ok = [i for i in imgs if i is not None]
stds = [float(np.asarray(i.convert("RGB"), dtype=np.float32).std()) for i in ok]
print(f"  atomize render: {len(ok)}/2 images, std={['%.1f' % v for v in stds]}")
if len(ok) != 2 or max(stds, default=0.0) < 5.0:
    sys.exit(1)
print("  atomize path OK")
PYATOM

echo
echo "=============================================================="
echo " preflight passed: $CORPUS"
echo "=============================================================="

if [ -n "${PREFLIGHT_ONLY:-}" ]; then
    echo "PREFLIGHT_ONLY set -- not starting training."
    exit 0
fi

# ---- 5) Train ------------------------------------------------------------------
echo "starting GraphRL: $EXAMPLE_DIR/run.sh"
exec bash "$EXAMPLE_DIR/run.sh" "$@"
