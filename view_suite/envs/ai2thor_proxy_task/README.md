# AI2-THOR proxy task

ViewSuite's three view-reasoning tasks on AI2-THOR instead of ScanNet. Same task
definitions, same metric, a different world: ScanNet is a fixed set of scanned meshes,
AI2-THOR is a live simulator, so the data is generated rather than collected and the
scene split can be made disjoint by construction.

| task | short | what the agent is given → asked for |
|---|---|---|
| Path-to-View | P2V | initial view + an action sequence → pick the resulting view (MCQ) |
| View-to-Path | V2P | initial view + target view + top-down reference → pick the action sequence (MCQ) |
| Interactive View Planning | IVP | a goal and a live camera → *act*, over up to 10 turns, to bring the target into view |

P2V and V2P are single-turn and need no simulator at eval time. **IVP is the multi-turn
one and renders every turn**, so it needs the render service below; it is also the task
the view graph is meant to help with.

## Layout

```
view_suite/envs/ai2thor_proxy_task/
    path_to_view.py view_to_path.py interactive_view_planning.py   the three envs
    gym_proxy_tool.py                                              multi-turn engine
    data_gen/                                                      generation + filtering
view_suite/ai2thor/
    service_http/          HTTP render service (one process per scene, GPU-pinned)
    gym_ai2thor_render_env.py  view_manipulator.py  pose_utils.py  scene_list.py
```

## Data

Generated from the simulator, not downloaded from a corpus:

```bash
export VIEWSUITE_ROOT=$(pwd)

# 120 iTHOR scenes x 24 samples, 8 GPUs, resumable
python -m view_suite.envs.ai2thor_proxy_task.data_gen.gen_parallel \
    --out_root=$VIEWSUITE_ROOT/data/viewagent15k_ai2thor_full --scenes=all --samples_per_scene=24 --n_gpus=8

# drop low-semantic views (blank walls and floors) with a VLM judge.
# Needs OPENROUTER_API; set EGRESS_PROXY if outbound traffic requires a proxy.
python -m view_suite.envs.ai2thor_proxy_task.data_gen.filter_low_semantic \
    --data_root=$VIEWSUITE_ROOT/data/viewagent15k_ai2thor_full --backend=openrouter \
    --model=qwen/qwen3.7-plus --workers=24 --review_dir=$VIEWSUITE_ROOT/filter_review

# scene-disjoint train/eval/test split
python -c "from view_suite.envs.utils.split_jsonl_by_scene import split_jsonl_by_scene as s; \
[s(f'data/viewagent15k_ai2thor/{t}.jsonl', ratios=(70,15,15)) for t in \
 ['path_to_view','view_to_path','interactive_view_planning']]"
```

The filter removes ~26% of samples. The released split is 120 scenes → 2,880 samples per
task → **train 1,468 / eval 325 / test 331**, over 84/18/18 disjoint scenes.
`bash scripts/download_ai2thor.sh` fetches the prepared copy instead of regenerating.

## Render service (IVP only)

```bash
bash scripts/ai2thor_http_service_loop.sh          # on a machine with a graphics stack
echo "http://<host>:<port>" > client_url_ai2thor.txt
```

Semicolon-separate several URLs to spread scenes across servers. IVP rollout throughput
is gated by how many render servers you give it, so this is the knob that decides how
long training takes.

## Evaluation

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
CONFIG=$VIEWSUITE_ROOT/examples/evaluation/eval_default_ai2thor.yaml \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

`eval_random_ai2thor.yaml` runs a random-navigation IVP baseline with no model attached —
a quick check that the service and plumbing work before spending a real eval on them.

## Training

```bash
cd /path/to/ViewAgent-slime/GraphRL
export VIEWSUITE_ROOT=$(cd .. && pwd)
bash examples/viewsuite/ai2thor_interactive_view_planning/run_smoke.sh   # RL-only smoke
bash examples/viewsuite/ai2thor_interactive_view_planning/run.sh         # 4 iterations
```

Each iteration is RL (IVP rollouts rendered live) → traj_to_sft (build the view graph
from those rollouts, distil supervision) → SFT → the next iteration's starting model.

## Results

Numbers are not kept in the repo. Regenerate with
`scripts/build_ai2thor_results_table.py` (base + trained + frontier) or
`examples/evaluation/eval_all_openrouter_ai2thor/build_ai2thor_table_full.py`.

The turn split the table uses is short (<=2 turns) versus long (>2).
