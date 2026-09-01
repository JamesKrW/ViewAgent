## Examples

### 1. Qwen2.5-VL-7B on Hopper / H100 / H200

Default attention kernels work out of the box.

```bash
export VIEWSUITE_ROOT="$(pwd)"
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
MODEL_NAME=qwen_25_vl_7b \
DP_SIZE=2 \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

### 2. Qwen2.5-VL-7B on RTX 6000 / Blackwell (sm_120)

The sgl-kernel flash-attn doesn't ship for sm_120; switch to flashinfer + triton.

```bash
export VIEWSUITE_ROOT="$(pwd)"
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
MODEL_NAME=qwen_25_vl_7b \
SGLANG_EXTRA_ARGS="--attention-backend=flashinfer --mm-attention-backend=triton_attn" \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

### 3. Qwen3-VL-8B (HF) on full 3-task benchmark

```bash
export VIEWSUITE_ROOT="$(pwd)"
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct \
MODEL_NAME=qwen_3_vl_8b \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

### 4. Local Model, IVP

```bash
export VIEWSUITE_ROOT="$(pwd)"
MODEL_PATH=/path/to/your/checkpoint \
MODEL_NAME=my_ckpt \
CONFIG=examples/evaluation/eval_sglang/interactive_view_planning_only.yaml \
DUMP_DIR="$(pwd)/rollouts/my_ckpt" \
CUDA_VISIBLE_DEVICES=1 \
SGLANG_EXTRA_ARGS="--attention-backend=flashinfer --mm-attention-backend=triton_attn" \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

### 5. Ablation YAMLs (NoSnap / NoSubmit) on the same checkpoint

```bash
export VIEWSUITE_ROOT="$(pwd)"
COMMON=( MODEL_PATH=/path/to/your/checkpoint
         CUDA_VISIBLE_DEVICES=1
         SGLANG_EXTRA_ARGS="--attention-backend=flashinfer --mm-attention-backend=triton_attn" )

# rotation-snap disabled
env "${COMMON[@]}" \
  MODEL_NAME=my_ckpt_nosnap \
  CONFIG=examples/evaluation/eval_sglang/interactive_view_planning_only_nosnap.yaml \
  DUMP_DIR="$(pwd)/rollouts_nosnap/my_ckpt" \
  bash examples/evaluation/eval_sglang/eval_model.sh

# no-submit (auto-success on threshold)
env "${COMMON[@]}" \
  MODEL_NAME=my_ckpt_nosubmit \
  CONFIG=examples/evaluation/eval_sglang/interactive_view_planning_only_nosubmit.yaml \
  DUMP_DIR="$(pwd)/rollouts_nosubmit/my_ckpt" \
  bash examples/evaluation/eval_sglang/eval_model.sh
```


---

## Tips

- Sweep multiple configs against one checkpoint: same `MODEL_PATH` + different
  `CONFIG` + different `MODEL_NAME` / `DUMP_DIR` → each YAML's rollouts go
  to a separate dump dir, but they all reuse the same `tag_<task>/` layout
  consumed by `view_suite/analysis/proxy_analysis/easy_hard_analysis.py`.
- Multi-GPU server: set `TP_SIZE=N` (single replica spanning N GPUs) or
  `DP_SIZE=N` (N independent replicas).
- Override any YAML key on the command line — anything after the script name
  is forwarded to `view_suite.evaluation.run_eval`:

  ```bash
  ... bash examples/evaluation/eval_sglang/eval_model.sh \
        run.max_concurrent_jobs=128 \
        backends.sglang.max_concurrency=64
  ```
