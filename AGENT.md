# InternVL3.5-8B on ViewSuite / GraphRL — branch `feat/internvl35-8b`

Adds training + evaluation of **`OpenGVLab/InternVL3_5-8B-HF`** to the ViewSuite GraphRL
pipeline (RL self-exploration + view-graph SFT for interactive view planning), alongside
the existing Qwen2.5-VL-7B / Qwen3-VL-8B recipes.

Runs on `devgpu004` (8×H200). conda env: **`viewsuite`** (Python 3.12).

---

## TL;DR — key decisions

- **Use the HF-native model `OpenGVLab/InternVL3_5-8B-HF`** (arch `InternVLForConditionalGeneration`,
  qwen3 LLM backbone), *not* the custom-code `OpenGVLab/InternVL3_5-8B`. LLaMA-Factory (SFT) and
  transformers/verl require the `-HF` conversion.
- **Serve/rollout with vLLM, not sglang.** sglang 0.5.3.post3 has no native
  `InternVLForConditionalGeneration` impl (falls back to the transformers backend → garbage output).
  vLLM 0.11.0 supports it natively. So **eval uses the vLLM backend** and **RL rollout uses
  `rollout.name=vllm`**. (This also matches verl PR #6578, which is vLLM-based.)
- The render service (`client_url.txt`, external RTX-4090 boxes) is only reachable from this Meta
  box **via `with-proxy`**. So run eval / training under `with-proxy`. `meta.wandb.io` and
  `127.0.0.1` are kept off the proxy (`no_proxy`).

---

## Environment

```bash
conda activate viewsuite      # torch 2.8+cu128, transformers 4.57.1, vllm 0.11.0,
                              # sglang 0.5.3.post3, verl 0.6.1, flash-attn 2.8.1, numpy 2.2.6
```

Env-setup fixes already applied (see git history / _setup/ scripts):
- Restored `GraphRL/LLaMA-Factory/src/llamafactory/data/` (dropped during vendoring; needed for SFT;
  contains the `intern_vl` template + `InternVLPlugin`).
- Pinned **numpy 2.2.6** (numba/vLLM need ≤2.2; sglang had pulled 2.5).
- `CUDA_HOME=/usr/local/cuda-12.8` set in the env `activate.d` so flashinfer/sgl JIT finds `nvcc`.
- Symlinked `data/viewsuite_15k/*_{train,dev,test}_filter.jsonl → *_{...}.jsonl` (configs reference
  the `_filter` names; the release ships non-filter).

---

## Evaluate the base model (P2V / V2P / IVP)

vLLM backend. IVP needs the render service → run under `with-proxy`.

```bash
export VIEWSUITE_ROOT=$(pwd)
# 1) serve (single GPU or DP across 8):
bash _setup/serve_vllm_internvl.sh            # single GPU, port 30011
# or 8-GPU data-parallel for speed:
bash _setup/serve_vllm_internvl_dp8.sh
# 2) run the 3-task eval against it:
with-proxy python -m vagen.evaluate.run_eval \
  --config examples/evaluation/eval_sglang/internvl35_8b_full.yaml fileroot=$(pwd)
```

**Base results (InternVL3.5-8B-HF, temp 0, 530 ep each):** P2V 8.87% · V2P 20.75% · IVP 0.57%.
Rollouts/summaries under `rollouts/internvl35_8b/tag_*/summary.json`.

---

## Train (GraphRL: RL → view-graph SFT, iterated)

```bash
export VIEWSUITE_ROOT=$(pwd)
cd GraphRL
# full run (4 iters, mirrors the qwen3vl8b recipe):
with-proxy bash examples/viewsuite/viewsuite_interactive_view_planning/run_internvl35_8b.sh
# small smoke (3 iters x 10 RL steps, tiny batch, 1 SFT epoch):
with-proxy bash examples/viewsuite/viewsuite_interactive_view_planning/run_internvl35_8b_smoke.sh
```

Config: `pipeline_internvl35_8b.yaml` (mirrors `pipeline_qwen3vl8b.yaml`):
`initial_model_path=OpenGVLab/InternVL3_5-8B-HF`, SFT `template=intern_vl`, `rollout.name=vllm`.

W&B: meta W&B creds live in repo `.env` (`WANDB_BASE_URL=https://meta.wandb.io`, entity `kangrui`).
The run scripts source `.env` and add `meta.wandb.io` to `no_proxy`; if `WANDB_API_KEY` is unset
they fall back to `WANDB_MODE=offline`.

---

## verl changes needed for InternVL (in `GraphRL/VAGEN/verl`)

1. `verl/utils/model.py` — `load_valuehead_model` now recognizes `AutoModelForImageTextToText`
   (InternVL critic; it's not an `AutoModelForCausalLM`).
2. `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   - `json.dumps` dict/list engine_kwargs when building the `vllm serve` CLI (e.g.
     `limit_mm_per_prompt` — Python-repr single quotes broke vLLM's `json.loads`).
   - **`_qwen2_5_vl_dedup_image_tokens` generalized to InternVL** — collapses the agent's
     pre-expanded `<IMG_CONTEXT>` runs to one-per-image so vLLM re-expands them. Without this the
     pre-expanded tokens conflict with vLLM's expansion → CUDA `masked_scatter_size_check`
     (`totalElements <= srcSize`) crash. **This was the main RL blocker.**

Config-side (in `pipeline_internvl35_8b.yaml`): `rollout.name=vllm`, `use_fused_kernels=false`
(the fused `forward_with_torch_backend` requires `return_dict`, unsupported for the value-head
forward), `enforce_eager=true`, `engine_kwargs.vllm.limit_mm_per_prompt={image:16,video:0}`
(video disabled — vLLM interns1 has a 448/384 video-profiling shape bug),
`enable_prefix_caching=false`, `disable_mm_preprocessor_cache=true`.

---

## Status

- ✅ Env, data, base eval (all 3 tasks).
- ✅ GraphRL RL loop for InternVL3.5-8B: rollout (multimodal, via render service) → reward → GAE →
  actor/critic FSDP update all run cleanly (verified step 1: reward mean ≈ 0.20, sane grads).
- ⏳ Full smoke (RL → traj_to_sft → SFT → next iter) validating end to end.

Helper scripts + logs: `/home/kangrui/projects/viewagent/_setup/`. verl PR #6578 diff:
`_setup/verl_pr6578.diff`.
