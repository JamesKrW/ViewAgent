# Standalone Sokoban evaluation

This example evaluates any OpenAI-compatible vision model against VAGEN's existing
`Sokoban` environment with the existing `concat`, `no_concat`, and `compact` harnesses.
It does not start Megatron or a training actor.

## Quickstart

From the VAGEN-SLIME repository root:

```bash
bash examples/eval/sokoban/run_local.sh sglang 4b
```

Use `vllm` instead of `sglang` when `VLLM_PYTHON` points to an environment containing
vLLM. Use `9b` instead of `4b` for Qwen3.5-9B. Quickstart evaluates seed 10001
once through each harness; append `full` to run all four fixed seeds:

```bash
bash examples/eval/sokoban/run_local.sh sglang 4b full
```

The local launchers enable Qwen3.5 thinking and start the server with
`--reasoning-parser qwen3`; SGLang additionally uses `--enable-strict-thinking`, which is
required for its token-budget grammar. The evaluator executes only the final `content`;
it records `reasoning_content` separately. This prevents draft actions inside native
reasoning from being mistaken for executable actions.

The action ceiling is 8192 tokens. A measured 2048-token run of Qwen3.5-4B ended with
`finish_reason=length` and empty final content, so it tested a thinking cutoff rather
than Sokoban ability. Both examples therefore leave room for a final answer while
enforcing a 4096-token thinking budget through each server's OAI extension
(`custom_params.thinking_budget` for SGLang, `thinking_token_budget` for vLLM).
The YAML exposes this uniformly as `thinking_token_budget` plus
`thinking_budget_field`; the backend performs the mapping. `thinking_answer_reserve`
keeps 512 tokens available for final content, and automatically reduces the effective
thinking budget for shorter calls such as compact summaries.
Per-episode results and the aggregate summary expose `length_limited_calls` and
`empty_final_content_calls` so an unsupported or misconfigured extension is visible.
The compact harness uses a 24000-token conversation budget and a 1024-token summary cap;
its budget includes thinking tokens reported by the API, not only final-answer text.

## Composition model

The evaluator keeps VAGEN's existing boundaries:

```text
OpenAI-compatible model backend
  -> concat | no_concat | compact | custom BaseHarness
  -> BaseNoConcatEnv | BaseConcatEnv | BaseCompactEnv
  -> Sokoban
```

`models:` selects one or more OAI-compatible endpoints. An endpoint may be SGLang,
vLLM, or a hosted API. `envs:` selects the environment, seed set, harness, turn budget,
and environment config. Repeating an environment with another harness is sufficient to
form a harness comparison.

For partial or distributed launches, the evaluator accepts repeatable `--model`,
`--tag`, and `--seed` filters. Every slice writes to the same stable run layout and a
later unfiltered invocation fills only the missing episodes.

Both example YAMLs use one episode at a time. Increase
`run.max_concurrent_episodes` and each model's `max_concurrency` together for parallel
evaluation. Every episode owns its own environment instance.

## Resume and trajectories

`resume: skip_completed` checkpoints at episode boundaries. A restarted command skips
every matching `(model, environment tag, seed, config hash)` with an atomic completed
`result.json`. An interrupted in-flight episode is reset and replayed from its seed; a
stateful environment is never resumed from an unsafe half-step.

Outputs are under `runs/eval/<experiment-id>/` and include:

- an evaluation manifest and aggregate summary;
- one stable directory per model, harness tag, and seed;
- `result.json`, `trajectory.jsonl`, and `transcript.txt`;
- content-addressed screenshots;
- separate reasoning and final response text, token usage, latency, parsed/native
  actions, rewards, termination state, and complete environment info.

View them with:

```bash
PYTHONPATH=. python -m vagen_agent.evaluation.viewer --root runs/eval --port 8910
```

## Hosted or externally managed endpoint

The same runner supports open-weight or closed hosted models. For example:

```bash
VAGEN_EXPERIMENT_ID=sokoban-hosted-v1 \
VAGEN_MODEL_RUN_NAME=my-model \
VAGEN_SERVED_MODEL=provider-model-id \
VAGEN_MODEL_BASE_URL=https://provider.example/v1 \
VAGEN_MODEL_API_KEY=... \
PYTHONPATH=. python -m vagen_agent.evaluation \
  --config examples/eval/sokoban/oai_api.yaml
```

API secrets are redacted from the persisted evaluation manifest. Provider-specific
OpenAI request fields may be placed under `models[].sampling`. For newer OpenAI APIs,
set `token_limit_field: max_completion_tokens`. If a provider exposes a token-level
reasoning budget, point `thinking_budget_field` at its request field; otherwise omit the
thinking budget and use a standard field such as `reasoning_effort` under `sampling`.
