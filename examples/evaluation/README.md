# ViewSuite evaluation on VAGEN-SLIME

Every existing evaluation YAML and shell entry point in this directory runs
through `python -m view_suite.evaluation.run_eval`. VAGEN-SLIME supplies the
evaluator, concurrency, resume recorder and environment harness. ViewAgent
supplies its environment adapters and provider-specific HTTP translation.

Run an existing entry point as before:

```bash
bash examples/evaluation/eval_gemini/eval.sh
bash examples/evaluation/eval_all_openrouter/eval_all.sh
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
  bash examples/evaluation/eval_sglang/eval_model.sh
```

Or invoke any YAML directly:

```bash
python -m view_suite.evaluation.run_eval \
  --config examples/evaluation/eval_random/config_fwd_inv.yaml
```

All launchers accept `--validate-only`. It resolves defaults, environment
seeds, provider settings and VAGEN-SLIME harness selection without contacting
an environment/model or starting SGLang:

```bash
bash examples/evaluation/eval_random/run_navigation.sh --validate-only
bash examples/evaluation/eval_sglang/eval_model.sh --validate-only
```

Any additional `key=value` arguments are OmegaConf overrides and retain the
old script interface. `VIEWSUITE_ROOT` defaults to the repository root.
For an internal renderer using a self-signed HTTPS certificate, set
`RENDER_TLS_NO_VERIFY=1` explicitly.

Provider credentials are read from the established variables:

- OpenAI/OpenRouter: `OPENAI_API_KEY` / `OPENROUTER_API_KEY`
- Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`

The native VAGEN-SLIME result is authoritative. After a run, ViewAgent also
projects it into the historical `tag_<task>/<seed>/{messages,metrics,meta}.json`
layout so existing analysis scripts continue to work.

Dataset prerequisites depend on the chosen YAML. ScanNet, AI2-THOR and
Habitat-GS configurations use their corresponding directories below
`$VIEWSUITE_ROOT/data`. The `eval_all_openrouter_gs` configs require the
separate `data/viewagent15k_scannet_gs_test` artifact; generate or download it before
running those jobs.
