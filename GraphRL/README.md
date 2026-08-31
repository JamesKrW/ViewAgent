# GraphRL

> **SLIME migration branch.** `GraphRL/VAGEN` is now a vendored snapshot of
> VAGEN-SLIME. The validated Habitat-GS path is synchronous and starts through
> `examples/viewsuite/habitat_gs_interactive_view_planning/run_slime.sh`.
> Environment behavior remains under `view_suite`; the vendored backend contains
> only generic build/runtime and Qwen2.5-VL checkpoint compatibility changes.

Build the local backend environment with:

```bash
bash VAGEN/scripts/build_slime_env.sh
```

Run the two-episode integration check with:

```bash
WANDB_MODE=disabled \
  bash examples/viewsuite/habitat_gs_interactive_view_planning/run_slime.sh \
  --rollout-only \
  --eval-yaml examples/viewsuite/habitat_gs_interactive_view_planning/slime/verify.yaml
```

The legacy `graphrl.main`/VAGEN-VERL instructions below describe the original
backend and are retained as migration reference. They are not the entry point
for the SLIME branch.

Iterative `RL → TrajToSFT → SFT` pipeline for MLLM closed-loop optimisation.

Mono-backend by design — RL is always **VAGEN/verl**, SFT is always
**LLaMA-Factory**. The single user-extension point is the **TrajToSFT** phase:
write a class that subclasses `TrajToSFTModule` (or its graph-flavoured variant
`TrajToSFTGraphBase`) and point pipeline.yaml at its dotted path.

## Installation

```bash
git clone --recurse-submodules https://github.com/JamesKrW/GraphRL.git
cd GraphRL
pip install -e .

cd VAGEN && pip install -e .
cd verl && USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh \
        && pip install --no-deps -e . && pip install "trl==0.26.2"
cd ../..

cd LLaMA-Factory && pip install -e . \
        && pip install -r requirements/metrics.txt \
        && pip install -r requirements/deepspeed.txt
cd ..
```

## Quickstart

```bash
bash examples/sokoban/sokoban_text/run.sh
```

---

## Architecture

```
                ┌─────────────────────────────────────┐
                │           graphrl/main.py            │
                │   (Hydra entry + GraphRLController)  │
                └────┬───────────┬────────────┬───────┘
                     │           │            │
            ┌────────▼─┐   ┌─────▼──────┐  ┌──▼──────┐
            │ VagenWrapper │ TrajToSFT  │  │ LFWrapper│
            │  (VAGEN/RL)  │ Module(*)  │  │(LLaMA-F) │
            └──────────────┘ └──────────┘  └──────────┘
            (*) user subclass — only extension point
```

### Per-iteration directory shape (uniform — every iter has both models)

```
iter_XXX/
├── rl/                      # ★ everything VAGEN produces
│   ├── rl_model/            # promoted RL HF model     (or symlink if RL skipped)
│   ├── rollout_data/        # VAGEN JSONL outputs (TrajToSFT input)
│   ├── verl_checkpoints/
│   └── rl_training.log
├── traj_to_sft/             # ★ everything TrajToSFT produces
│   ├── graph/               #   graph-based generators' artefact
│   ├── sft_data/            #   ← LLaMA-Factory reads this
│   ├── sft_data_old/        #   reasoning's pre-augment snapshot
│   └── reasoning_dump/      #   reasoning rollout dumps
├── sft/                     # ★ everything LLaMA-Factory produces
│   ├── sft_model/           # SFT HF model              (or symlink ← rl_model if SFT skipped)
│   └── sft_training.log
└── .pending_delete          # [transient] deferred deletes (resume-safe)
```

Design rule: phase-X artefacts live under ``iter_XXX/<phase>/``. The three
phase dirs (``rl``, ``traj_to_sft``, ``sft``) are the only top-level entries.

### Phase skipping

Set any phase config to `null` in `iteration_overrides.iterN`:

```yaml
iteration_overrides:
  iter0: { rl: null }           # SFT-only iter; iter_0/rl_model must be pre-placed
  iter1: { sft: null }          # RL-only iter; sft_model is auto-symlinked from rl_model
```

When a phase is skipped the controller adds a `MaterializePhase` that
symlinks the correct upstream into the missing slot, so the unified shape
holds and the next iter starts cleanly. **iter_0 + skip-RL** requires either
a local `initial_model_path` or a pre-placed `iter_0/rl_model/`.

### Resume

`detect_progress` walks `iter_XXX/` directories backwards looking for
the most recent completed phase (sft_model > sft_data > rl_model). Pipeline
restarts from the next phase using the model that ran most recently.

VAGEN's verl writes its own `verl_checkpoints/global_step_<N>/` and resumes
mid-iteration automatically when re-launched. The framework's
`CheckpointMonitor` daemon promotes the final HF model to `iter_N/rl_model/`
once `training_steps` is hit.

---

## Per-iteration HF upload + cleanup

Three top-level keys (also overridable in `iteration_overrides.iterN`).
**Defaults are sane** — set to `[]` (or `null`) to disable per-scope.

```yaml
upload_to_hf:               [rl_model, sft_model]                       # default
delete_on_sft_model:        [rl_model, rollout_data, rl, traj_to_sft]   # default
delete_on_next_rl_model:    [sft_model, sft, sft_data]                  # default
```

**Two file-system triggers**, both checked at the end of each phase
(not just iter end — the trigger fires the moment the relevant model
becomes real, so iter_(N-1)'s artefacts are dropped *before* iter_N
starts populating its own scratch):

| Trigger fires when | List defaults |
|---|---|
| same iter's `sft_model` is **real** (not a symlink, has `config.json` + weights) | drop SFT inputs: `rl_model`, `rollout_data`, the whole `rl/` and `traj_to_sft/` dirs |
| **next** iter's `rl_model` is real (next iter has produced its own resume anchor) | drop the rest: `sft_model`, the whole `sft/` dir |

Items not yet safe are written to `iter_XXX/.pending_delete` and retried at
every subsequent iter's end. Symlink-aliased model dirs (created by
`MaterializePhase` when a phase is skipped) **never count as real** —
upstream models stay safe.

### `upload_to_hf`

After phase completion the listed resources go to HuggingFace:
- Directories that look like HF models (`config.json` + safetensors/bin) →
  `api.upload_folder` to a **model** repo (default repo id =
  `<project_name>` from pipeline.yaml).
- Other directories → packed into `.tar.gz` (pigz when available) and uploaded
  to a **dataset** repo (same repo id, `repo_type=dataset`).

`HF_TOKEN` env var is read automatically; missing → warn and skip.
`HF_HUB_ENABLE_HF_TRANSFER=1` is set for fast multipart uploads.

### Resource aliases

Short names → relative paths under `iter_XXX/`:

| alias | resolves to |
|---|---|
| `rl` | `rl/` (whole RL phase dir) |
| `rl_model` | `rl/rl_model/` |
| `rollout_data` | `rl/rollout_data/` |
| `verl_checkpoints` | `rl/verl_checkpoints/` |
| `traj_to_sft` | `traj_to_sft/` (whole TrajToSFT phase dir) |
| `sft_data` | `traj_to_sft/sft_data/` |
| `graph` | `traj_to_sft/graph/` |
| `random_sft_stage` | `traj_to_sft/random_sft_stage/` |
| `sft` | `sft/` (whole SFT phase dir) |
| `sft_model` | `sft/sft_model/` |

Aliases are used by **cleanup** (the delete lists). For
**`upload_to_hf`** the resolver is bypassed — write the literal path
you want uploaded (`rl/rl_model`, `sft/sft_model`, `rl/rollout_data`,
…). The uploader sanitises slashes when constructing tarball
filenames, so any nested path works as a name.

Anything not listed is treated as a literal relative path.

---

## Customising for a New Environment

The single extension point is **TrajToSFT**. There are two base classes:

| Subclass | When to use |
|---|---|
| `TrajToSFTModule` | Anything goes — read `paths.rollout_data`, write `paths.sft_data/` |
| `TrajToSFTGraphBase(TrajToSFTModule)` | Standard graph-build → sample-paths flow (most envs) |

### Minimal example (graph-based)

```python
# graphrl/envs/my_env/traj_to_sft.py
from typing import Dict, List, Optional, Tuple, Type

from graphrl import TrajToSFTGraphBase
from graphrl.traj_to_sft.utils.base_graph import BaseGraph
from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder

# Your env-specific graph builder (subclass of VagenGraphBuilder).
from graphrl.envs.my_env.graph_builder import MyEnvGraphBuilder


class MyEnvTrajToSFT(TrajToSFTGraphBase):
    name = "TrajToSFT(my_env)"

    def graph_builder_class(self) -> Type[VagenGraphBuilder]:
        return MyEnvGraphBuilder

    def generate_datasets(
        self, graph: BaseGraph, images_dir,
    ) -> Dict[str, Tuple[List[Dict], Optional[Dict]]]:
        records = build_records(graph, self.config)
        return {"my_dataset": (records, None)}    # None → Alpaca; dict → ShareGPT/etc.
```

`TrajToSFTGraphBase.run()` handles graph build (or cache reuse), graph load,
and writing `dataset_info.json`. Subclass just supplies which builder to use
and how to turn a graph into LLaMA-Factory records.

For non-graph envs (e.g. random eval, fixed dataset, filter-only) extend
`TrajToSFTModule` directly and override `run()` — see
`graphrl/envs/viewsuite/viewsuite_random_sft_rl/` and
`graphrl/envs/viewsuite/viewsuite_sft_rl/`.

### pipeline.yaml

```yaml
project_name: my_project
experiment_name: my_experiment
initial_model_path: Qwen/Qwen2.5-VL-7B-Instruct
iterations: 3

general_overrides:
  rl:
    training_steps: 600
    vagen_dir: VAGEN
    hydra_overrides:
      data: { train_files: ..., val_files: ... }
      # …all VAGEN/verl knobs go here

  traj_to_sft:
    module: graphrl.envs.my_env.MyEnvTrajToSFT     # ← dotted path
    # …subclass-specific config (your generator reads self.config)

  sft:
    n_gpus: 8
    hydra_overrides:
      stage: sft
      template: qwen2_vl
      # …all LLaMA-Factory knobs

iteration_overrides:
  iter0: { rl: { training_steps: 61 } }
  iter1: { rl: { training_steps: 61 } }
  iter2:
    rl: { training_steps: 1000, timeout: 259200 }   # 72h for the long iter
```

### Custom env class for VAGEN

If your env needs a Python class registered with VAGEN, add it to
`graphrl/configs/vagen_configs/env_registry.yaml`:

```yaml
env_registry:
  MyEnv: graphrl.envs.my_env.env.MyEnv
```

VAGEN reads this on its own (via Hydra searchpath), no other wiring needed.

---

## Repository Layout

```
GraphRL/
├── graphrl/
│   ├── main.py                       # Hydra entry + GraphRLController
│   ├── state.py                      # ModuleState, ModuleOutput
│   ├── vagen/
│   │   ├── vagen_wrapper.py          # class VagenWrapper  (the only RL phase impl)
│   │   └── utils/command_builder.py
│   ├── llama_factory/
│   │   ├── lf_wrapper.py             # class LFWrapper     (the only SFT phase impl)
│   │   └── utils/config_generator.py
│   ├── traj_to_sft/
│   │   ├── traj_to_sft_base.py       # TrajToSFTModule + TrajToSFTPaths
│   │   ├── traj_to_sft_graph_base.py # TrajToSFTGraphBase
│   │   └── utils/
│   │       ├── base_graph.py         # BaseGraph + NodeData/EdgeData
│   │       └── graph_builder.py      # VagenGraphBuilder (NetworkX-backed)
│   ├── envs/                         # reference envs (grouped by project)
│   │   ├── sokoban/
│   │   │   └── sokoban_text/                 # ← graph-based starting point (text)
│   │   └── viewsuite/
│   │       ├── viewsuite_interactive_view_planning/     # ← multimodal graph-based env
│   │       ├── viewsuite_interactive_view_planning_selfboot/
│   │       ├── viewsuite_random_sft_rl/      # ← non-graph TrajToSFT (runs eval subprocess)
│   │       └── viewsuite_sft_rl/             # ← non-graph TrajToSFT (fixed dataset symlink)
│   ├── configs/                      # framework defaults
│   │   ├── vagen_configs/            # VAGEN/verl Hydra defaults
│   │   ├── llamafactory_configs/     # LLaMA-Factory Hydra defaults
│   │   └── pipeline.yaml
│   └── utils/
│       ├── checkpoint_monitor.py
│       ├── config.py                 # load_pipeline_config + merge_iteration_config
│       ├── hf_uploader.py
│       ├── iter_cleanup.py           # double-trigger resume-safe cleanup
│       ├── logging.py
│       ├── periodic_task.py
│       ├── process.py
│       └── progress.py               # detect_progress (resume)
└── examples/                         # runnable pipelines (grouped by project)
    ├── sokoban/
    │   └── sokoban_text/
    └── viewsuite/
        ├── viewsuite_interactive_view_planning/
        ├── viewsuite_interactive_view_planning_selfboot/
        ├── viewsuite_random_sft_rl/
        └── viewsuite_sft_rl/
```

For env-author guides see
[`graphrl/envs/sokoban/sokoban_text/README.md`](graphrl/envs/sokoban/sokoban_text/README.md)
(graph-based) and
[`graphrl/envs/viewsuite/viewsuite_interactive_view_planning/README.md`](graphrl/envs/viewsuite/viewsuite_interactive_view_planning/README.md)
(multimodal graph + dedup customisation).
