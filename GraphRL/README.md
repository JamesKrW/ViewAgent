# GraphRL on VAGEN-SLIME

This checkout uses **VAGEN-SLIME** for both the environment harness and RL
training. `GraphRL/VAGEN-SLIME` is the vendored VAGEN-SLIME source tree; ViewAgent
integration code stays outside it.

The ownership boundary is:

- `GraphRL/VAGEN-SLIME/`: generic VAGEN-SLIME evaluator, harness and SLIME trainer.
- `view_suite/envs/slime_adapter.py`: adapters for the nine ViewSuite task/env
  combinations.
- `view_suite/evaluation/`: compatibility loader and provider backends for the
  existing evaluation YAML files.
- `GraphRL/graphrl/slime/`: GraphRL launch, rollout export, metrics,
  checkpointing and adaptive-control integration.
- `GraphRL/graphrl/main.py`: fixed `RL -> TrajToSFT -> SFT` controller.
- `GraphRL/graphrl/main_adaptive.py`: separate metric-driven controller.

No shipped script invokes the removed `vagen.main_ppo`/verl path.

## Environment setup

Build the VAGEN-SLIME environment once:

```bash
cd /path/to/ViewAgent-slime/GraphRL
bash VAGEN/scripts/build_slime_env.sh
```

The example scripts source `examples/_slime_env.sh`. It selects the adjacent
`../conda_envs/slime/bin/python` when available and exports all required source
roots to `PYTHONPATH`.

For ViewSuite runs, datasets and renderer URL files are resolved below
`VIEWSUITE_ROOT`, which defaults to the ViewAgent checkout root. A different
asset root can be supplied without editing a script:

```bash
VIEWSUITE_ROOT=/path/to/assets bash examples/viewsuite/.../run.sh
```

## Entry points

All scripts under `GraphRL/examples` now launch VAGEN-SLIME:

```bash
cd /path/to/ViewAgent-slime/GraphRL

# Fixed schedules
bash examples/sokoban/sokoban_text/run.sh
bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh
bash examples/viewsuite/ai2thor_interactive_view_planning/run.sh
bash examples/viewsuite/habitat_gs_interactive_view_planning/run.sh

# Metric-driven schedules
bash examples/viewsuite/viewsuite_interactive_view_planning/run_adaptive.sh
bash examples/viewsuite/ai2thor_interactive_view_planning/run_adaptive.sh
bash examples/viewsuite/habitat_gs_interactive_view_planning/run_adaptive.sh
```

The baseline, smoke, Qwen3-VL, self-boot and random-SFT scripts in the same
tree use the same backend. Hydra overrides remain supported, for example:

```bash
bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh \
  general_overrides.rl.slime.eval_interval=10 \
  general_overrides.rl.slime.num_gpus=8
```

To validate an entry point without allocating GPUs or starting training:

```bash
bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh --cfg job
```

## Fixed and adaptive control flows

`graphrl.main` preserves the fixed experiment structure. Each configured round
runs its explicit `training_steps`, then TrajToSFT and SFT unless a phase is
disabled for that round. Existing experiment directories and phase completion
markers continue to be recognized.

`graphrl.main_adaptive` is an independent control flow selected only by the
`run_adaptive.sh` scripts. It has no per-round RL step cap. Its important
parameters are:

```yaml
adaptive_schedule:
  metric: val-aux/ae/traj_success/mean@1
  direction: maximize
  eval_every_steps: 20
  min_delta: 0.03
  min_relative_delta: 0.10
  sft_patience: 3
  latch_threshold: 0.1
  finish_patience: 8
  total_rl_steps: 801
```

At step 0, validation runs before training and the initial model is stored as a
best-model snapshot. At each later validation, patience resets when either the
absolute gain reaches `min_delta` or the relative gain reaches
`min_relative_delta`. Before the latch, `sft_patience` misses switch the round
to TrajToSFT/SFT. Once the best score reaches `latch_threshold`, SFT is disabled
for the rest of the run and `finish_patience` or `total_rl_steps` terminates RL.

After the latch, rollout JSONL, validation and W&B metrics continue; only
rollout image capture stops.

## RL configuration

New configurations use the backend-native `slime` block:

```yaml
general_overrides:
  rl:
    training_steps: 600
    slime:
      train_envs: /absolute/path/to/train.yaml
      eval_envs: /absolute/path/to/val.yaml
      harness: concat
      algorithm: default_gae
      rollout_batch_size: 128
      n_samples_per_prompt: 1
      num_gpus: 8
      actor_gpus: 4
      rollout_gpus: 4
      eval_interval: 20
      save_interval: 20
      rollout_max_prompt_len: 4000
      rollout_max_response_len: 10000
      actor_lr: 1.0e-6
      critic_lr: 1.0e-5
      record_rollout_images: true
```

Historical `hydra_overrides` fields that occur in existing configs are
translated once by `graphrl.slime.config`; they are never passed to the old
Hydra/verl trainer. New experiments should use `slime` directly.

The current migration intentionally uses synchronous SLIME training:
rollout, training, evaluation and checkpoint commit happen in order. No async
training driver is selected by the GraphRL examples.

## Environment integration

VAGEN-SLIME owns the harness lifecycle. ViewAgent owns task behavior. The
adapter registers these environment names:

- `Path2View`, `View2Path`, `InteractiveViewPlanning`
- `Ai2ThorPath2View`, `Ai2ThorView2Path`, `Ai2ThorInteractiveViewPlanning`
- `HabitatGSPath2View`, `HabitatGSView2Path`,
  `HabitatGSInteractiveViewPlanning`

The adapter converts the established four-value ViewSuite step API to the
VAGEN-SLIME contract, preserves task metrics, supports concat/no-concat/compact
harnesses, and resolves historical `*_filter.jsonl` names to the canonical
dataset file when that sibling exists.

Sokoban uses the VAGEN-SLIME environment implementation through GraphRL's
compatibility class.

## Output layout

Each fixed or adaptive round keeps the existing GraphRL phase layout:

```text
iter_XXX/
├── rl/
│   ├── rl_model/                 # promoted final/best HF model
│   ├── rollout_data/             # <step>.jsonl, image_<step>/, <step>.complete
│   ├── slime_checkpoints/        # actor/critic/dataset resume state
│   ├── hf_checkpoints/           # HF exports by rollout id
│   └── rl_training.log
├── traj_to_sft/
│   ├── graph/
│   └── sft_data/
└── sft/
    └── sft_model/
```

The deprecated cleanup alias `verl_checkpoints` maps to
`rl/slime_checkpoints` so old cleanup lists still work.

## Resume guarantees

A resumable RL checkpoint is committed only after all required state is
durable:

- actor/model, optimizer, scheduler and RNG state;
- critic state when the algorithm uses a critic;
- global rollout-dataset cursor;
- controller state for adaptive runs;
- the matching complete rollout JSONL prefix;
- same-step evaluation when evaluation is scheduled.

The commit manifest is written last. On restart, GraphRL ignores incomplete
or ahead-of-rollout checkpoints, rewrites SLIME's tracker to the newest
consistent checkpoint, and removes newer checkpoint/HF/rollout artifacts before
continuing. A best-model snapshot is model-only and is never treated as a
mid-training optimizer/RNG resume point; the step-0 snapshot may be used as a
cold-start fallback.

Each RL phase also receives a stable W&B run id derived from its experiment
directory and round, so restarting that phase appends to the same W&B run.

SFT resume similarly rejects incomplete DeepSpeed/optimizer/RNG checkpoints.

## TrajToSFT and SFT

TrajToSFT remains the ViewAgent extension point. A pipeline supplies a dotted
module path under `general_overrides.traj_to_sft.module`; graph construction and
dataset generation stay under `GraphRL/graphrl`. LLaMA-Factory remains the SFT
backend.

Set a phase to `null` in a fixed round to skip it. The controller materializes
the expected model path with a symlink where appropriate, preserving the
existing directory and resume contract.

## Validation

The fast migration checks are:

```bash
cd /path/to/ViewAgent-slime

env PYTHONPATH="$PWD:$PWD/GraphRL:$PWD/GraphRL/VAGEN-SLIME:$PWD/GraphRL/VAGEN-SLIME/slime" \
  ../conda_envs/slime/bin/python -m pytest \
  GraphRL/tests view_suite/habitat_gs/tests -q

find GraphRL/examples examples/evaluation -name '*.sh' -print0 \
  | xargs -0 -n1 bash -n
```

Standalone evaluation usage and compatibility details are documented in
`../examples/evaluation/README.md`.
