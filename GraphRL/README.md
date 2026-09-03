# GraphRL on VAGEN/verl

This checkout uses **VAGEN** (verl-backed) for both the environment harness and
RL training. `GraphRL/VAGEN` and `GraphRL/LLaMA-Factory` are pinned Git
submodules; ViewAgent integration code stays outside them.

The ownership boundary is:

- `GraphRL/VAGEN/`: the VAGEN harness, agent loop and PPO trainer, with its own
  pinned `verl` submodule underneath.
- `GraphRL/LLaMA-Factory/`: the pinned SFT backend.
- `GraphRL/graphrl/configs/vagen_configs/`: GraphRL's own RL defaults and the
  ViewSuite environment registry.
- `GraphRL/graphrl/vagen/`: the launch seam — builds the `vagen.training.main`
  command line and owns the subprocess and checkpoint promotion.
- `GraphRL/graphrl/main.py`: fixed `RL -> TrajToSFT -> SFT` controller.

ViewSuite's environments are registered into VAGEN directly, without an adapter:
they already implement VAGEN's `GymBaseEnv` contract, because that is the
contract they were written against.

## Environment setup

```bash
cd /path/to/ViewAgent
git submodule update --init --recursive   # also fetches VAGEN's verl
```

`verl` is used as a checkout on `PYTHONPATH`, not as an installed package, and
it must come first so it wins over any installed copy — the training env ships
verl 0.6.1 as a package, which would otherwise shadow it. The example scripts
source `examples/_vagen_env.sh`, which does this, selects the adjacent
`../conda_envs/slime/bin/python` when available, and fails early with a readable
message if the `verl` submodule was never initialised.

For ViewSuite runs, datasets and renderer URL files are resolved below
`VIEWSUITE_ROOT`, which defaults to the ViewAgent checkout root:

```bash
VIEWSUITE_ROOT=/path/to/assets bash examples/viewsuite/.../run.sh
```

Render services served over HTTPS with a self-signed certificate need
`RENDER_TLS_NO_VERIFY=1`, or `SSL_CERT_FILE` pointed at the certificate.

## Entry points

```bash
cd /path/to/ViewAgent/GraphRL

bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh
bash examples/viewsuite/ai2thor_interactive_view_planning/run.sh
bash examples/viewsuite/habitat_gs_interactive_view_planning/run.sh
```

The baseline, smoke, Qwen3-VL, self-boot and random-SFT scripts in the same tree
use the same backend. Hydra overrides remain supported:

```bash
bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh \
  general_overrides.rl.hydra_overrides.trainer.test_freq=10 \
  general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node=8
```

To validate an entry point without allocating GPUs or starting training:

```bash
bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh --cfg job
```

## How the command line is assembled

`graphrl/vagen/utils/command_builder.py` launches
`python3 -m vagen.training.main` with **VAGEN's own**
`vagen/configs/vagen_multiturn.yaml` as the primary Hydra config. It has to be
the primary one: it sets `hydra.searchpath`, and Hydra allows that only there.
GraphRL contributes three things on top, in increasing priority:

1. **VAGEN's `vagen/configs/baseline_vllm.flags`**, read from the submodule
   rather than copied. These are the flags that make a run work at all — most
   importantly the two that select VAGEN's agent loop. Without them verl runs
   its own, and the job comes up looking healthy while none of VAGEN's rollout
   code executes.
2. **`graphrl/configs/vagen_configs/env_registry.yaml`**, emitted as
   `+env_registry.<Name>=<dotted.path>` appends.
3. **`hydra_overrides`** from `vagen_configs/config.yaml` and the pipeline.

A baseline flag that an experiment also sets is dropped rather than repeated.
Hydra would take the last value either way, but leaving both makes the effective
value invisible in the logged command — and that is the command people read when
a run does not do what the config says.

Nothing that merely restates a VAGEN default belongs in GraphRL's configs:
restating one turns a backend upgrade into a silent no-op.

## RL configuration

Configurations use verl's own key space under `hydra_overrides`:

```yaml
general_overrides:
  rl:
    training_steps: 600
    hydra_overrides:
      data:
        max_prompt_length: 4000
        max_response_length: 10000
        train_batch_size: 128
      algorithm:
        adv_estimator: token_level_gae
      actor_rollout_ref:
        actor:
          ppo_mini_batch_size: 128
          checkpoint:
            save_contents: [model, hf_model, optimizer, extra]
        rollout:
          n: 1
          gpu_memory_utilization: 0.6
      critic:
        enable: true
      trainer:
        harness: concat            # concat | no_concat | compact
        n_gpus_per_node: 8
        save_freq: 20
        test_freq: 20
```

`save_contents` must include `hf_model`: the checkpoint monitor promotes
`<ckpt>/actor/huggingface` into `rl_model/`, and that directory only exists when
the HF export is written.

Runs are **colocated by construction**. verl's hybrid engine puts actor, critic
and rollout on the same GPUs, so there is no placement knob — and no equivalent
of the previous backend's disjoint `actor_gpus`/`rollout_gpus` split.

## Environment integration

The registry names these environments:

- `Path2View`, `View2Path`, `InteractiveViewPlanning`
- `Ai2ThorPath2View`, `Ai2ThorView2Path`, `Ai2ThorInteractiveViewPlanning`
- `HabitatGSPath2View`, `HabitatGSView2Path`, `HabitatGSInteractiveViewPlanning`

They are appended to VAGEN's built-ins (`Sokoban`, `FrozenLake`, `SpatialGym`,
`PrimitiveSkill`, `RemoteEnv`), which stay registered. Because these are Hydra
`+` appends, the file must not repeat a name VAGEN already defines.

Environment YAMLs name the canonical dataset files. The historical
`*_filter.jsonl` spelling referred to the same already-filtered corpora and is
no longer resolved for you — the fallback that used to do that lived in the
removed SLIME adapter.

## Output layout

```text
iter_XXX/
├── rl/
│   ├── rl_model/                 # promoted final/best HF model
│   ├── rollout_data/             # <step>.jsonl, image_<step>/, <step>.complete
│   ├── verl_checkpoints/         # actor/critic/dataset resume state
│   └── rl_training.log
├── traj_to_sft/
│   ├── graph/
│   └── sft_data/
└── sft/
    └── sft_model/
```

## TrajToSFT and SFT

TrajToSFT remains the ViewAgent extension point. A pipeline supplies a dotted
module path under `general_overrides.traj_to_sft.module`; graph construction and
dataset generation stay under `GraphRL/graphrl`. LLaMA-Factory remains the SFT
backend.

Set a phase to `null` in a fixed round to skip it. The controller materializes
the expected model path with a symlink where appropriate, preserving the
existing directory and resume contract.

## Not carried over from the SLIME backend

- **The metric-driven adaptive controller** (`graphrl.main_adaptive`,
  `run_adaptive.sh`, `pipeline_adaptive.yaml`). It drove a
  `vagen.main_ppo_adaptive` entrypoint that this VAGEN does not have; only
  `vagen.training.main` exists. Re-enabling it means adding that entrypoint to
  VAGEN, not changing anything here.
- **Self-reasoning TrajToSFT augmentation**
  (`graphrl.traj_to_sft.self_reasoning.augment.run_vagen_eval_and_collect`).
  VAGEN's evaluation package exposes `run_eval_parallel(...)`, whose signature
  and result layout differ from the SLIME runner this was written against, so
  it is not a rename. It raises `NotImplementedError` where the gap is. No
  shipped pipeline reaches it.

## Validation

```bash
cd /path/to/ViewAgent

env PYTHONPATH="$PWD:$PWD/GraphRL:$PWD/GraphRL/VAGEN:$PWD/GraphRL/VAGEN/verl" \
  ../conda_envs/slime/bin/python -m pytest \
  GraphRL/tests view_suite/habitat_gs/tests -q

find GraphRL/examples examples/evaluation -name '*.sh' -print0 \
  | xargs -0 -n1 bash -n
```

Standalone evaluation usage and compatibility details are documented in
`../examples/evaluation/README.md`.
