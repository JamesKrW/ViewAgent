# VAGEN-SLIME

## Installation

```bash
git clone --branch main --recurse-submodules git@github.com:JamesKrW/VAGEN-SLIME.git
cd VAGEN-SLIME
git submodule update --init --recursive
bash scripts/build_slime_env.sh
conda activate ../conda_envs/slime
```

## Architecture

The extensible axes use one shared-contract package plus one directory per concrete type:

```text
vagen_agent/
├── envs/          # abstract BaseEnv, shared BaseVagenEnv, and Harness capability markers
├── harness/       # _common/ + concat/, no_concat/, compact/, ...
├── models/        # _common/ + qwen/, ...
├── algorithms/    # _common/ + one folder per credit-assignment algorithm
├── rollout/       # invariant token, trajectory, reward-alignment, and lifecycle core
└── evaluation/    # standalone runner; backends/ follows _common/ + concrete folders
```

Axis roots own registries and public APIs. Code in `_common/` must not depend on a
concrete implementation, and rollout code must use an axis's public facade rather than
branching on an environment, harness, model family, or algorithm name.

## Quickstart

Download the default Sokoban model once:

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir ~/models/Qwen3-VL-4B-Instruct
```

Run PPO with each Sokoban harness:

```bash
python examples/train/sokoban/run_sokoban.py \
  --model-name Qwen3-VL-4B-Instruct \
  --harness-config examples/train/sokoban/concat_ppo.yaml \
  --exp-name sokoban_concat

python examples/train/sokoban/run_sokoban.py \
  --model-name Qwen3-VL-4B-Instruct \
  --harness-config examples/train/sokoban/no_concat_ppo.yaml \
  --exp-name sokoban_no_concat

python examples/train/sokoban/run_sokoban.py \
  --model-name Qwen3-VL-4B-Instruct \
  --harness-config examples/train/sokoban/compact_ppo.yaml \
  --exp-name sokoban_compact
```

## Remote environments

`RemoteEnv` lets rollout workers use a stateful environment hosted behind a
FastAPI service, including environments with incompatible simulator or GPU
dependencies. See [vagen_agent/envs/remote/README.md](vagen_agent/envs/remote/README.md)
for the client config and service handler API.
