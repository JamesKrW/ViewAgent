# Multi-turn agent RL on Sokoban (no_concat + PPO/GAE)

Training an agent on [VAGEN](https://github.com/RAGEN-AI/VAGEN)'s Sokoban environment, using
PPO with a critic. The agent plays a gridworld over up to 5 turns: each turn it sees the
board as a rendered frame, writes an action, and the environment scores it and returns the
next board. Every turn carries an image, so this exercises the multimodal path end to end.

The three things this example picks on the four axes:

| axis | value | where |
|---|---|---|
| env | `Sokoban` vision, 5 turns, seeds 1–10000 (train) / 10001–10256 (val) | [`train.yaml`](train.yaml) |
| harness | `no_concat` — one conversation, and therefore one training row, **per turn** | [`no_concat_ppo.yaml`](no_concat_ppo.yaml) |
| model | `qwen` adapter, Qwen3-VL-4B-Instruct | `--model-name` |
| algorithm | `default_gae` — episode-global GAE, reached through slime's own per-row GAE | [`no_concat_ppo.yaml`](no_concat_ppo.yaml) |

The multi-turn rollout is implemented through a
[custom generate function](../../../vagen_agent/rollout/runner.py), overriding the
original generate function. It returns a `list[Sample]` — one per conversation — sharing
a `rollout_id`, which is slime's native representation of an episode that spans several
rows.

The environment scores its own actions inside the episode, so there is no reward model:
`--rm-type` is unset and `sample.reward` is already populated when the rollout returns.

<details>
<summary>Environment API (vagen_agent/envs/_common/common.py)</summary>

- `__init__(env_config: dict)`: stores the config; nothing expensive.
- `reset(seed: int) -> tuple[dict, dict]`: returns the first observation and info.
- `system_prompt() -> dict`: the instructions, represented as an observation.
- `step(response) -> tuple[dict, float | list[float], bool, dict]`: parses the model
  response, advances the world, and returns observation, reward, done, info. Set
  `info["truncated"]` to distinguish "ran out of turns" from "finished".
- `close()`: optional.

An environment observation is
`{"obs_str": str, "multi_modal_input": {"<image>": [PIL.Image, ...]}}`; the adapter
turns it into the model-facing message and keeps `<image>` markers 1:1 with frames.

Adding an environment is a class, one line in
[`vagen_agent/configs/env_registry.yaml`](../../../vagen_agent/configs/env_registry.yaml), and
one dataset row — no change to the rollout, the harness or the launcher.
</details>

## Results

The historical numbers below came from a Qwen3-4B-Instruct-2507 run. The current example
defaults to Qwen3-VL because its environment configuration emits image observations.
That run used 4 GPUs (2 actor / 2 sglang), 60 rollout steps, and `train.py`.
Held-out set is 256 episodes, seeds disjoint from training (verified, not assumed).

| step | 0 | 20 | 40 | 60 |
|---|---|---|---|---|
| held-out `env/success` | 0.258 | 0.434 | 0.570 | 0.563 |

Training-side over the same run: success 0.298 → 0.515, mean turns 3.99 → 3.19, mean reward
0.206 → 0.328. Step 0 is the **untrained** baseline — both `train.py` and the pinned
`train_async.py` evaluate before the first update, which is what distinguishes a learning
curve from an easy eval set.

## Verifying a rollout without training

slime's `--debug-rollout-only` brings up the sglang engines and *no* Megatron actors, so a
model needs no `torch_dist` conversion to be checked and a 2-GPU box finishes in minutes.
`--dump-details` writes every `Sample` to disk; `tests/verify_rollout_dump.py` reads it back
and checks that the loss mask covers the model's own tokens and nothing else.

```bash
python examples/train/sokoban/run_sokoban.py --rollout-only --num-gpus 2 \
    --model-name Qwen3-VL-4B-Instruct --eval-yaml examples/train/sokoban/verify.yaml
python tests/verify_rollout_dump.py runs/<exp>/dump/rollout_data/eval_0.pt \
    --model ~/models/Qwen3-VL-4B-Instruct

bash tests/verify_qwen_matrix.sh          # every supported family, both thinking modes
```

`--chat-template-kwargs '{"enable_thinking": false}'` reaches the model adapter through
slime's `--apply-chat-template-kwargs`. It matters on Qwen3.5 and is ignored by every other
Qwen family here -- see the matrix script for what that changes.

## Reproduce

```bash
# 1) Set environment variables
conda activate /path/to/slime-env
export WANDB_API_KEY=...
export SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-4B-Instruct
export SLIME_SCRIPT_NUM_GPUS=4

# 2) Download the model (the dataset is generated, not downloaded -- see below)
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir ~/models/Qwen3-VL-4B-Instruct

# 3) Evaluate before training. Do this first for any new model or environment.
python examples/train/sokoban/run_sokoban.py --num-rollout 0

# 4) Train
python examples/train/sokoban/run_sokoban.py --actor-gpus 2 --tp 2
```

Step 3 is not optional advice. `--num-rollout 0` is slime's own eval-only path
(`train.py`: `if args.num_rollout == 0 and args.eval_interval is not None`). It exercises
the entire rollout — model adapter, harness, environment, reward, metrics — against the
held-out set and reports a success rate, without spending anything on training. A rollout
that is subtly wrong shows up there as a number you can judge; started as a training run it
shows up as a loss curve that looks perfectly plausible.

There is no dataset to download: `run_sokoban.py` builds the jsonl from the env yaml with
`vagen_agent.make_dataset` whenever the yaml is newer. Seed expansion is deterministic in
`--base-seed`, so the data stays in step with the config for free.

Every field of `Config` in the launcher is both a `--flag` and a `SLIME_SCRIPT_<NAME>` env
var (slime's `dataclass_cli`); `--help` lists them.

## Varying it

| change | how |
|---|---|
| context policy | `harness:` in `no_concat_ppo.yaml` (`concat` / `no_concat` / `compact`) |
| algorithm | `algorithm:` in the same file |
| environment | `--envs-yaml <your>.yaml` (the env rides per-row in the dataset) |
| add an eval env | one entry under `datasets:` in `eval_datasets.yaml`, no launcher change |
| model | `--model-name ...` (needs a matching `scripts/models/*.sh` in slime) |

## What each file does

- `run_sokoban.py`: converts the checkpoint, builds the datasets, sets training/rollout args,
  and launches — all through `slime.utils.external_utils.command_utils`, so process cleanup,
  `ray start`, the runtime-env JSON and the wandb naming stay slime's.
- `no_concat_ppo.yaml`: `--custom-config-path`. Which harness, which algorithm, which model
  adapter, and the per-row defaults. This is the file that says what the run *is*.
- `eval_datasets.yaml`: `--eval-config`. The held-out sets, in slime's own multi-dataset shape, each
  with its own sampling parameters.
- `megatron_roles.yaml`: `--megatron-config-path`. The critic's learning rate — slime has no
  `--critic-lr`, and a role-tagged Megatron config is the channel.
- `train.yaml` / `val.yaml`: the environment configurations -- which levels, how many turns,
  what a turn is worth. Copies of VAGEN's, because they *are* the experiment.
- `verify.yaml`: the same thing cut to 32 episodes, for `--rollout-only`.
- `ALIGNMENT.md`: every VAGEN hyperparameter mapped to its slime equivalent, including the
  ones deliberately not set and the gaps still open.
