# VAGEN → slime setting map

Every hyperparameter VAGEN ships for Sokoban, and where it lives on this port. Sources:
`vagen/configs/baseline_vllm.flags`, `vagen/configs/vagen_multiturn.yaml`,
`examples/train/sokoban/train_*_qwen35_4b.sh`, and verl's own defaults
(`verl/trainer/config/_generated_ppo_megatron_trainer.yaml`) for anything VAGEN does not
override.

Kept as a file rather than as comments because the *absences* matter as much as the
values: a setting VAGEN does not set is a setting this port must not set either, and that
is invisible in a diff of two launch scripts.

## Matched

| VAGEN | value | slime |
|---|---|---|
| `data.train_batch_size` | 128 | `--rollout-batch-size 128` |
| `actor_rollout_ref.rollout.n` | 1 | `--n-samples-per-prompt 1` |
| `actor.optim.lr` | 1e-6 | `--lr 1e-6` |
| `critic.optim.lr` | 1e-5 | `megatron_roles.yaml` → `role: critic, lr: 1.0e-5` |
| `algorithm.kl_ctrl.kl_coef` | 0.0 | `--kl-coef 0.0` |
| `actor.use_kl_loss` | False | `--kl-loss-coef 0.0`, `--use-kl-loss` absent |
| `actor.entropy_coeff` | 0.0 | `--entropy-coef 0.0` |
| `algorithm.gamma` / `lam` | 1.0 / 1.0 | `--gamma 1.0 --lambd 1.0` |
| `algorithm.adv_estimator` | `default_gae` | `algorithm: default_gae` in the yaml → `--advantage-estimator ppo` |
| `trainer.harness` | `no_concat` | `harness: no_concat` in the yaml |
| `trainer.critic_warmup` | 0 | `--num-critic-only-steps 0` (slime default) |
| `clip_ratio` (verl default) | 0.2, symmetric | `--eps-clip 0.2`, **no** `--eps-clip-high` |
| `ppo_epochs` (verl default) | 1 | slime default |
| `grad_clip` (verl default) | 1.0 | `--clip-grad 1.0` (slime default) |
| `loss_agg_mode` (verl default) | `token-mean` | slime's `sum_of_sample_mean` — see caveats |
| `response_length_per_turn` | 512 | env yaml → dataset `metadata` |
| `data.max_response_length` (concat / compact) | 4000 | `run_sokoban.Config.rollout_response_tokens` → `--rollout-max-response-len 4000`; individual calls are still capped by the 512-token row metadata |
| `max_turns` | 5 | env yaml → dataset `metadata` |
| `max_env_response_per_turn` | 320 for the Qwen3-VL vision recipe | env yaml → dataset `metadata` |
| base `trainer.compact_budget` / derived summary cap | 4000 / 512 | `compact_ppo.yaml` |
| `trainer.total_training_steps` | 401 | `--num-rollout 401` |

## Deliberately different

| VAGEN | why it cannot carry over |
|---|---|
| `actor.ppo_mini_batch_size=32` | This experiment deliberately uses `--num-steps-per-rollout 1`, i.e. a full 128-episode global batch and one optimizer update per rollout, matching slime's Geo3K schedule. |
| `data.max_prompt_length=1000` | verl pads a row into two fixed-width regions. slime rows are variable-length; there is no prompt region to size. |
| `data.max_response_length=800` (no_concat) | The shared launcher remains at 4000, while a no-concat row contains one action capped at 512. Thus the effective row stays below VAGEN's 800 without coupling the shared launcher to the selected harness. |
| `actor.fsdp_config.*_offload`, `use_remove_padding`, `use_fused_kernels`, `ppo_micro_batch_size_per_gpu` | verl/FSDP knobs. The Megatron equivalents are `--use-dynamic-batch-size --max-tokens-per-gpu` and `--recompute-*`. |
| `rollout.name=vllm`, `gpu_memory_utilization`, `enforce_eager`, `max_num_batched_tokens` | vLLM knobs; the backend here is sglang. |
| `trainer.n_gpus_per_node=4` | Formal runs use all 8 GPUs: B200 uses 2 train + 6 rollout; A100 uses 4 train + 4 rollout. |

## Closed since first draft

* **`success` is now a metric.** `vagen_agent/metrics.py` goes through
  `--custom-rollout-log-function-path`, which replaces slime's default logging -- so it
  re-computes slime's own metrics first and adds `env/*` on top. Episode facts are
  deduplicated by `rollout_id`: under `no_concat` an episode is several Samples carrying
  identical metadata, and averaging over samples would weight it by turn count, so a policy
  that learned to stall would *raise* the apparent success rate. Also publishes
  `env/samples_per_episode`, which is the layout made visible (1 under concat, the turn count
  under no_concat, and a quantity that *moves during training* under compact).
* **Held-out eval.** slime's own `--eval-config eval_datasets.yaml` with `--eval-interval 20`
  (VAGEN's `trainer.test_freq=20`). Richer than the `--eval-prompt-data <name> <path>` pair
  in exactly the way a multi-environment run needs: each dataset carries its own sampling
  parameters and its own `metadata_key`, so adding an eval environment is a yaml entry
  rather than a launcher change. The val set is VAGEN's own
  `val_sokoban_free_wm_text.yaml`: 256 episodes, seeds 10001-10256 against the training
  set's 1-10000 — verified zero overlap, not assumed.

## Known gaps — not yet aligned

0. **`loss_agg_mode`.** verl's `token-mean` averages over all tokens in the batch; slime's
   default reducer is a per-sample mean summed over samples, and under `no_concat` a
   "sample" is one turn, so a long episode contributes more turns and therefore more
   weight. slime has `--custom-pg-loss-reducer-function-path` if this needs matching.
   Unmeasured — flagged, not fixed.
2. **`1 - rollout/truncated` is NOT the success rate.** It was used as a proxy before
   `env/success` existed, and it undercounts by 2.36x on Sokoban (measured over 20
   steps: direct 0.342 against proxy 0.145). `BaseEnv.step` now keeps termination and
   truncation separate while enforcing the configured turn limit. Any figure quoted from
   a run before the metrics hook landed is that proxy, not the success rate.
3. **Eval sampling temperature.** slime falls back to `rollout_temperature` (1.0) for eval
   when `--eval-temperature` is unset, so the held-out numbers carry sampling noise. Left at
   the default for now because it is what VAGEN's val does; worth revisiting if the eval
   curve turns out too noisy to read at 256 episodes.

## Compact budget choice

VAGEN has two relevant settings, for two different experiments. Its base config uses
`compact_budget=4000` and derives a 512-token summary cap; a normal five-turn Sokoban
episode then usually remains one conversation and compact is close to concat. Its
`train_default_gae_compact_qwen25vl3b.sh` ablation overrides those to `1200/300` on
purpose, so compaction actually fires (about 0.4 times per episode in the measurement
recorded there).

This port uses the base `4000/512` behavior. The former `1600/256` pair matched neither
VAGEN setting and made compaction frequency rise sharply as the learned policy became
more verbose, changing the training-row layout late in training.

## Reusable slime mechanisms not adopted yet, and why

* **`--custom-rm-path` / `slime.rollout.rm_hub`.** The scoring registry every other example
  uses. Ours does not, and should not: the environment scores its own actions inside the
  episode, so there is nothing left for a reward model to see at the end.
* **`--use-tis`** (truncated importance sampling, off-policy correction). Relevant once we
  move to `train_async.py` or fully-async, where a rollout is generated under older weights.
  Untried.
* **`--disable-compute-advantages-and-returns`.** Useful for a rollout-only measurement run
  (e.g. sweeping the per-turn budget without paying for training), which is what a "B" style
  experiment wants.
