# Pinned VAGEN submodule

`GraphRL/VAGEN` is the RL backend: the VAGEN harness and agent loop on top of
verl. It replaces the previous `GraphRL/VAGEN-SLIME` (Megatron + SGLang) tree.

- `JamesKrW/VAGEN` on branch `main`
- its `JamesKrW/verl` dependency pinned beneath it, at the commit VAGEN's own
  `.gitmodules` records for tag `vagen-260814` (upstream verl v0.9.0 plus VAGEN's
  patch stack)

Initialize both this dependency and `GraphRL/LLaMA-Factory` with:

```bash
git submodule update --init --recursive
```

The nesting matters: `GraphRL/VAGEN/verl` is a submodule of a submodule, so a
non-recursive init leaves it present but empty. `examples/_vagen_env.sh` probes
for `verl/verl/trainer/config/ppo_trainer.yaml` and fails with that instruction
rather than letting the run continue to a Hydra error that never mentions verl.

## Local patch boundary

Nothing is patched inside `GraphRL/VAGEN`. Everything ViewAgent-specific lives
outside it:

- `GraphRL/graphrl/vagen/`: the launch seam — command construction, subprocess
  ownership, checkpoint promotion
- `GraphRL/graphrl/configs/vagen_configs/`: GraphRL's RL defaults and the
  ViewSuite environment registry
- `GraphRL/examples/viewsuite/`: experiment definitions
- `view_suite/envs/`: the task environments themselves

This is deliberate, and it is why there is no adapter layer. The ViewSuite
environments already implement VAGEN's `GymBaseEnv` contract — async
`reset(seed)` / `step(action_str)` / `system_prompt()` / `close()`, with `step`
returning the four-value `(obs, reward, done, info)` — so they are named
directly in the registry. The SLIME backend was the one that needed them
reshaped.

Two integration points are reached without editing the submodule:

- **Environment registration**: `vagen/training/agent_loop/base.py` resolves
  environments from the Hydra `env_registry` node, so ViewSuite's entries are
  appended with `+env_registry.<Name>=<path>` overrides.
- **Run-critical flags**: `vagen/configs/baseline_vllm.flags` is read from the
  submodule at launch rather than copied, so the two cannot drift.

## Previous backend

The SLIME-based backend is preserved on the `slime-backend` tag, including
`GraphRL/VAGEN-SLIME` and its nested `slime` pin. See that tag's
`GraphRL/VAGEN_SLIME_VERSION.md`.
