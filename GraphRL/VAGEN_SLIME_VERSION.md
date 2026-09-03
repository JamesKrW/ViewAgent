# Pinned VAGEN-SLIME submodule

`GraphRL/VAGEN-SLIME` replaces the legacy VAGEN/VERL tree with a recursive
submodule pinned to:

- `JamesKrW/VAGEN-SLIME` at `4704188975ee984e5a1826ea32554d80c738e6de`
- its `JamesKrW/slime` dependency at `36e70ca64071eeea71017ca395da8f14b763d976`

Initialize both this dependency and `GraphRL/LLaMA-Factory` with:

```bash
git submodule update --init --recursive
```

The environment integrations and experiment launchers remain outside the
submodule, under `GraphRL/` and `view_suite/`. The only changes kept inside the
pinned fork are generic H200 build/runtime fixes, a configurable
rollout-function seam, and the Qwen2.5-VL/Megatron checkpoint compatibility
fixes required by the validated synchronous PPO and resume path.

## Local patch boundary

The project-specific code is outside `GraphRL/VAGEN-SLIME`:

- `GraphRL/examples/viewsuite/.../slime/`: launch and experiment configuration
- `GraphRL/graphrl/slime/`: environment-registration seam
- `view_suite/envs/habitat_gs_proxy_task/slime_adapter.py`: task adapter

Changes retained inside the pinned backend are limited to:

- `scripts/build_slime_env.sh` and the reusable Sokoban launcher: reproducible
  Hopper/Blackwell builds, explicit model/config paths, and Ray worker library paths
- `slime/backends/megatron_utils/`: Qwen2.5-VL Bridge loading/export and critic
  checkpoint compatibility
- `slime/utils/torch_memory_saver_utils.py` and `slime/ray/actor_group.py`:
  CUDA-version-aware memory-saver preload handling
- `slime/utils/external_utils/command_utils.py`: reliable local Ray dashboard startup
- `vagen_agent/rollout/frames.py` and `vagen_agent/rollout/dump.py`: frame keys are
  hashed over raw pixels instead of PNG bytes, and frames encode as JPEG q90 by default
  (`VAGEN_FRAME_FORMAT=PNG` / `VAGEN_FRAME_QUALITY` override it). The old path
  PNG-encoded every frame just to compute its key, on the rollout event loop, which
  serialised all concurrent episodes behind one encode. Hashing pixels is both cheaper
  and lossless, and frames already in the table now skip encoding entirely. The JPEG
  default is lossy; set `VAGEN_FRAME_FORMAT=PNG` where that matters.
- corresponding focused regression tests
