# Vendored VAGEN-SLIME

`GraphRL/VAGEN-SLIME` replaces the legacy VAGEN/VERL tree with a source snapshot of:

- `JamesKrW/VAGEN-SLIME` at `056e139`
- its `JamesKrW/slime` dependency at `b533d81`

The environment integrations and experiment launchers remain outside the
vendored tree, under `GraphRL/` and `view_suite/`. The only local changes kept
inside the snapshot are generic H200 build/runtime fixes, a configurable
rollout-function seam, and the Qwen2.5-VL/Megatron checkpoint compatibility
fixes required by the validated synchronous PPO and resume path.

## Local patch boundary

The project-specific code is outside `GraphRL/VAGEN-SLIME`:

- `GraphRL/examples/viewsuite/.../slime/`: launch and experiment configuration
- `GraphRL/graphrl/slime/`: environment-registration seam
- `view_suite/envs/habitat_gs_proxy_task/slime_adapter.py`: task adapter

Changes retained inside the vendored backend are limited to:

- `scripts/build_slime_env.sh` and the reusable Sokoban launcher: reproducible
  Hopper/Blackwell builds, explicit model/config paths, and Ray worker library paths
- `slime/backends/megatron_utils/`: Qwen2.5-VL Bridge loading/export and critic
  checkpoint compatibility
- `slime/utils/torch_memory_saver_utils.py` and `slime/ray/actor_group.py`:
  CUDA-version-aware memory-saver preload handling
- `slime/utils/external_utils/command_utils.py`: reliable local Ray dashboard startup
- corresponding focused regression tests
