#!/usr/bin/env bash
# =============================================================================
# Run GraphRL pipeline for Sokoban (text-only mode)
# =============================================================================
#
# Usage:
#   bash run.sh
#   bash run.sh iterations=5
#   bash run.sh general_overrides.rl.training_steps=200
#   bash run.sh initial_model_path=Qwen/Qwen2.5-1.5B-Instruct
#
# All SLIME/SFT parameters live in pipeline.yaml (same directory).
# experiment_dir is computed by pipeline.yaml as
#   exps/${project_name}/${experiment_name}  →  exps/sokoban/sokoban_text/
# resolved relative to the current working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_slime_env.sh"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${SLIME_PYTHON}" -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    general_overrides.rl.slime.train_envs="${SCRIPT_DIR}/train.yaml" \
    general_overrides.rl.slime.eval_envs="${SCRIPT_DIR}/val.yaml" \
    "$@"
