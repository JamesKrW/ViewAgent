#!/usr/bin/env bash
# =============================================================================
# Sokoban (text) — Pure-RL Baseline (1 iteration, ~401 steps, no SFT)
# =============================================================================
#
# Usage:
#   bash run_pure_rl.sh
#   bash run_pure_rl.sh general_overrides.rl.training_steps=200
#
# experiment_dir is computed by pipeline_pure_rl.yaml as
#   exps/sokoban/sokoban_text_pure_rl/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_slime_env.sh"
EXPERIMENT_DIR="${PWD}/exps/sokoban/sokoban_text_pure_rl"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${SLIME_PYTHON}" -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline_pure_rl \
    general_overrides.rl.slime.train_envs="${SCRIPT_DIR}/train.yaml" \
    general_overrides.rl.slime.eval_envs="${SCRIPT_DIR}/val.yaml" \
    "$@" 2>&1 | tee "${LOG_FILE}"
