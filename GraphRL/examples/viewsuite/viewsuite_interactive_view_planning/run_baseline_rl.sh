#!/usr/bin/env bash
# =============================================================================
# Run GraphRL pipeline — ViewSuite Interactive View Planning (RL-only baseline)
# =============================================================================
#
# Single iteration, RL only (no TrajToSFT/SFT).  Reuses pipeline.yaml from
# the active-exploration example but overrides experiment_name so this run
# lands in its own dir alongside the main pipeline.
#
# Usage:
#   bash run_baseline_rl.sh
#   bash run_baseline_rl.sh iterations=2
#
# experiment_dir is exps/viewsuite/viewsuite_interactive_view_planning_baseline_rl/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_vagen_env.sh"
EXPERIMENT_NAME="viewsuite_interactive_view_planning_baseline_rl"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/${EXPERIMENT_NAME}"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${VAGEN_PYTHON}" -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    experiment_name="${EXPERIMENT_NAME}" \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=1 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node=8 \
    general_overrides.sft.n_gpus=8 \
    'general_overrides.traj_to_sft.generators=[action_gen,path_to_view,multi_turn_action_gen]' \
    iteration_overrides.iter0.rl.training_steps=1000 \
    +iteration_overrides.iter0.rl.timeout=259200 \
    "$@" 2>&1 | tee "${LOG_FILE}"
