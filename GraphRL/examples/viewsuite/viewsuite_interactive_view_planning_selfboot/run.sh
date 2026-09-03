#!/usr/bin/env bash
# =============================================================================
# Run GraphRL pipeline — Self-Bootstrapping Baseline
# =============================================================================
#
# Same RL settings as the graph-based pipeline, but SFT data comes from
# filtered high-reward RL trajectories (reward > 0.5) instead of graph
# sampling. Reuses the VAGEN environment configs from the original
# viewsuite_interactive_view_planning example.
#
# Usage:
#   bash run.sh
#   bash run.sh iterations=5
#
# experiment_dir is computed by pipeline.yaml as
#   exps/viewsuite/viewsuite_interactive_view_planning_selfboot/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_vagen_env.sh"
ORIG_EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/../viewsuite_interactive_view_planning" && pwd)"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/viewsuite_interactive_view_planning_selfboot"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${VAGEN_PYTHON}" -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    general_overrides.rl.hydra_overrides.data.train_files="${ORIG_EXAMPLE_DIR}/train_turn_format.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${ORIG_EXAMPLE_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node=8 \
    general_overrides.sft.n_gpus=8 \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    +iteration_overrides.iter3.rl.hydra_overrides.data.train_files="${ORIG_EXAMPLE_DIR}/train.yaml" \
    "$@" 2>&1 | tee "${LOG_FILE}"
