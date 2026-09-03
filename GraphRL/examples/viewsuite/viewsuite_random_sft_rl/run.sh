#!/usr/bin/env bash
# =============================================================================
# Run GraphRL pipeline for ViewSuite Random-Action SFT → RL
# =============================================================================
#
# Phase shape per iteration matches ``viewsuite_interactive_view_planning``:
#   RL → TrajToSFT → SFT
#
# The only difference: TrajToSFT does NOT reuse the just-finished RL
# rollouts. It launches ``view_suite.evaluation.run_eval`` through the
# ``random_navigation`` backend on the TRAINING split, converts the dump
# into GraphRL rollout format, builds a graph with InteractiveViewPlanningGraphBuilder,
# and then generates the SFT datasets with InteractiveViewPlanningSFTGenerator — so
# SFT is always trained on fresh random-action data, with a new random seed
# every iteration.
#
# Requirements:
#   - The ViewSuite rendering client must be up; its URL lives in
#     ${VIEWSUITE_ROOT}/client_url.txt (same as the other pipelines).
#
# Usage:
#   bash run.sh
#   bash run.sh iterations=5
#
# experiment_dir is computed by pipeline.yaml as
#   exps/viewsuite/viewsuite_random_sft_rl/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_vagen_env.sh"

EXPERIMENT_DIR="${PWD}/exps/viewsuite/viewsuite_random_sft_rl"

echo "=== GraphRL: ViewSuite Random-Action SFT → RL ==="
echo "ViewSuite root: ${VIEWSUITE_ROOT}"
echo "Experiment dir: ${EXPERIMENT_DIR}"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${VAGEN_PYTHON}" -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node=8 \
    general_overrides.rl.hydra_overrides.trainer.log_image.enable=false \
    general_overrides.sft.n_gpus=8 \
    general_overrides.traj_to_sft.eval_config="${SCRIPT_DIR}/collect_random_train.yaml" \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    +iteration_overrides.iter3.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train.yaml" \
    "$@" 2>&1 | tee "${LOG_FILE}"
