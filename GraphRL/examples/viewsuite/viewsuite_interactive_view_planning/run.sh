#!/usr/bin/env bash
# =============================================================================
# Run GraphRL pipeline for ViewSuite Interactive View Planning (v2)
# =============================================================================
# Changes from v1:
#   - Uses pipeline.yaml with prefer_single_action knob
#
# Usage:
#   bash run_v2.sh
#   bash run_v2.sh iterations=5
#
# experiment_dir is computed by pipeline.yaml as
#   exps/viewsuite/viewsuite_interactive_view_planning_v2/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_slime_env.sh"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/viewsuite_interactive_view_planning"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"
echo "Using ${N_GPUS_PER_NODE} GPU(s) for RL and ${SFT_N_GPUS} GPU(s) for SFT"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${SLIME_PYTHON}" -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    general_overrides.rl.slime.train_envs="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.slime.eval_envs="${SCRIPT_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.slime.num_gpus="${N_GPUS_PER_NODE}" \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    iteration_overrides.iter0.rl.training_steps=65 \
    iteration_overrides.iter1.rl.training_steps=65 \
    iteration_overrides.iter2.rl.training_steps=65 \
    iteration_overrides.iter3.rl.training_steps=300 \
    general_overrides.traj_to_sft.graph_builder.atomize.enabled=true \
    general_overrides.traj_to_sft.graph_builder.merge_tol.position=0.2 \
    general_overrides.traj_to_sft.graph_builder.merge_tol.angle=10.0 \
    +iteration_overrides.iter3.rl.slime.train_envs="${SCRIPT_DIR}/train.yaml" \
    +iteration_overrides.iter3.rl.slime.record_rollout_images=false \
    "$@" 2>&1 | tee "${LOG_FILE}"
