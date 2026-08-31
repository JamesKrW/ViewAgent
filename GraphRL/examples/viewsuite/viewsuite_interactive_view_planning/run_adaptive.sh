#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_slime_env.sh"

EXPERIMENT_DIR="${PWD}/exps/viewagent/viewsuite_ivp_relative_earlystop"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"
mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${SLIME_PYTHON}" -m graphrl.main_adaptive \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline_adaptive \
    general_overrides.rl.slime.train_envs="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.slime.eval_envs="${SCRIPT_DIR}/val.yaml" \
    general_overrides.rl.slime.num_gpus="${N_GPUS_PER_NODE}" \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    general_overrides.traj_to_sft.graph_builder.atomize.enabled=true \
    general_overrides.traj_to_sft.graph_builder.merge_tol.position=0.2 \
    general_overrides.traj_to_sft.graph_builder.merge_tol.angle=10.0 \
    experiment_dir="${EXPERIMENT_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
