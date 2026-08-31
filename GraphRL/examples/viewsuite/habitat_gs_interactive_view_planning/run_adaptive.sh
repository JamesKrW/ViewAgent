#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_slime_env.sh"

EXPERIMENT_DIR="${PWD}/exps/viewagent/habitat_gs_ivp_relative_earlystop"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"
mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

export WANDB_MODE="${WANDB_MODE:-online}"

"${SLIME_PYTHON}" -m graphrl.main_adaptive \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline_adaptive \
    general_overrides.rl.slime.train_envs="${SCRIPT_DIR}/train.yaml" \
    general_overrides.rl.slime.eval_envs="${SCRIPT_DIR}/val.yaml" \
    general_overrides.rl.slime.num_gpus="${N_GPUS_PER_NODE}" \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    experiment_dir="${EXPERIMENT_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
