#!/usr/bin/env bash
set -euo pipefail
: "${VIEWSUITE_ROOT:?VIEWSUITE_ROOT must be exported}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHRL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${VIEWSUITE_ROOT}:${GRAPHRL_ROOT}:${GRAPHRL_ROOT}/VAGEN:${GRAPHRL_ROOT}/VAGEN/verl:${GRAPHRL_ROOT}/LLaMA-Factory/src${PYTHONPATH:+:${PYTHONPATH}}"

# Keep the normal experiment on train.yaml/val.yaml while allowing controlled
# resolution ablations to select dedicated data configs from the launcher.
TRAIN_CONFIG="${AI2THOR_TRAIN_CONFIG:-train.yaml}"
VAL_CONFIG="${AI2THOR_VAL_CONFIG:-val.yaml}"
[[ "${TRAIN_CONFIG}" = /* ]] || TRAIN_CONFIG="${SCRIPT_DIR}/${TRAIN_CONFIG}"
[[ "${VAL_CONFIG}" = /* ]] || VAL_CONFIG="${SCRIPT_DIR}/${VAL_CONFIG}"

EXPERIMENT_DIR="${PWD}/exps/viewagent/ai2thor_ivp_relative_earlystop"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"
mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

export WANDB_MODE="${WANDB_MODE:-online}"

python -m graphrl.main_adaptive \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline_adaptive \
    general_overrides.rl.hydra_overrides.data.train_files="${TRAIN_CONFIG}" \
    general_overrides.rl.hydra_overrides.data.val_files="${VAL_CONFIG}" \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    experiment_dir="${EXPERIMENT_DIR}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
