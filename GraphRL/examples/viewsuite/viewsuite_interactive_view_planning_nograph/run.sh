#!/usr/bin/env bash
# =============================================================================
# NO-GRAPH ablation (exp 02) — relabel-and-distill WITHOUT cross-trajectory
# graph composition. Same pipeline as ../viewsuite_interactive_view_planning
# (same RL, same 3 supervision signals, same schedule) but:
#   - traj_to_sft uses NoGraphInteractiveViewPlanningTrajToSFT (nodes scoped
#     per trajectory → no merge across trajectories), and
#   - sample_per_scene is halved (set in this dir's pipeline.yaml).
#
# This isolates the effect of GRAPH COMPOSITION vs. within-trajectory relabel.
#
# Usage:
#   bash run.sh
#   # small dev smoke (3 iters, 10 RL steps each, 1 SFT epoch):
#   bash run.sh iterations=3 \
#     iteration_overrides.iter0.rl.training_steps=10 \
#     iteration_overrides.iter1.rl.training_steps=10 \
#     iteration_overrides.iter2.rl.training_steps=10 \
#     general_overrides.sft.hydra_overrides.num_train_epochs=1
#
# Render fleet: export CLIENT_URL_FILE=/path/to/client_url_N.txt to point RL
# rollouts at a specific fleet (copied into ${VIEWSUITE_ROOT}/client_url.txt).
# =============================================================================
set -euo pipefail
: "${VIEWSUITE_ROOT:?VIEWSUITE_ROOT must be exported}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="viewsuite_interactive_view_planning_nograph"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/${EXPERIMENT_NAME}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"

if [ -n "${CLIENT_URL_FILE:-}" ]; then
  cp "${CLIENT_URL_FILE}" "${VIEWSUITE_ROOT}/client_url.txt"
fi
echo "[nograph] render fleet: $(cat "${VIEWSUITE_ROOT}/client_url.txt" 2>/dev/null || echo '(client_url.txt missing!)')"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    +iteration_overrides.iter3.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train.yaml" \
    +iteration_overrides.iter3.rl.hydra_overrides.trainer.log_image.enable=false \
    "$@" 2>&1 | tee "${LOG_FILE}"
