#!/usr/bin/env bash
# =============================================================================
# Supervision-signal ablation (exp 01) — condition B: MAIN ONLY
#   generators  = multi_turn_action_gen           (main task)
#   sft dataset = multi_turn_action_gen
# Everything else identical to the paper run (run.sh). The only variable is the
# set of view-graph-distilled supervision signals.
#
# Usage:
#   bash run_ablation_main_only.sh
#   # small dev smoke (3 iters, 10 RL steps each, 1 SFT epoch):
#   bash run_ablation_main_only.sh iterations=3 \
#     iteration_overrides.iter0.rl.training_steps=10 \
#     iteration_overrides.iter1.rl.training_steps=10 \
#     iteration_overrides.iter2.rl.training_steps=10 \
#     general_overrides.sft.hydra_overrides.num_train_epochs=1
#
# Render fleet: RL rollouts render via ${VIEWSUITE_ROOT}/client_url.txt.
#   Point at a specific fleet by exporting CLIENT_URL_FILE=/path/to/client_url_N.txt
#   (its contents are copied into client_url.txt at launch).
# =============================================================================
set -euo pipefail
: "${VIEWSUITE_ROOT:?VIEWSUITE_ROOT must be exported}"

COND=main_only
GENERATORS="multi_turn_action_gen"
DATASET="multi_turn_action_gen"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_NAME="abl_supervision_${COND}"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/${EXPERIMENT_NAME}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"

# Optional: select which render fleet the RL rollouts hit.
if [ -n "${CLIENT_URL_FILE:-}" ]; then
  cp "${CLIENT_URL_FILE}" "${VIEWSUITE_ROOT}/client_url.txt"
fi
echo "[${COND}] render fleet: $(cat "${VIEWSUITE_ROOT}/client_url.txt" 2>/dev/null || echo '(client_url.txt missing!)')"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"
echo "Ablation ${COND}: generators=[${GENERATORS}] dataset=${DATASET}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    experiment_name="${EXPERIMENT_NAME}" \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    "general_overrides.traj_to_sft.generators=[${GENERATORS}]" \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    +iteration_overrides.iter3.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train.yaml" \
    +iteration_overrides.iter3.rl.hydra_overrides.trainer.log_image.enable=false \
    "$@" 2>&1 | tee "${LOG_FILE}"
