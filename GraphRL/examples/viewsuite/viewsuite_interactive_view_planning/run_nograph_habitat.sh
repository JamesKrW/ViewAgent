#!/usr/bin/env bash
# ScanNet-Habitat no-graph, no-atomize ablation.  The compatibility graph
# contains one disconnected chain per rollout, so hindsight paths never compose
# transitions across trajectories.  SFT uses the graph baseline's exact
# per-scene budget: 193 scenes * (20 + 15 + 15) = 9,650 records.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_vagen_env.sh"

TRAIN_CONFIG="${VIEWSUITE_TRAIN_CONFIG:-train_turn_format_512_habitat.yaml}"
VAL_CONFIG="${VIEWSUITE_VAL_CONFIG:-val_512_habitat.yaml}"
FINAL_TRAIN_CONFIG="${VIEWSUITE_FINAL_TRAIN_CONFIG:-train_512_habitat.yaml}"
[[ "${TRAIN_CONFIG}" = /* ]] || TRAIN_CONFIG="${SCRIPT_DIR}/${TRAIN_CONFIG}"
[[ "${VAL_CONFIG}" = /* ]] || VAL_CONFIG="${SCRIPT_DIR}/${VAL_CONFIG}"
[[ "${FINAL_TRAIN_CONFIG}" = /* ]] || FINAL_TRAIN_CONFIG="${SCRIPT_DIR}/${FINAL_TRAIN_CONFIG}"

DATASET_DIR="${VIEWSUITE_DATASET_DIR:-${VIEWSUITE_ROOT}/data/viewsuite_15k_habitat}"
EXPERIMENT_NAME="${EXP_NAME:-viewsuite_scannet_habitat_nograph_budgetmatch_512}"
EXPERIMENT_DIR="${GRAPHRL_EXPERIMENT_DIR:-${PWD}/exps/viewagent/${EXPERIMENT_NAME}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"
mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${VAGEN_PYTHON}" -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline \
    project_name=viewagent \
    experiment_name="${EXPERIMENT_NAME}" \
    experiment_dir="${EXPERIMENT_DIR}" \
    iterations=4 \
    general_overrides.rl.hydra_overrides.data.train_files="${TRAIN_CONFIG}" \
    general_overrides.rl.hydra_overrides.data.val_files="${VAL_CONFIG}" \
    general_overrides.rl.hydra_overrides.algorithm.adv_estimator=default_gae \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    general_overrides.traj_to_sft.module=graphrl.envs.viewsuite.viewsuite_interactive_view_planning_nograph.NoGraphInteractiveViewPlanningTrajToSFT \
    general_overrides.traj_to_sft.viewsuite_15k_dir="${DATASET_DIR}" \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    general_overrides.traj_to_sft.multi_turn_action_gen.sample_per_scene=20 \
    general_overrides.traj_to_sft.view_difference.sample_per_scene=15 \
    general_overrides.traj_to_sft.view_difference_mcq.sample_per_scene=15 \
    +general_overrides.traj_to_sft.expected_records.multi_turn_action_gen=3860 \
    +general_overrides.traj_to_sft.expected_records.view_difference=2895 \
    +general_overrides.traj_to_sft.expected_records.view_difference_mcq=2895 \
    general_overrides.traj_to_sft.graph_builder.atomize.enabled=false \
    general_overrides.traj_to_sft.graph_builder.merge_tol.position=0.2 \
    general_overrides.traj_to_sft.graph_builder.merge_tol.angle=10.0 \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    iteration_overrides.iter3.rl.training_steps=800 \
    +iteration_overrides.iter3.rl.hydra_overrides.data.train_files="${FINAL_TRAIN_CONFIG}" \
    +iteration_overrides.iter3.rl.hydra_overrides.trainer.log_image.enable=false \
    +iteration_overrides.iter3.traj_to_sft=null \
    +iteration_overrides.iter3.sft=null \
    "$@" 2>&1 | tee "${LOG_FILE}"
