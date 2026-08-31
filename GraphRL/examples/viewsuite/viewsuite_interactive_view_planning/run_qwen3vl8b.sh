#!/usr/bin/env bash
# =============================================================================
# Run GraphRL pipeline for ViewSuite Interactive View Planning (Qwen3-VL-8B)
# =============================================================================
# Mirrors run_v2.sh exactly, except:
#   - Uses pipeline_qwen3vl8b.yaml (Qwen/Qwen3-VL-8B-Instruct + qwen3_vl_nothink SFT template)
#
# Usage:
#   bash run_qwen3vl8b.sh
#   bash run_qwen3vl8b.sh iterations=5
# pip install transformers==4.57.1
# pip install "sglang[all]==0.5.3.post3"
#
# experiment_dir is computed by pipeline_qwen3vl8b.yaml as
#   exps/viewsuite/viewsuite_interactive_view_planning_qwen3vl8b/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_slime_env.sh"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/viewsuite_interactive_view_planning_qwen3vl8b"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

"${SLIME_PYTHON}" -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline_qwen3vl8b \
    general_overrides.rl.slime.train_envs="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.slime.eval_envs="${SCRIPT_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.slime.num_gpus=8 \
    general_overrides.sft.n_gpus=8 \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    +iteration_overrides.iter3.rl.slime.train_envs="${SCRIPT_DIR}/train.yaml" \
    +iteration_overrides.iter3.rl.slime.record_rollout_images=false \
    "$@" 2>&1 | tee "${LOG_FILE}"
