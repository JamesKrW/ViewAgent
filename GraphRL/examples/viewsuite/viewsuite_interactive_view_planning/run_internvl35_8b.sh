#!/usr/bin/env bash
# =============================================================================
# Run GraphRL pipeline for ViewSuite Interactive View Planning (InternVL3.5-8B)
# =============================================================================
# Mirrors run_qwen3vl8b.sh exactly, except:
#   - Uses pipeline_internvl35_8b.yaml
#     (OpenGVLab/InternVL3_5-8B-HF + intern_vl SFT template)
#
# Requirements (installed by scripts/install.sh):
#   transformers==4.57.1      (>=4.52.1 needed for InternVL3.5-HF)
#   sglang[all]==0.5.3.post3  (must serve InternVL3.5-HF for rollout)
#   verl 0.6.1 + InternVL backport (PR #6578, adapted for sglang rollout)
#
# The render service (client_url.txt) is reached via with-proxy on this box,
# so launch the whole thing under `with-proxy`:
#   with-proxy bash run_internvl35_8b.sh
#
# Usage:
#   with-proxy bash run_internvl35_8b.sh
#   with-proxy bash run_internvl35_8b.sh iterations=5
#
# experiment_dir is computed by pipeline_internvl35_8b.yaml as
#   exps/viewsuite/viewsuite_interactive_view_planning_internvl35_8b/
# resolved relative to the current working directory.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/viewsuite_interactive_view_planning_internvl35_8b"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

# Meta W&B creds from repo .env (WANDB_API_KEY / WANDB_BASE_URL=https://meta.wandb.io / WANDB_ENTITY).
# meta.wandb.io is internal -> keep it off the forward proxy so it isn't 403'd under with-proxy.
[ -f "${REPO_ROOT}/.env" ] && { set -a; . "${REPO_ROOT}/.env"; set +a; }
export no_proxy="${no_proxy:-},meta.wandb.io,.wandb.io"; export NO_PROXY="${no_proxy}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline_internvl35_8b \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=4 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node=8 \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.sft.n_gpus=8 \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    iteration_overrides.iter0.rl.training_steps=61 \
    iteration_overrides.iter1.rl.training_steps=61 \
    iteration_overrides.iter2.rl.training_steps=61 \
    +iteration_overrides.iter3.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train.yaml" \
    +iteration_overrides.iter3.rl.hydra_overrides.huggingface_hub.repo_id=viewsuite_interactive_view_planning_internvl35_8b \
    +iteration_overrides.iter3.rl.hydra_overrides.trainer.log_image.enable=false \
    "$@" 2>&1 | tee "${LOG_FILE}"
