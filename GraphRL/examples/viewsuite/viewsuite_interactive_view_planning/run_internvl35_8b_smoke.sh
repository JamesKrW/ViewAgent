#!/usr/bin/env bash
# =============================================================================
# SMOKE TEST — GraphRL pipeline for InternVL3.5-8B (small scale)
# =============================================================================
# Purpose: shake out the full RL -> TrajToSFT -> SFT loop end-to-end on
# InternVL3.5-8B without committing to a long run. Small everything:
#   - 3 iterations (iter0, iter1, iter2)
#   - 10 RL training steps per iteration
#   - small RL batch (train_batch_size / mini_batch = 16)
#   - 1 SFT epoch per iteration
#   - RL checkpoint saved at step 10 (so the pipeline has an rl_model to hand off)
#   - validation disabled (test_freq huge) to keep it fast
#
# Reach the render service via the Meta forward proxy:
#   with-proxy bash run_internvl35_8b_smoke.sh
#
# Override anything on the CLI, e.g.:
#   with-proxy bash run_internvl35_8b_smoke.sh iterations=1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
EXPERIMENT_DIR="${PWD}/exps/viewsuite/viewsuite_interactive_view_planning_internvl35_8b_smoke"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SFT_N_GPUS="${SFT_N_GPUS:-${N_GPUS_PER_NODE}}"

mkdir -p "${EXPERIMENT_DIR}"
LOG_FILE="${EXPERIMENT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"
echo "SMOKE: 3 iters x 10 RL steps, small batch, 1 SFT epoch, ${N_GPUS_PER_NODE} GPU(s)"

# Meta W&B creds from repo .env; keep meta.wandb.io off the forward proxy.
[ -f "${REPO_ROOT}/.env" ] && { set -a; . "${REPO_ROOT}/.env"; set +a; }
export no_proxy="${no_proxy:-},meta.wandb.io,.wandb.io"; export NO_PROXY="${no_proxy}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

python -m graphrl.main \
    --config-path="${SCRIPT_DIR}" \
    --config-name=pipeline_internvl35_8b \
    experiment_name=viewsuite_interactive_view_planning_internvl35_8b_smoke \
    general_overrides.rl.hydra_overrides.data.train_files="${SCRIPT_DIR}/train_turn_format.yaml" \
    general_overrides.rl.hydra_overrides.data.val_files="${SCRIPT_DIR}/val.yaml" \
    iterations=3 \
    general_overrides.rl.training_steps=10 \
    general_overrides.rl.hydra_overrides.data.train_batch_size=16 \
    general_overrides.rl.hydra_overrides.actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    general_overrides.rl.hydra_overrides.trainer.nnodes=1 \
    general_overrides.rl.hydra_overrides.trainer.save_freq=10 \
    general_overrides.rl.hydra_overrides.trainer.test_freq=100000 \
    +general_overrides.rl.hydra_overrides.trainer.log_image.enable=false \
    general_overrides.sft.n_gpus="${SFT_N_GPUS}" \
    general_overrides.sft.hydra_overrides.num_train_epochs=1.0 \
    general_overrides.sft.hydra_overrides.val_size=0.1 \
    'general_overrides.traj_to_sft.generators=[multi_turn_action_gen,view_difference,view_difference_mcq]' \
    iteration_overrides.iter0.rl.training_steps=10 \
    iteration_overrides.iter1.rl.training_steps=10 \
    iteration_overrides.iter2.rl.training_steps=10 \
    'iteration_overrides.iter0.sft.hydra_overrides.num_train_epochs=1.0' \
    'iteration_overrides.iter1.sft.hydra_overrides.num_train_epochs=1.0' \
    'iteration_overrides.iter2.sft.hydra_overrides.num_train_epochs=1.0' \
    "$@" 2>&1 | tee "${LOG_FILE}"
