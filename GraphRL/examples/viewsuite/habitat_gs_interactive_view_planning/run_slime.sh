#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWAGENT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SLIME_ENV="${SLIME_ENV:-${VIEWAGENT_ROOT}/GraphRL/conda_envs/slime}"

export PATH="${SLIME_ENV}/bin:${PATH}"
export CONDA_PREFIX="${SLIME_ENV}"
export CUDA_HOME="${SLIME_ENV}"
SLIME_CUDNN_LIB="${SLIME_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${SLIME_CUDNN_LIB}:${SLIME_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-${VIEWAGENT_ROOT}}"
export PYTHONPATH="${VIEWAGENT_ROOT}:${VIEWAGENT_ROOT}/GraphRL:${VIEWAGENT_ROOT}/GraphRL/VAGEN:${VIEWAGENT_ROOT}/GraphRL/VAGEN/slime${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec "${SLIME_ENV}/bin/python" \
  "${SCRIPT_DIR}/slime/run_habitat_gs.py" \
  "$@"
