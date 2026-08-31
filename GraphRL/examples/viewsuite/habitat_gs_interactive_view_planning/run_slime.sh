#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_slime_env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"

exec "${SLIME_PYTHON}" \
  "${SCRIPT_DIR}/slime/run_habitat_gs.py" \
  "$@"
