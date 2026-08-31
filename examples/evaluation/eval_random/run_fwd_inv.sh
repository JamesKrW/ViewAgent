#!/usr/bin/env bash
set -euo pipefail

# Random baseline evaluation for forward & inverse dynamics.
# No external server needed — responses are sampled locally.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_slime_env.sh"
CONFIG="${CONFIG:-$SCRIPT_DIR/config_fwd_inv.yaml}"
if [[ $# -gt 0 && "$1" != -* && "$1" == *.yaml ]]; then
  CONFIG="$1"
  shift
fi

"${SLIME_PYTHON}" -m view_suite.evaluation.run_eval --config "$CONFIG" "$@" \
  2>&1 | tee "${SCRIPT_DIR}/run_fwd_inv.log"
