#!/usr/bin/env bash
set -euo pipefail

# Random baseline evaluation for active exploration (navigation).
# Requires the rendering server to be running:
#   see client_url.txt for the expected endpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_slime_env.sh"
CONFIG="${CONFIG:-$SCRIPT_DIR/config_navigation.yaml}"
if [[ $# -gt 0 && "$1" != -* && "$1" == *.yaml ]]; then
  CONFIG="$1"
  shift
fi

"${SLIME_PYTHON}" -m view_suite.evaluation.run_eval --config "$CONFIG" "$@" \
  2>&1 | tee "${SCRIPT_DIR}/run_navigation.log"
