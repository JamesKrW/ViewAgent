#!/usr/bin/env bash
# export AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_slime_env.sh"

fileroot="${fileroot:-${VIEWSUITE_ROOT}}"

LOG_FILE="${SCRIPT_DIR}/run.log"

"${SLIME_PYTHON}" -m view_suite.evaluation.run_eval \
  --config "${SCRIPT_DIR}/config.yaml" \
  run.backend=azure \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
