#!/usr/bin/env bash
# export OPENAI_API_KEY=sk-xxxxxxxx
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_slime_env.sh"

fileroot="${fileroot:-${VIEWSUITE_ROOT}}"

LOG_FILE="${SCRIPT_DIR}/run.log"

"${SLIME_PYTHON}" -m view_suite.evaluation.run_eval \
  --config "${SCRIPT_DIR}/evaluate_gpt4o.yaml" \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
