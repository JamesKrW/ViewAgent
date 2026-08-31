#!/usr/bin/env bash
# Launch all proxy-model IVP no_submit evals in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_slime_env.sh"
fileroot="${fileroot:-${VIEWSUITE_ROOT}}"

MODELS=(
  gemini_3_1_pro
  gpt_5_4
)

pids=()
trap 'echo "Interrupted, killing all child processes..."; kill "${pids[@]}" 2>/dev/null; exit 1' INT TERM

for model in "${MODELS[@]}"; do
  echo "Launching: ${model}"
  "${SLIME_PYTHON}" -m view_suite.evaluation.run_eval \
    --config "${SCRIPT_DIR}/${model}.yaml" \
    fileroot="${fileroot}" \
    "$@" \
    > "${SCRIPT_DIR}/log_${model}.log" 2>&1 &
  pids+=($!)
done

echo "All ${#MODELS[@]} jobs launched in parallel. Waiting..."

fail=0
for i in "${!MODELS[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[PASS] ${MODELS[$i]}"
  else
    echo "[FAIL] ${MODELS[$i]} (see log_${MODELS[$i]}.log)"
    fail=1
  fi
done

if [ $fail -eq 0 ]; then
  echo "All evaluations complete."
else
  echo "Some evaluations failed."
  exit 1
fi
