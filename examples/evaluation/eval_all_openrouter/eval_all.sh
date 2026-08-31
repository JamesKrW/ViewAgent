#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_slime_env.sh"
fileroot="${fileroot:-${VIEWSUITE_ROOT}}"

MODELS=(
  claude_opus_4_6
  gemini_3_1_pro
  glm_4_6v
  #gpt_5_4_pro
  grok_4_20_beta
  qwen2_5_vl_72b
  qwen3_5_397b
  qwen3_vl_32b
  gpt_5_4
  gpt_5_1
  gemini_3_pro
  kimi_k2_5
)

pgids=()
# Map pgid -> model name for pretty pass/fail reporting.
declare -A pgid_model

cleanup() {
  trap '' INT TERM EXIT
  if ((${#pgids[@]} == 0)); then return; fi
  echo "[cleanup] terminating ${#pgids[@]} job group(s)..."
  for pg in "${pgids[@]}"; do
    kill -TERM "-$pg" 2>/dev/null || true
  done
  sleep 5
  for pg in "${pgids[@]}"; do
    if kill -0 "-$pg" 2>/dev/null; then
      echo "[cleanup] SIGKILL pgid=$pg (${pgid_model[$pg]:-?})"
      kill -KILL "-$pg" 2>/dev/null || true
    fi
  done
}
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

for model in "${MODELS[@]}"; do
  echo "Launching: ${model}"
  # setsid puts the whole job in its own process group so grandchildren
  # (http workers, torch subprocs, etc.) get killed together via kill -PGID.
  setsid "${SLIME_PYTHON}" -m view_suite.evaluation.run_eval \
    --config "${SCRIPT_DIR}/${model}.yaml" \
    fileroot="${fileroot}" \
    "$@" \
    > "${SCRIPT_DIR}/log_${model}.log" 2>&1 &
  child=$!
  pgids+=("$child")
  pgid_model[$child]="$model"
done

echo "All ${#MODELS[@]} jobs launched in parallel. Waiting..."

fail=0
pending=("${pgids[@]}")
while ((${#pending[@]})); do
  # wait -n reaps any single finished job and returns its exit status.
  if wait -n; then :; else fail=1; fi
  new=()
  for pg in "${pending[@]}"; do
    if kill -0 "$pg" 2>/dev/null; then
      new+=("$pg")
    else
      echo "[done]  ${pgid_model[$pg]:-pgid=$pg}"
    fi
  done
  pending=("${new[@]}")
done

if [ $fail -eq 0 ]; then
  echo "All evaluations complete."
else
  echo "Some evaluations failed (see log_*.log)."
  exit 1
fi
