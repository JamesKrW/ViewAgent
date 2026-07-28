#!/usr/bin/env bash
# Reasoning-effort sweep launcher.
#
# Runs the 4 models in parallel for each mode, modes sequentially (nothink then
# high) to bound concurrent load on OpenRouter and the shared render server.
# "medium" is NOT run — it reuses the published baseline (see
# analyze_reasoning_sweep.py). All traffic goes through `with-proxy` because
# dev-machine egress to the render server IPs is only reachable via fwdproxy.
#
# Designed to be launched under `systemd-run --user` so it survives Claude Code
# session teardown, e.g.:
#   systemd-run --user --unit=reasoning_sweep --same-dir \
#     bash examples/evaluation/eval_reasoning_sweep/run_sweep.sh
#
# Env: expects OPENROUTER_API_KEY + VIEWSUITE_ROOT (sourced from .env.reasoning).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# --- environment -----------------------------------------------------------
source /home/kangrui/miniconda3/etc/profile.d/conda.sh
conda activate viewagent_reason
# shellcheck disable=SC1091
source "${REPO_ROOT}/.env.reasoning"
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-$REPO_ROOT}"
cd "${REPO_ROOT}"

# Which modes to run (space-separated). Override with: MODES_STR="nothink" ...
IFS=' ' read -r -a MODES <<< "${MODES_STR:-nothink high}"
MODELS=(gpt_5_4 gemini_3_1_pro claude_opus_4_6 grok_4_20_beta)

run_mode() {
  local mode="$1"
  echo "==== MODE ${mode} :: launching ${#MODELS[@]} models in parallel ===="
  local pids=() ; declare -A pid_name
  for model in "${MODELS[@]}"; do
    local cfg="${SCRIPT_DIR}/${mode}_${model}.yaml"
    local log="${SCRIPT_DIR}/log_${mode}_${model}.log"
    echo "  launch ${mode}/${model}  ->  ${log}"
    setsid with-proxy python -m vagen.evaluate.run_eval \
      --config "${cfg}" fileroot="${VIEWSUITE_ROOT}" \
      > "${log}" 2>&1 &
    pids+=("$!"); pid_name[$!]="${mode}/${model}"
  done
  local fail=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then echo "  [done] ${pid_name[$pid]}"
    else echo "  [FAIL] ${pid_name[$pid]} (see log)"; fail=1; fi
  done
  return $fail
}

overall_fail=0
for mode in "${MODES[@]}"; do
  run_mode "${mode}" || overall_fail=1
done

echo "==== sweep finished (fail=${overall_fail}) ===="

# --- analysis --------------------------------------------------------------
echo "==== running analysis ===="
python "${SCRIPT_DIR}/analyze_reasoning_sweep.py" \
  --rollouts_root "${VIEWSUITE_ROOT}/rollouts/reasoning_sweep" \
  --data_dir "${VIEWSUITE_ROOT}/data/viewsuite_15k" \
  || echo "analysis step failed"

exit $overall_fail
