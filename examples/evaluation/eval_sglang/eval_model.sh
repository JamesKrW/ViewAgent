#!/usr/bin/env bash
# =============================================================================
# Generic sglang-based eval launcher.
#
# Boots an sglang server for ${MODEL_PATH} on 127.0.0.1:${PORT},
# waits until it answers /v1/models, then runs the ViewSuite eval harness
# against it. The server is torn down on exit.
#
# All knobs are env-vars with defaults — see README.md in this directory.
# =============================================================================
set -euo pipefail

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_slime_env.sh"
fileroot="${fileroot:-${VIEWSUITE_ROOT}}"
CONFIG="${CONFIG:-"$SCRIPT_DIR/config.yaml"}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-"$SCRIPT_DIR/logs"}"

VALIDATE_ONLY=0
for argument in "$@"; do
  if [[ "${argument}" == "--validate-only" ]]; then
    VALIDATE_ONLY=1
    break
  fi
done

# ---------- Model / server config ----------
if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
else
  MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set (HF repo id or local checkpoint path)}"
fi
MODEL_NAME="${MODEL_NAME:-"$(basename "${MODEL_PATH}")"}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.80}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-""}"
read -r -a SGLANG_EXTRA_ARGV <<< "${SGLANG_EXTRA_ARGS}"

DUMP_DIR="${DUMP_DIR:-"$fileroot/rollouts/${MODEL_NAME}"}"

SERVER_LOG="${LOG_DIR}/${MODEL_NAME}_server.log"
EVAL_LOG="${LOG_DIR}/${MODEL_NAME}_eval.log"

echo "[eval_model] model_name=${MODEL_NAME}"
echo "[eval_model] model_path=${MODEL_PATH}"
echo "[eval_model] config=${CONFIG}"
echo "[eval_model] dump_dir=${DUMP_DIR}"
echo "[eval_model] port=${PORT}  TP=${TP_SIZE}  DP=${DP_SIZE}  MEM=${MEM_FRACTION}"
[[ -n "${SGLANG_EXTRA_ARGS}" ]] && echo "[eval_model] sglang extra: ${SGLANG_EXTRA_ARGS}"

# Config-only validation must not reserve GPUs or boot an SGLang service.  It
# still exercises the same VAGEN-SLIME evaluator translation and overrides as
# the real launch path.
if [[ "${VALIDATE_ONLY}" == "1" ]]; then
  exec "${SLIME_PYTHON}" -m view_suite.evaluation.run_eval --config "${CONFIG}" \
    run.backend=sglang \
    backends.sglang.base_url="http://127.0.0.1:${PORT}/v1" \
    backends.sglang.model="${MODEL_PATH}" \
    experiment.dump_dir="${DUMP_DIR}" \
    fileroot="${fileroot}" \
    "$@"
fi

mkdir -p "$LOG_DIR" "$DUMP_DIR"

# ---------- Launch server ----------
"${SLIME_PYTHON}" -m sglang.launch_server \
  --host 0.0.0.0 \
  --log-level warning \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --dp-size "${DP_SIZE}" \
  --tp "${TP_SIZE}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  "${SGLANG_EXTRA_ARGV[@]}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---------- Wait for server ----------
source "${SCRIPT_DIR}/wait_for_server.sh"
wait_for_server

# ---------- Run eval ----------
"${SLIME_PYTHON}" -m view_suite.evaluation.run_eval --config "${CONFIG}" \
  run.backend=sglang \
  backends.sglang.base_url="http://127.0.0.1:${PORT}/v1" \
  backends.sglang.model="${MODEL_PATH}" \
  experiment.dump_dir="${DUMP_DIR}" \
  fileroot="${fileroot}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"
