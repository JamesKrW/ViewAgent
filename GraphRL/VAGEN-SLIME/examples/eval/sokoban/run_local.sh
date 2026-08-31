#!/usr/bin/env bash
set -euo pipefail

backend="${1:-sglang}"
model_size="${2:-4b}"
scope="${3:-smoke}"
case "${backend}" in
  sglang|vllm) ;;
  *) echo "usage: $0 {sglang|vllm} {4b|9b} {smoke|full}" >&2; exit 2 ;;
esac
case "${model_size}" in
  4b|9b) ;;
  *) echo "usage: $0 {sglang|vllm} {4b|9b} {smoke|full}" >&2; exit 2 ;;
esac
case "${scope}" in
  smoke) eval_args=(--seed 10001) ;;
  full) eval_args=() ;;
  *) echo "usage: $0 {sglang|vllm} {4b|9b} {smoke|full}" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
config="${script_dir}/qwen35_${backend}.yaml"
eval_python="${VAGEN_EVAL_PYTHON:-python}"
export VAGEN_MODEL_PORT="${VAGEN_MODEL_PORT:-18080}"
export VAGEN_MODEL_BASE_URL="http://127.0.0.1:${VAGEN_MODEL_PORT}/v1"
export VAGEN_MODEL_API_KEY="${VAGEN_MODEL_API_KEY:-EMPTY}"
export VAGEN_SERVED_MODEL="qwen3.5-${model_size}"
export VAGEN_MODEL_RUN_NAME="qwen35-${model_size}-${backend}"
export VAGEN_EXPERIMENT_ID="${VAGEN_EXPERIMENT_ID:-sokoban-qwen35-${model_size}-${backend}-${scope}-v1}"

log_dir="${VAGEN_EVAL_LOG_DIR:-${repo_root}/runs/eval/_services/sokoban_${backend}_${model_size}}"
mkdir -p "${log_dir}"
server_pid=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${eval_python}" - "${VAGEN_MODEL_PORT}" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.settimeout(0.5)
    if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        raise SystemExit(f"model port {sys.argv[1]} is already in use")
PY

"${script_dir}/serve_${backend}.sh" "${model_size}" >"${log_dir}/model.log" 2>&1 &
server_pid="$!"
for _ in $(seq 1 180); do
  if curl --fail --silent --max-time 2 "${VAGEN_MODEL_BASE_URL}/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl --fail --silent --max-time 2 "${VAGEN_MODEL_BASE_URL}/models" >/dev/null; then
  echo "model server did not become ready; tail of ${log_dir}/model.log:" >&2
  tail -n 100 "${log_dir}/model.log" >&2 || true
  exit 1
fi

cd "${repo_root}"
PYTHONPATH="${repo_root}:${PYTHONPATH:-}" \
  "${eval_python}" -m vagen_agent.evaluation --config "${config}" "${eval_args[@]}"
