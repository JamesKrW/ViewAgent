#!/usr/bin/env bash
set -euo pipefail

model_size="${1:-4b}"
model_root="${VAGEN_MODEL_ROOT:-${HOME}/models}"
case "${model_size}" in
  4b)
    model_path="${QWEN35_4B_PATH:-${model_root}/Qwen3.5-4B}"
    served_model="qwen3.5-4b"
    ;;
  9b)
    model_path="${QWEN35_9B_PATH:-${model_root}/Qwen3.5-9B}"
    served_model="qwen3.5-9b"
    ;;
  *)
    echo "usage: $0 {4b|9b}" >&2
    exit 2
    ;;
esac

python_bin="${SGLANG_PYTHON:-python}"
env_root="$(dirname "$(dirname "${python_bin}")")"
export PATH="$(dirname "${python_bin}"):${PATH}"
export LD_LIBRARY_PATH="${env_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${VAGEN_MODEL_GPU:-7}"

exec "${python_bin}" -m sglang.launch_server \
  --model-path "${model_path}" \
  --served-model-name "${served_model}" \
  --host 127.0.0.1 \
  --port "${VAGEN_MODEL_PORT:-18080}" \
  --trust-remote-code \
  --enable-multimodal \
  --reasoning-parser qwen3 \
  --enable-strict-thinking \
  --mem-fraction-static "${VAGEN_MODEL_MEMORY_FRACTION:-0.7}" \
  --mm-attention-backend sdpa \
  --disable-flashinfer-autotune
