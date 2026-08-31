#!/bin/bash
# Rollout-only verification across the Qwen families this port claims to support.
#
# No training: each row is slime's own `--debug-rollout-only` (sglang engines, zero
# Megatron actors) over a 32-episode Sokoban set, with `--dump-details` writing every
# Sample to disk. `tests/verify_rollout_dump.py` then reads the dump back and checks that
# the loss mask covers the model's own tokens and nothing else.
#
# The matrix exists because the seam moves with the family. Only Qwen3.5 honours
# `enable_thinking`, and it changes the generation prompt:
#   on  -> ...<|im_start|>assistant\n<think>\n
#   off -> ...<|im_start|>assistant\n<think>\n\n</think>\n\n
# The other three ignore the kwarg entirely.
#
#   bash tests/verify_qwen_matrix.sh [GPUS]      # GPUS defaults to "2,3"
#   TURN_TOKENS=2048 bash tests/verify_qwen_matrix.sh
#
# TURN_TOKENS is the per-turn generation budget. A native-thinking model spends it on
# reasoning before it writes anything parseable, so the same checkpoint can look unable to
# play the game at 512 and competent at 2048.
# No `set -u`: the conda cuda-nvcc activate hook reads NVCC_PREPEND_FLAGS unbound.
GPUS="${1:-2,3}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNS="$REPO/runs"
LOGS="$REPO/logs/runs"
MODELS="$HOME/models"

# exp | model | chat-template-kwargs | env yaml | verifier flag
# Sokoban renders a frame every turn, so every row exercises the multimodal path.
MATRIX=(
  "q25vl|Qwen2.5-VL-3B-Instruct||verify|"
  "q3vl|Qwen3-VL-4B-Instruct||verify|"
  "q35_think|Qwen3.5-4B|{\"enable_thinking\": true}|verify|--expect-thinking"
  "q35_nothink|Qwen3.5-4B|{\"enable_thinking\": false}|verify|--expect-no-thinking"
)

PYTHON_BIN="${SLIME_PYTHON:-python}"
cd "$REPO" || exit 1
mkdir -p "$LOGS"
fail=0

for row in "${MATRIX[@]}"; do
  IFS='|' read -r exp model kwargs envs flag <<< "$row"
  log="$LOGS/verify_$exp.log"
  echo "=== $exp  ($model, ${kwargs:-default template}, $envs)"

  args=(--rollout-only --num-gpus 2 --rollout-tp 1 --exp-name "$exp" --model-name "$model"
        --eval-yaml "examples/train/sokoban/$envs.yaml"
        # sglang's default vision kernel here is fa4, which needs a cutlass wheel this env
        # does not have; sdpa is pure torch. See Config.sglang_mm_attention_backend.
        --sglang-mm-attention-backend "${MM_BACKEND:-sdpa}"
        --turn-tokens "${TURN_TOKENS:-512}")
  [ -n "$kwargs" ] && args+=(--chat-template-kwargs "$kwargs")

  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" examples/train/sokoban/run_sokoban.py "${args[@]}" > "$log" 2>&1
  if [ $? -ne 0 ]; then
    echo "  ROLLOUT FAILED -- see $log"; tail -20 "$log"; fail=$((fail + 1)); continue
  fi

  dump="$RUNS/$exp/dump/rollout_data/eval_0.pt"
  "$PYTHON_BIN" tests/verify_rollout_dump.py "$dump" --model "$MODELS/$model" ${flag:+$flag} \
    || fail=$((fail + 1))
  grep -ao "eval 0: {.*}" "$log" | tail -1
  echo
done

echo "matrix done: $fail failure(s)"
exit $((fail > 0))
