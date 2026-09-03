#!/usr/bin/env bash
# Wait for the GPUs to free, then smoke-test all three ViewSuite IVP tasks on the
# VAGEN/verl backend.
#
# The training box is shared. When this was written all 8 GPUs were held by
# another user's job, so this waits rather than competing: it only starts once
# enough memory is genuinely free, and it never kills anything.
#
# It runs the SHORT smoke pipeline (2 RL steps, RL only, 4 GPUs) for each task --
# enough to prove the stack end to end. Full runs are a separate, deliberate
# decision; see the commands printed at the end.
#
#   bash scripts/await_gpus_and_smoke.sh            # wait, then smoke all three
#   MIN_FREE_MIB=100000 bash scripts/await_gpus_and_smoke.sh
#   MAX_WAIT_HOURS=0 bash scripts/await_gpus_and_smoke.sh   # run now, no waiting

set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${GPUS:-0,2,3,4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-100000}"   # per GPU, of ~143 GiB
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-24}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${LOG_DIR:-$REPO/exps/_vagen_smoke}"

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/watcher_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "== watcher started $(date '+%F %T')"
echo "   waiting for GPUs [$GPUS] to each have >= ${MIN_FREE_MIB} MiB free"
echo "   log: $MAIN_LOG"

free_enough() {
    local idx free
    for idx in ${GPUS//,/ }; do
        free=$(nvidia-smi --id="$idx" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
        [ -n "$free" ] || return 1
        [ "$free" -ge "$MIN_FREE_MIB" ] || return 1
    done
    return 0
}

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
until free_enough; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "== gave up waiting after ${MAX_WAIT_HOURS}h; GPUs still busy. Nothing was run."
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
        exit 2
    fi
    echo "   $(date '+%T') still busy: $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tr '\n' ' ')"
    sleep "$POLL_SECONDS"
done

echo "== GPUs free at $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

# Self-signed render certificates; see view_suite/service_http/async_client.py.
export RENDER_TLS_NO_VERIFY="${RENDER_TLS_NO_VERIFY:-1}"
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-$REPO}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export N_GPUS_PER_NODE=4

# The render services must answer before a rollout can produce anything. A task
# whose service is down would otherwise fail deep inside the trainer.
for probe in habitat_gs:8812 scannet:8813 ai2thor:8814; do
    name=${probe%%:*}; port=${probe##*:}
    code=$(curl -sk -o /dev/null -w "%{http_code}" --max-time 10 \
           "https://$(hostname -f):$port/health" 2>/dev/null)
    echo "   render service $name (:$port) -> ${code:-unreachable}"
done

status=0
for task in habitat_gs_interactive_view_planning \
            ai2thor_interactive_view_planning \
            viewsuite_interactive_view_planning; do
    smoke="$REPO/GraphRL/examples/viewsuite/$task/run_smoke.sh"
    if [ ! -f "$smoke" ]; then
        # Only habitat_gs and ai2thor ship one; drive the others through run.sh
        # with the same short settings rather than skipping the task.
        smoke=""
    fi
    echo
    echo "══════════════════════════════════════════════════════════════"
    echo "== $task  ($(date '+%T'))"
    echo "══════════════════════════════════════════════════════════════"
    cd "$REPO/GraphRL" || exit 1
    if [ -n "$smoke" ]; then
        bash "$smoke"
    else
        d="$REPO/GraphRL/examples/viewsuite/$task"
        # shellcheck disable=SC1091
        source "$REPO/GraphRL/examples/_vagen_env.sh"
        "${VAGEN_PYTHON}" -m graphrl.main \
            --config-path="$d" --config-name=pipeline \
            experiment_name="${task}_smoke" \
            general_overrides.rl.hydra_overrides.data.train_files="$d/train.yaml" \
            general_overrides.rl.hydra_overrides.data.val_files="$d/val.yaml" \
            iterations=1 \
            general_overrides.rl.hydra_overrides.trainer.n_gpus_per_node=4 \
            general_overrides.rl.hydra_overrides.data.train_batch_size=8 \
            iteration_overrides.iter0.rl.training_steps=2 \
            +iteration_overrides.iter0.traj_to_sft=null \
            iteration_overrides.iter0.sft=null
    fi
    rc=$?
    echo "== $task exit=$rc"
    [ $rc -eq 0 ] || status=1
done

echo
echo "== all smokes finished $(date '+%F %T'), overall=$status"
if [ $status -eq 0 ]; then
cat <<'NEXT'

All three tasks ran. Full runs (8 GPUs, several days each) are a separate call:

  cd ViewAgent/GraphRL
  RENDER_TLS_NO_VERIFY=1 bash examples/viewsuite/habitat_gs_interactive_view_planning/run.sh
  RENDER_TLS_NO_VERIFY=1 bash examples/viewsuite/ai2thor_interactive_view_planning/run.sh
  RENDER_TLS_NO_VERIFY=1 bash examples/viewsuite/viewsuite_interactive_view_planning/run.sh
NEXT
fi
exit $status
