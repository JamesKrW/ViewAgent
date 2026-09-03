#!/bin/bash
# Convenience wrapper: scannet_http_service_loop.sh with BACKEND defaulting to gsplat.
MAX_WORKERS=${1:-32}
GPU_IDS=${2:-""}
OMP_CAP=${3:-1}
PORT=${4:-8767}
T=${5:-10800}            # restart interval in seconds (default: 3h)
BACKEND=${6:-gsplat}     # open3d (mesh) or gsplat (3DGS)

SCRIPT_DIR="$(dirname "$0")"
exec "${SCRIPT_DIR}/scannet_http_service_loop.sh" \
    "$MAX_WORKERS" "$GPU_IDS" "$OMP_CAP" "$PORT" "$T" "$BACKEND"
