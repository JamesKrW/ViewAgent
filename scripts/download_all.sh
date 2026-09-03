#!/usr/bin/env bash
# Run every download script for one corpus, in order.
#
# Usage:
#   ./download_all.sh scannet
#   ./download_all.sh ai2thor
#   ./download_all.sh habitat_gs
#   ./download_all.sh mindcube
#
# Sizes are large and this makes no attempt to be clever about it. In particular
# `scannet` pulls all three client renderings (Open3D, Habitat, 3DGS) plus the
# ~32 GB scene meshes; if you only need one renderer, run that script directly
# from scripts/scannet/ instead.
#
# download_scannet_service_3dgs.sh additionally needs HF_TOKEN and is skipped
# when it is unset, since most users never render ScanNet from splats themselves.
set -euo pipefail
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${1:?usage: download_all.sh <scannet|ai2thor|habitat_gs|mindcube>}"
[ -d "$DIR/$CORPUS" ] || { echo "[error] unknown corpus: $CORPUS" >&2; exit 2; }

for script in "$DIR/$CORPUS"/download_*.sh; do
    if [[ "$script" == *_3dgs.sh && "$script" == *service* && -z "${HF_TOKEN:-}" ]]; then
        echo "== skip $(basename "$script") (needs HF_TOKEN)"
        continue
    fi
    echo "== $(basename "$script")"
    bash "$script"
done
