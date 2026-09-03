#!/usr/bin/env bash
# Pre-fetch the AI2-THOR scene assets so the first render does not pay for the
# download. AI2-THOR ships its own CloudRendering build and caches it under
# ~/.ai2thor, so there is no tarball to unpack here.
#
# Usage:
#   ./download_ai2thor_service_data.sh            # all scenes, auto GPU
#   ./download_ai2thor_service_data.sh FloorPlan1 # one scene
#   ./download_ai2thor_service_data.sh all 0      # all scenes, GPU 0
set -euo pipefail
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

SCENES=${1:-all}
GPU=${2:-}

ARGS=(--scenes="$SCENES")
[ -n "$GPU" ] && ARGS+=(--gpu="$GPU")

exec python -m view_suite.ai2thor.pre_download_scenes "${ARGS[@]}"
