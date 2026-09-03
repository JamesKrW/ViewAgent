#!/usr/bin/env bash
# 3DGS checkpoints for the ScanNet scenes referenced by the gs_test manifest.
# Only needed to render ScanNet from splats yourself; the released test split
# (download_scannet_client_3dgs.sh) already ships rendered images.
#
# Requires HF_TOKEN with access to GaussianWorld/scannet_mcmc_1.5M_3dgs.
#
# Produces:
#   data/scannet_3dgs_mcmc/<scene_id>/...
#
# Usage:
#   export HF_TOKEN=hf_...
#   ./download_scannet_service_3dgs.sh                    # manifest + dest defaults
#   ./download_scannet_service_3dgs.sh <MANIFEST> <GS_ROOT>
set -euo pipefail
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"
: "${HF_TOKEN:?HF_TOKEN is not set. export HF_TOKEN=hf_...}"

# Defaults live in the Python (derived from VIEWSUITE_ROOT); only pass overrides.
ARGS=()
[ $# -ge 1 ] && ARGS+=(--manifest "$1")
[ $# -ge 2 ] && ARGS+=(--gs_root "$2")

exec python "$(dirname "${BASH_SOURCE[0]}")/download_scannet_3dgs.py" "${ARGS[@]}"
