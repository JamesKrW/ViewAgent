#!/usr/bin/env bash
# ScanNet scene meshes for the render service. Needed only by IVP, which renders
# every turn; P2V/V2P read pre-rendered images and need none of this.
#
# Produces:
#   data/viewagent_scannet/scans/<scene_id>/...   (~32 GB)
#
# Serve it with:
#   scripts/scannet/scannet_http_service_loop.sh 110 0,1,2,3,4,5,6,7 1 8813 86400 habitat
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=MLL-Lab/viewsuite \
    --files="viewagent_scannet.tar.gz" \
    --out="$VIEWSUITE_ROOT/data"
