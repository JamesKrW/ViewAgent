#!/usr/bin/env bash
# Habitat-GS splat scenes for the render service (~30 GB, public, no token).
# Served in place rather than unpacked, so this is a repo snapshot, not a tarball.
#
# Produces:
#   data/gs_scenes/{train,val}/...  + *.scene_dataset_config.json
#
# Serve it with:
#   scripts/scannet/scannet_http_service_loop.sh 110 0,1,2,3,4,5,6,7 1 8812 86400 habitat_gs
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_snapshot_hf \
    --repo=RukawaY/gs_scenes \
    --allow="train/**,val/**,*.scene_dataset_config.json,README.md" \
    --out="${HABITAT_GS_ROOT:-$VIEWSUITE_ROOT/data/gs_scenes}"
