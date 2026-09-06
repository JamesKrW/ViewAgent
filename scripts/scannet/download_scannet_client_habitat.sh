#!/usr/bin/env bash
# ScanNet proxy-task data re-rendered through Habitat instead of Open3D.
# Same tasks and splits as the Open3D release; different renderer, so the two are
# a clean A/B on identical task definitions.
#
# Produces:
#   data/viewagent15k_scannet_habitat/     (sibling of viewagent15k_scannet_open3d, same layout)
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=MLL-Lab/viewsuite \
    --files="viewagent15k_scannet_habitat.tar.gz" \
    --out="$VIEWSUITE_ROOT/data"
