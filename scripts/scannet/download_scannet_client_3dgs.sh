#!/usr/bin/env bash
# ScanNet proxy-task test split re-rendered from 3D Gaussian Splatting
# reconstructions instead of the scanned mesh.
#
# Produces:
#   data/viewagent15k_scannet_gs_test/     (sibling of viewagent15k_scannet_open3d, same layout)
#
# Only the test split is released this way. To render more yourself you need the
# 3DGS checkpoints -- see download_scannet_service_3dgs.sh.
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=MLL-Lab/viewsuite \
    --files="viewagent15k_scannet_gs_test.tar.gz" \
    --out="$VIEWSUITE_ROOT/data"
