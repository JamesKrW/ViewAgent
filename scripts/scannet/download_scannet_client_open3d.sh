#!/usr/bin/env bash
# ScanNet proxy-task data (P2V / V2P / IVP) rendered with the Open3D mesh renderer.
# This is the default ScanNet client dataset.
#
# Produces:
#   data/viewsuite_15k/
#     {path_to_view,view_to_path,interactive_view_planning}_{train,eval,test}.jsonl
#     scene*/...   (rendered init/option/target/top-down views)
#
# P2V/V2P read these images straight from the jsonl. Only IVP additionally needs
# the render service -- see download_scannet_service_data.sh.
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=MLL-Lab/viewsuite \
    --files="viewsuite_15k.tar.gz" \
    --out="$VIEWSUITE_ROOT/data"
