#!/usr/bin/env bash
# AI2-THOR proxy-task data (P2V / V2P / IVP).
#
# Produces:
#   data/viewagent15k_ai2thor/
#     {path_to_view,view_to_path,interactive_view_planning}_{train,eval,test}.jsonl
#     FloorPlan*/...   (rendered init/option/target/top-down views)
#
# P2V/V2P read these images straight from the jsonl. Only IVP additionally needs
# the simulator -- see download_ai2thor_service_data.sh.
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=MLL-Lab/viewsuite \
    --files="viewagent15k_ai2thor.tar.gz" \
    --out="$VIEWSUITE_ROOT/data"
