#!/usr/bin/env bash
# Habitat-GS proxy-task data (P2V / V2P / IVP) on 3D Gaussian-Splatting scenes.
#
# Produces:
#   data/viewagent_habitat_gs/
#     {path_to_view,view_to_path,interactive_view_planning}_{train,eval,test}.jsonl
#     {interior_*,sceneNN*}/...   (rendered views + top_down.png)
#
# P2V/V2P read these images straight from the jsonl. Only IVP additionally needs
# the render service -- see download_habitat_gs_service_data.sh.
: "${VIEWSUITE_ROOT:?set up VIEWSUITE_ROOT first (default: your repo dir), e.g. export VIEWSUITE_ROOT=/path/to/ViewSuite}"

python -m view_suite.utils.download_targz_hf \
    --repo=MLL-Lab/viewsuite \
    --files="viewagent_habitat_gs_full.tar.gz" \
    --out="$VIEWSUITE_ROOT/data"
