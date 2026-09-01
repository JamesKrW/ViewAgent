#!/usr/bin/env bash
# Unified install script for the merged ViewSuite monorepo.
#
# Usage:
#   conda create -n viewsuite python=3.12 -y && conda activate viewsuite
#   bash scripts/install.sh
#
# Environment knobs:
#   SKIP_SLIME_BOOTSTRAP=1  skip the VAGEN-SLIME CUDA/runtime build
#   SLIME_ENV_PREFIX=/path  install into this conda environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-${REPO_ROOT}}"

GRAPHRL_DIR="${REPO_ROOT}/GraphRL"
VAGEN_SLIME_DIR="${GRAPHRL_DIR}/VAGEN-SLIME"
LF_DIR="${GRAPHRL_DIR}/LLaMA-Factory"
SLIME_ENV_PREFIX="${SLIME_ENV_PREFIX:-${CONDA_PREFIX:-${REPO_ROOT}/../conda_envs/slime}}"

cat > "${REPO_ROOT}/.env" <<EOF
export VIEWSUITE_ROOT="${VIEWSUITE_ROOT}"
EOF

if [ "${SKIP_SLIME_BOOTSTRAP:-${SKIP_VERL_BOOTSTRAP:-0}}" != "1" ]; then
    ENV_PREFIX="${SLIME_ENV_PREFIX}" \
        bash "${VAGEN_SLIME_DIR}/scripts/build_slime_env.sh"
fi

SLIME_PYTHON="${SLIME_PYTHON:-${SLIME_ENV_PREFIX}/bin/python}"
if [ ! -x "${SLIME_PYTHON}" ]; then
    echo "SLIME Python not found at ${SLIME_PYTHON}" >&2
    echo "Run without SKIP_SLIME_BOOTSTRAP, activate the target conda env, or set SLIME_PYTHON." >&2
    exit 1
fi

"${SLIME_PYTHON}" -m pip install --no-deps -e "${VAGEN_SLIME_DIR}/slime"

"${SLIME_PYTHON}" -m pip install -e "${LF_DIR}"
"${SLIME_PYTHON}" -m pip install -r "${LF_DIR}/requirements/metrics.txt"
"${SLIME_PYTHON}" -m pip install -r "${LF_DIR}/requirements/deepspeed.txt"

"${SLIME_PYTHON}" -m pip install -e "${GRAPHRL_DIR}"
"${SLIME_PYTHON}" -m pip install -e "${REPO_ROOT}"

"${SLIME_PYTHON}" -m pip install transformers==4.57.1

echo ""
echo "============================================================"
echo "Done. Quick sanity check:"
echo "============================================================"
PYTHONPATH="${REPO_ROOT}:${GRAPHRL_DIR}:${VAGEN_SLIME_DIR}:${VAGEN_SLIME_DIR}/slime${PYTHONPATH:+:${PYTHONPATH}}" \
    "${SLIME_PYTHON}" - <<'PY'
import importlib, sys
mods = ["view_suite", "graphrl", "vagen_agent", "slime", "llamafactory", "transformers", "sglang"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "n/a")
        print(f"  OK  {m:<14} {ver}")
    except Exception as e:
        print(f"  FAIL {m:<14} {type(e).__name__}: {e}")
        sys.exit(1)
PY
