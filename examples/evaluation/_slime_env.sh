#!/usr/bin/env bash
# Shared runtime wiring for standalone ViewSuite evaluation through VAGEN-SLIME.

EVALUATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIEWAGENT_ROOT="$(cd "${EVALUATION_ROOT}/../.." && pwd)"
GRAPHRL_ROOT="${VIEWAGENT_ROOT}/GraphRL"
VAGEN_SLIME_ROOT="${GRAPHRL_ROOT}/VAGEN-SLIME"

export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-${VIEWAGENT_ROOT}}"
export PYTHONPATH="${VIEWAGENT_ROOT}:${GRAPHRL_ROOT}:${VAGEN_SLIME_ROOT}:${VAGEN_SLIME_ROOT}/slime${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${SLIME_PYTHON:-}" ]; then
    LOCAL_SLIME_PYTHON="${VIEWAGENT_ROOT}/../conda_envs/slime/bin/python"
    if [ -x "${LOCAL_SLIME_PYTHON}" ]; then
        SLIME_PYTHON="${LOCAL_SLIME_PYTHON}"
    else
        SLIME_PYTHON="python"
    fi
fi
export SLIME_PYTHON
