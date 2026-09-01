#!/usr/bin/env bash
# Shared runtime wiring for GraphRL's VAGEN-SLIME examples.

EXAMPLES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHRL_ROOT="$(cd "${EXAMPLES_ROOT}/.." && pwd)"
VIEWAGENT_ROOT="$(cd "${GRAPHRL_ROOT}/.." && pwd)"
VAGEN_SLIME_ROOT="${GRAPHRL_ROOT}/VAGEN-SLIME"

export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-${VIEWAGENT_ROOT}}"
export PYTHONPATH="${VIEWAGENT_ROOT}:${GRAPHRL_ROOT}:${VAGEN_SLIME_ROOT}:${VAGEN_SLIME_ROOT}/slime:${GRAPHRL_ROOT}/LLaMA-Factory/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${SLIME_PYTHON:-}" ]; then
    LOCAL_SLIME_PYTHON="${VIEWAGENT_ROOT}/../conda_envs/slime/bin/python"
    if [ -x "${LOCAL_SLIME_PYTHON}" ]; then
        SLIME_PYTHON="${LOCAL_SLIME_PYTHON}"
    else
        SLIME_PYTHON="python"
    fi
fi
export SLIME_PYTHON
