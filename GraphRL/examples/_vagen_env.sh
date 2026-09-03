#!/usr/bin/env bash
# Shared runtime wiring for GraphRL's VAGEN/verl examples.

EXAMPLES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHRL_ROOT="$(cd "${EXAMPLES_ROOT}/.." && pwd)"
VIEWAGENT_ROOT="$(cd "${GRAPHRL_ROOT}/.." && pwd)"
VAGEN_ROOT="${GRAPHRL_ROOT}/VAGEN"
VERL_ROOT="${VAGEN_ROOT}/verl"

# Probed for a file, not a directory: an uninitialised submodule leaves
# VAGEN/verl present but empty, and left unresolved that surfaces much later as a
# Hydra error that does not mention verl.
if [ ! -f "${VERL_ROOT}/verl/trainer/config/ppo_trainer.yaml" ]; then
    echo "verl not found at ${VERL_ROOT}." >&2
    echo "Run: git submodule update --init --recursive" >&2
    return 1 2>/dev/null || exit 1
fi

export VIEWSUITE_ROOT="${VIEWSUITE_ROOT:-${VIEWAGENT_ROOT}}"
# verl comes first so this checkout wins over any installed copy -- the training
# conda env ships verl 0.6.1 as a package, which would otherwise shadow it.
export PYTHONPATH="${VERL_ROOT}:${VAGEN_ROOT}:${VIEWAGENT_ROOT}:${GRAPHRL_ROOT}:${GRAPHRL_ROOT}/LLaMA-Factory/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${VAGEN_PYTHON:-}" ]; then
    # Still conda_envs/slime: the interpreter is named for the backend it was
    # first built for, but it is the env carrying vllm, ray and verl's deps.
    LOCAL_PYTHON="${VIEWAGENT_ROOT}/../conda_envs/slime/bin/python"
    if [ -x "${LOCAL_PYTHON}" ]; then
        VAGEN_PYTHON="${LOCAL_PYTHON}"
    else
        VAGEN_PYTHON="python"
    fi
fi
export VAGEN_PYTHON
