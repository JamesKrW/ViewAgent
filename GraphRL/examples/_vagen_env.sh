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
    # conda_envs/verl_sglang058, built to the version set verified to run this stack
    # (sglang 0.5.8 + flashinfer 0.6.1 + torch 2.9.1+cu128 + transformers 4.57.1).
    # Deliberately NOT conda_envs/slime:
    # that one is the SLIME-era environment and the two cannot be merged --
    # sglang 0.5.15 requires kernels>=0.14.1 while transformers[kernels] requires
    # <0.13, so VAGEN's own `[sglang]` extra does not resolve as written. See
    # GraphRL/README.md for the version set this env was built to.
    LOCAL_PYTHON="${VIEWAGENT_ROOT}/../conda_envs/verl_sglang058/bin/python"
    if [ -x "${LOCAL_PYTHON}" ]; then
        VAGEN_PYTHON="${LOCAL_PYTHON}"
    else
        echo "conda_envs/verl not found. Build it first -- see GraphRL/README.md" >&2
        VAGEN_PYTHON="python"
    fi
fi
export VAGEN_PYTHON
