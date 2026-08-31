#!/bin/bash
# Build the `slime` conda env on a GPU host (Hopper or Blackwell).
#
# Adapted from slime/build_conda.sh. Four differences, all forced by the box:
#   1. conda, not micromamba (already installed, and `feature install genai_conda`
#      is what put it there).
#   2. Build products and logs live inside this repository and are gitignored.
#   3. Package downloads go through `with-proxy`, while GitHub repositories use SSH.
#      Set the proxy only per package command, never globally: a global proxy also
#      forwards *internal* requests and they come back
#      "403 Forbidden. Your destination may have been blocked by a destination
#      filter." (general/boxes/dev_machine.md §3).
#   4. sglang and Megatron are cloned here; slime is the repository submodule at
#      $SLIME_DIR, so the pinned fork revision is the one that gets installed.
#
# NOTE (cluster packaging): this env uses `pip install -e` for sglang /
# Megatron / slime, which conda_pack refuses, so this development environment
# is not directly packable
# (dev_machine.md §5.4). That is deliberate for now: local 8-GPU dev first.
#
# Run under `systemd-run --user` so it survives the agent session:
#   systemd-run --user --unit=slime_env_build --same-dir \
#     --setenv=PATH="$PATH" -- bash build_slime_env.sh

set -eo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE_ROOT=$(cd "$REPO_DIR/.." && pwd)
BASE_DIR=${BASE_DIR:-"$REPO_DIR/build"}
SLIME_DIR=${SLIME_DIR:-"$REPO_DIR/slime"}
SGLANG_DIR=${SGLANG_DIR:-"$BASE_DIR/sglang"}
MEGATRON_DIR=${MEGATRON_DIR:-"$BASE_DIR/Megatron-LM"}
HIDAGENT_DIR=${HIDAGENT_DIR:-"$WORKSPACE_ROOT/hidagent"}
ENV_PREFIX=${ENV_PREFIX:-"$WORKSPACE_ROOT/conda_envs/slime"}
LOG_DIR=${LOG_DIR:-"$REPO_DIR/logs"}
mkdir -p "$BASE_DIR" "$LOG_DIR"

if [ ! -f "$SLIME_DIR/pyproject.toml" ]; then
  echo "slime submodule is missing; run: git submodule update --init --recursive" >&2
  exit 1
fi

# Keep in sync with slime/build_conda.sh and docker/Dockerfile.
export SGLANG_COMMIT="0b3bb0cbe31873994c9f989fddfe2f87ca839fdd"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="v0.5.15.post1"

# Keep GitHub fetches on SSH. The dev environment injects HTTPS rewrites through
# GIT_CONFIG_COUNT; HTTPS is blocked for this workspace while the user's SSH identity is
# allowed. Reset only those injected command-line entries, leaving normal git/ssh config.
export GIT_CONFIG_COUNT=0

# State the local GPU architecture rather than trusting each build system's autodetect:
# apex, TE and flash-attn guess differently, and a missing kernel appears only at runtime.
# H200 reports 9.0; B200 reports 10.0. An explicit caller override still wins.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
    | head -n 1 | tr -d '[:space:]')
  export TORCH_CUDA_ARCH_LIST
fi
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
  echo "could not detect GPU compute capability; set TORCH_CUDA_ARCH_LIST" >&2
  exit 1
fi
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# The H200 path below is the stack validated end-to-end on this host.  Keep
# cu129 for Blackwell, where it is slime upstream's default.
if [ -z "${TORCH_CUDA_VARIANT:-}" ]; then
  case "$TORCH_CUDA_ARCH_LIST" in
    9.*) TORCH_CUDA_VARIANT=cu128 ;;
    *) TORCH_CUDA_VARIANT=cu129 ;;
  esac
fi
case "$TORCH_CUDA_VARIANT" in
  cu128)
    CUDA_RELEASE=12.8
    CUDA_FULL_VERSION=12.8.1
    CUDA_NVTX_VERSION=12.8.90
    CUDNN_WHEEL_VERSION=9.19.0.56
    ;;
  cu129)
    CUDA_RELEASE=12.9
    CUDA_FULL_VERSION=12.9.1
    CUDA_NVTX_VERSION=12.9.79
    CUDNN_WHEEL_VERSION=9.16.0.29
    ;;
  *)
    echo "unsupported TORCH_CUDA_VARIANT=$TORCH_CUDA_VARIANT (expected cu128 or cu129)" >&2
    exit 1
    ;;
esac
export TORCH_CUDA_VARIANT CUDA_RELEASE CUDA_FULL_VERSION
echo "TORCH_CUDA_VARIANT=$TORCH_CUDA_VARIANT CUDA_RELEASE=$CUDA_RELEASE"

# Resolve conda without an interactive shell: under `systemd-run --user` there is
# no ~/.bashrc, so the `conda` shell function does not exist and `conda info`
# would be a command-not-found several minutes into the run.
CONDA_BASE=${CONDA_BASE:-}
if [ -z "$CONDA_BASE" ]; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/.conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && CONDA_BASE="$c" && break
  done
fi
[ -z "$CONDA_BASE" ] && command -v conda >/dev/null && CONDA_BASE=$(conda info --base)
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "cannot locate conda; set CONDA_BASE=/path/to/miniconda3" >&2
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

step() { echo; echo "=============== $* ==============="; date; }

# ---------------------------------------------------------------- 1. the env
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  step "create conda env $ENV_PREFIX (python 3.12)"
  mkdir -p "$(dirname "$ENV_PREFIX")"
  if [ -n "${BASE_ENV_PREFIX:-}" ]; then
    conda create -p "$ENV_PREFIX" --clone "$BASE_ENV_PREFIX" -y
  else
    with-proxy conda create -p "$ENV_PREFIX" python=3.12 pip -c conda-forge -y
  fi
fi
conda activate "$ENV_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_DIR:$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS="${MAX_JOBS:-8}"

# CUDA 12.8 rejects GCC 14 before compilation starts.  Conda activation may
# pre-populate CC/CXX with its GCC 14 wrappers, so inspect the selected compiler
# rather than checking only whether the variables are empty.  The validation host
# ships a supported GCC 11.
selected_cc=${CC:-cc}
selected_gcc_major=$($selected_cc -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1)
if [ -z "$selected_gcc_major" ] || [ "$selected_gcc_major" -ge 14 ]; then
  if [ -x /usr/bin/gcc ] && [ -x /usr/bin/g++ ]; then
    host_gcc_major=$(/usr/bin/gcc -dumpfullversion -dumpversion | cut -d. -f1)
    if [ "$host_gcc_major" -lt 14 ]; then
      export CC=/usr/bin/gcc
      export CXX=/usr/bin/g++
      export CUDAHOSTCXX=/usr/bin/g++
    fi
  fi
elif [ -n "${CXX:-}" ]; then
  export CUDAHOSTCXX="$CXX"
fi
echo "python: $(which python)   CUDA_HOME=$CUDA_HOME"
echo "native compiler: CC=${CC:-default} CXX=${CXX:-default} MAX_JOBS=$MAX_JOBS"

# ---------------------------------------------------------------- 2. CUDA toolkit
# A cloned environment may already contain nvcc from another stack (the current
# ViewAgent environment has 12.8). Presence alone is therefore not sufficient.
cuda_release=""
if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
  cuda_release=$(
    "$CONDA_PREFIX/bin/nvcc" --version \
      | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
      | head -n 1
  )
fi
if [ "$cuda_release" != "$CUDA_RELEASE" ]; then
  step "cuda $CUDA_FULL_VERSION + nccl + cudnn + rust"
  with-proxy conda install -p "$ENV_PREFIX" -y \
    "cuda=$CUDA_FULL_VERSION" "cuda-nvtx=$CUDA_NVTX_VERSION" \
    "cuda-nvtx-dev=$CUDA_NVTX_VERSION" nccl \
    -c "nvidia/label/cuda-$CUDA_FULL_VERSION" -c nvidia -c conda-forge
  with-proxy conda install -p "$ENV_PREFIX" -c conda-forge cudnn -y
  # sglang's editable install builds a Rust extension (sglang-grpc via
  # setuptools-rust), so the env needs a working rustc + cargo.
  with-proxy conda install -p "$ENV_PREFIX" -c conda-forge rust -y
fi
with-proxy pip install cuda-python==12.9

# ------------------------------------------------- 2b. make $CUDA_HOME look standard
# conda's CUDA packages do not lay the toolkit out the way every build system expects.
# The headers land in $CONDA_PREFIX/targets/x86_64-linux/include while $CONDA_PREFIX/include
# has none of them, so anything compiling against $CUDA_HOME/include -- which is
# transformer_engine, apex, and torch's own cpp_extension -- dies on
#   fatal error: cuda_runtime_api.h: No such file or directory
# several hundred lines into a ninja build, with the real cause nowhere near the last
# error printed. (Observed here: TE then swallowed it and re-raised an unrelated 404 from
# its own fallback download, which is what the traceback actually showed.)
#
# The libraries are already mirrored into $CONDA_PREFIX/lib; only the headers are not.
# Symlink rather than copy, and skip anything already present so nothing is clobbered.
step "normalise the CUDA toolkit layout under \$CUDA_HOME"
_cuda_inc="$CONDA_PREFIX/targets/x86_64-linux/include"
normalise_cuda_headers() {
  linked=0
  for include_dir in \
    "$_cuda_inc" \
    "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/include; do
    [ -d "$include_dir" ] || continue
    for header in "$include_dir"/*; do
      target="$CONDA_PREFIX/include/$(basename "$header")"
      [ -e "$target" ] || { ln -s "$header" "$target" && linked=$((linked + 1)); }
    done
  done
  echo "linked $linked CUDA/cuDNN headers into $CONDA_PREFIX/include"
}
normalise_cuda_headers
# Same class of problem on the library side, and it bites at *runtime* rather than at
# build time. flashinfer JIT-compiles its kernels on first use and links with
#   -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs
# but conda puts the libraries in lib/, not lib64/. The sglang engine then dies mid-startup
# with `/usr/bin/ld: cannot find -lcudart`, several hundred lines into a ninja log, hours
# after this script reported success.
if [ ! -e "$CONDA_PREFIX/lib64" ]; then
  ln -s lib "$CONDA_PREFIX/lib64"
  echo "linked \$CONDA_PREFIX/lib64 -> lib (flashinfer JIT looks for CUDA libs there)"
fi

# Belt and braces: some builds read CPATH instead of probing $CUDA_HOME/include.
export CPATH="$_cuda_inc${CPATH:+:$CPATH}"
_wheel_cudnn_lib="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [ -d "$_wheel_cudnn_lib" ]; then
  export LIBRARY_PATH="$_wheel_cudnn_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$_wheel_cudnn_lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
python -c "
import os, pathlib
h = pathlib.Path(os.environ['CONDA_PREFIX']) / 'include' / 'cuda_runtime_api.h'
assert h.exists(), f'{h} still missing -- the compile steps below would all fail'
print('cuda_runtime_api.h visible at', h)
"

# ---------------------------------------------------------------- 3. sglang
step "sglang @ $SGLANG_COMMIT"
# Guarded so a resumed build does not rebuild sglang's Rust extension and re-pin torch,
# which together are most of the wall-clock before the first compile step.
if python -c "
import importlib.metadata as metadata
import sys

import sglang
import torch

ready = (
    torch.__version__ == '2.11.0+$TORCH_CUDA_VARIANT'
    and sglang.__version__.startswith('0.5.15')
    and metadata.version('sglang-kernel') == '0.4.4'
    and metadata.version('sgl-deep-gemm') == '0.1.4'
)
sys.exit(0 if ready else 1)
" 2>/dev/null; then
  echo "sglang 0.5.15 + torch 2.11.0/$TORCH_CUDA_VARIANT already installed; skipping"
else
if [ ! -d "$SGLANG_DIR" ]; then
  git clone git@github.com:sgl-project/sglang.git "$SGLANG_DIR"
fi
cd "$SGLANG_DIR"
git checkout ${SGLANG_COMMIT}
with-proxy pip install -e "python[all]" \
  --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA_VARIANT"

# PyPI defaults are cu13 wheels; force the selected CUDA 12 builds back.
# torchvision is PINNED. Unpinned, pip takes the newest on the index (0.28.0), which
# pairs with torch 2.13 -- the C extension then fails to register and the *only* symptom is
# `RuntimeError: operator torchvision::nms does not exist`, raised from inside sglang's
# import, plus a transformers lazy-import failure that surfaces as
# `ModuleNotFoundError: Could not import module 'PreTrainedModel'` and takes Bridge with
# it. Three broken packages, one unpinned version. The pairing is
# tv 0.24<->torch 2.9, 0.25<->2.10, 0.26<->2.11, 0.27<->2.12, 0.28<->2.13.
with-proxy pip install --force-reinstall --no-deps \
  "torch==2.11.0+$TORCH_CUDA_VARIANT" \
  "torchvision==0.26.0+$TORCH_CUDA_VARIANT" \
  "torchaudio==2.11.0+$TORCH_CUDA_VARIANT" \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_VARIANT"
with-proxy pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.4 sgl-deep-gemm==0.1.4 \
  --index-url "https://docs.sglang.ai/whl/$TORCH_CUDA_VARIANT/"

# Repair the cu12/cu13 spill: pip uninstall stomps libs co-owned across the two.
with-proxy pip uninstall -y \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx nvidia-cutlass-dsl-libs-cu13 || true
with-proxy pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 "nvidia-cudnn-cu12==$CUDNN_WHEEL_VERSION" nvidia-cufft-cu12 \
  nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_VARIANT" \
  --extra-index-url https://pypi.org/simple
fi   # end of the sglang/torch guard

with-proxy pip install cmake ninja

# The PyTorch/SGLang wheels above add component-specific CUDA include trees
# after the first normalisation pass.  Expose those headers before compiling
# flash-attn and Transformer Engine.
normalise_cuda_headers
_wheel_cudnn_lib="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [ -d "$_wheel_cudnn_lib" ]; then
  export LIBRARY_PATH="$_wheel_cudnn_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$_wheel_cudnn_lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# ---------------------------------------------------------------- 4. kernels
# Use the CUDA 12 / PyTorch 2.11 wheel validated by slime main.
step "flash-attn 2.8.3"
if python -c "import importlib.metadata as m; assert m.version('flash-attn') == '2.8.3'" \
    2>/dev/null; then
  echo "flash-attn 2.8.3 already installed; skipping"
else
  with-proxy pip uninstall -y flash-attn-4 flash_attn_4 || true
  flash_attn_wheel="https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697"
  if ! with-proxy pip install --no-deps "$flash_attn_wheel"; then
    echo "prebuilt flash-attn wheel unavailable; compiling 2.8.3 locally"
    FLASH_ATTENTION_FORCE_BUILD=TRUE with-proxy pip install \
      --no-build-isolation --no-cache-dir flash-attn==2.8.3
  fi
fi

if [ "${INSTALL_QWEN35_EXTRAS:-0}" = "1" ]; then
  step "optional Qwen3.5 GDN kernels"
  with-proxy pip install flash-linear-attention==0.4.2
  with-proxy pip install git+ssh://git@github.com/QwenLM/FlashQLA.git --no-build-isolation
fi

step "transformer_engine 2.16.1"
if python -c "import importlib.metadata as m; assert m.version('transformer-engine') == '2.16.1'; import transformer_engine.pytorch" \
    2>/dev/null; then
  echo "transformer_engine 2.16.1 already installed; skipping"
else
  with-proxy pip install --no-build-isolation "transformer_engine[pytorch]==2.16.1"
fi

if [ "${INSTALL_APEX:-0}" = "1" ]; then
  step "optional apex"
  NVCC_APPEND_FLAGS="--threads 4" with-proxy pip -v install \
    --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+ssh://git@github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
fi

step "torch_memory_saver / Megatron-Bridge / modelopt / sgl-router"
# These published wheels are the exact combination exercised by the synchronous
# Habitat-GS smoke and resume tests.  In particular, torch-memory-saver ships
# both cu12 and cu13 preload hooks; slime selects the one matching torch.
with-proxy pip install --force-reinstall --no-deps torch-memory-saver==0.0.9.post1
with-proxy pip install --no-deps megatron-bridge==0.3.1
with-proxy pip install --no-deps "nvidia-modelopt[torch]==0.37.0" pulp torchprofile
with-proxy pip install --no-deps sglang-router==0.3.2

# ---------------------------------------------------------------- 5. megatron
step "Megatron-LM @ $MEGATRON_COMMIT"
if [ ! -d "$MEGATRON_DIR" ]; then
  git clone git@github.com:NVIDIA/Megatron-LM.git --recursive "$MEGATRON_DIR"
fi
with-proxy pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
cd "$MEGATRON_DIR" && git checkout ${MEGATRON_COMMIT}
git submodule update --init --recursive
# --no-build-isolation: setup.py builds megatron.core.datasets.helpers_cpp and
# subprocess-shells `python3 -m pybind11`. Without it pip's build venv has no
# pybind11, the extension is marked optional and silently skipped, and GPT
# dataset loading breaks later with nothing pointing here.
with-proxy pip install -e . --no-build-isolation

# ---------------------------------------------------------------- 6. slime
step "slime (existing checkout: $SLIME_DIR)"
cd "$SLIME_DIR"
# Runtime deps first, then slime itself with --no-deps so pip does not
# re-resolve and stomp the pinned native libs (torch+cu12, sglang-kernel+cu12).
with-proxy pip install -r requirements.txt
# openai-agents currently permits OpenAI 3.x while SGLang 0.5.15 requires
# exactly 2.6.1.  Reassert the serving stack's pin after the broad slime list.
with-proxy pip install --force-reinstall --no-deps openai==2.6.1
with-proxy pip install -e . --no-deps
cd "$SLIME_DIR/slime/backends/megatron_utils/kernels/int4_qat"
if python -c "import importlib.metadata as m; assert m.version('fake-int4-quant-cuda') == '0.0.0'" \
    2>/dev/null; then
  echo "fake_int4_quant_cuda already installed; skipping"
else
  with-proxy pip install . --no-build-isolation
fi

# https://github.com/pytorch/pytorch/issues/168167 -- conv3d perf regression
with-proxy pip install --no-deps "nvidia-cudnn-cu12==$CUDNN_WHEEL_VERSION"
# numpy<2 is required by gym / gym_sokoban, which the VAGEN environments use. scipy has
# to come down with it: the version pulled in by the sglang deps wants numpy>=2 and dies on
# `module 'numpy' has no attribute 'long'` from inside scipy.sparse -- reached, confusingly,
# through transformers' lazy PreTrainedModel import.
with-proxy pip install "numpy==1.26.4" "scipy==1.17.1"
# kernels 0.15.x raises ValueError("Either a revision or a version must be
# specified") from transformers.integrations.hub_kernels, so `import sglang`
# fails at runtime.
with-proxy pip install "kernels<0.15.0"

# ---------------------------------------------------------------- 7. patches
step "patches"
patch_dir="$SLIME_DIR/docker/patch/${PATCH_VERSION}"
if [ ! -d "$patch_dir" ]; then
  echo "Patch directory does not exist: $patch_dir" >&2
  exit 1
fi

if [ -d "$SGLANG_DIR/.git" ]; then
  cd "$SGLANG_DIR"
  for patch_name in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch; do
    patch_path="$patch_dir/$patch_name"
    [ -f "$patch_path" ] || continue
    if git apply --check "$patch_path"; then
      git apply "$patch_path"
    elif git apply --reverse --check "$patch_path"; then
      echo "$patch_name already applied, skipping"
    else
      echo "$patch_name does not apply cleanly" >&2
      exit 1
    fi
  done
else
  echo "using published SGLang wheel; source-only SGLang patches skipped"
fi

cd "$MEGATRON_DIR"
megatron_patch="$patch_dir/megatron.patch"
if [ ! -f "$megatron_patch" ]; then
  echo "Megatron patch does not exist: $megatron_patch" >&2
  exit 1
fi
if git apply --reverse --check "$megatron_patch"; then
  echo "megatron.patch already applied, skipping"
else
  git update-index --refresh || true
  if ! git apply "$megatron_patch" --3way; then
    echo "megatron.patch does not apply cleanly" >&2
    exit 1
  fi
  git grep -n '^<<<<<<< ' -- . && { echo "megatron patch conflicted" >&2; exit 1; } || true
fi

# ------------------------------------------------------- 8. VAGEN env deps
# The environments we are porting come with their own dependencies. Kept here
# rather than in a second env: the rollout runs Sokoban *inside* the slime
# process, so they have to be importable from the same interpreter.
step "VAGEN environment deps"
with-proxy pip install gym-sokoban gymnasium "gym<1.0" || \
  echo "WARN: sokoban deps failed; the env will not import"

if [ -f "$HIDAGENT_DIR/pyproject.toml" ]; then
  step "HIDAgent runtime/evaluation overlay"
  with-proxy pip install -e "$HIDAGENT_DIR[slime]"
else
  echo "WARN: HIDAgent checkout not found at $HIDAGENT_DIR; skipping overlay"
fi

# Keep activation relocatable and make one internally consistent cuDNN set win.
# The conda toolkit and PyTorch wheel both ship cuDNN libraries; putting the
# wheel directory first avoids mixing e.g. libcudnn_cnn from one with
# libcudnn_graph from the other. Editable source paths point at this workspace,
# never at the checkout from which an environment may have been cloned.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_vagen_slime.sh" <<EOF
export VAGEN_SLIME_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
_vagen_cudnn_lib="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="\${_vagen_cudnn_lib}:$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
unset _vagen_cudnn_lib
export VAGEN_SLIME_OLD_PYTHONPATH="\${PYTHONPATH:-}"
export PYTHONPATH="$REPO_DIR:$MEGATRON_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export CUDA_HOME="$CONDA_PREFIX"
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/zz_vagen_slime.sh" <<'EOF'
export LD_LIBRARY_PATH="${VAGEN_SLIME_OLD_LD_LIBRARY_PATH:-}"
export PYTHONPATH="${VAGEN_SLIME_OLD_PYTHONPATH:-}"
unset VAGEN_SLIME_OLD_LD_LIBRARY_PATH VAGEN_SLIME_OLD_PYTHONPATH
EOF

step "DONE"
python - <<'PY'
import importlib
import os
import sys

report = {}
required = ["torch", "sglang", "megatron.core", "transformer_engine", "slime",
            "megatron.bridge", "sglang_router", "flash_attn"]
for mod in [*required, "gym_sokoban"]:
    try:
        m = importlib.import_module(mod)
        report[mod] = getattr(m, "__version__", "ok")
    except Exception as exc:
        report[mod] = f"MISSING ({type(exc).__name__}: {exc})"
import torch
expected_torch = f"2.11.0+{os.environ['TORCH_CUDA_VARIANT']}"
assert torch.__version__ == expected_torch, (torch.__version__, expected_torch)
assert not [mod for mod in required if report[mod].startswith("MISSING")], report
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  "
      f"available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print("capability:", torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
    print("cuda sum:", torch.arange(1024, device="cuda").sum().item())
for k, v in report.items():
    print(f"  {k:22s} {v}")
PY
