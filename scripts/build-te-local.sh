#!/usr/bin/env bash
# Build Transformer Engine into the local uv venv.
# NOT needed on the clusterr (NGC PyTorch container already ships TE).
#
# TE's core ships prebuilt wheels, but its PyTorch bindings (transformer_engine_torch)
# are source-only and must compile against the SAME CUDA major as the installed torch.
set -euo pipefail
cd "$(dirname "$0")/.."

TE_VERSION="${TE_VERSION:-2.17.0}"

TORCH_CUDA=$(uv run python -c "import torch; print(torch.version.cuda)")   # e.g. 13.0
CUDA_MAJOR=${TORCH_CUDA%%.*}
CUDA_HOME=$(ls -d "/usr/local/cuda-${TORCH_CUDA}" "/usr/local/cuda-${CUDA_MAJOR}" 2>/dev/null | head -1 || true)
if [[ -z "${CUDA_HOME}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "ERROR: no CUDA ${CUDA_MAJOR}.x toolkit under /usr/local matching torch's CUDA ${TORCH_CUDA}." >&2
  echo "       Install the matching CUDA toolkit (nvcc), or realign torch to your toolkit." >&2
  exit 1
fi

SP=$(uv run python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
NV="${SP}/nvidia"
echo "torch CUDA=${TORCH_CUDA}  CUDA_HOME=${CUDA_HOME}  site-packages=${SP}"

# Prebuilt frontend + core (no build). The cuNN suffix must match torch's CUDA major.
uv pip install "transformer_engine==${TE_VERSION}" "transformer_engine_cu${CUDA_MAJOR}==${TE_VERSION}"
uv pip install setuptools wheel ninja   # build deps for the torch bindings (no build isolation)

# Source-build the PyTorch bindings against the matching toolkit + torch's bundled headers.
CUDA_HOME="${CUDA_HOME}" \
PATH="${CUDA_HOME}/bin:${PATH}" \
CPATH="${NV}/nccl/include:${NV}/cudnn/include:${NV}/cu${CUDA_MAJOR}/include" \
LD_LIBRARY_PATH="${NV}/nccl/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}" \
MAX_JOBS="${MAX_JOBS:-8}" NVTE_FRAMEWORK=pytorch \
  uv pip install --no-build-isolation "transformer_engine_torch==${TE_VERSION}"

# Verify (the launchers put torch's nccl/cudnn on LD_LIBRARY_PATH at runtime the same way).
LD_LIBRARY_PATH="${NV}/nccl/lib:${NV}/cudnn/lib:${LD_LIBRARY_PATH:-}" \
  uv run python -c "import transformer_engine.pytorch as te; print('Transformer Engine OK:', te.__name__)"
echo "Done. Run training/inference normally: the launchers wire LD_LIBRARY_PATH automatically."
