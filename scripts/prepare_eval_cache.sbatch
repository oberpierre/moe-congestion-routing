#!/bin/bash
#SBATCH --job-name=eval-cache-prep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1           # only so the EDF/NGC container's NVIDIA hook can start
#SBATCH --time=02:00:00
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/eval_cache_prep_%j.log
#SBATCH --error=logs/eval_cache_prep_%j.log

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

DATA_ENV_EXPORT=""
[ -n "${DATA_STORE:-}" ] && DATA_ENV_EXPORT+="export DATA_STORE=$DATA_STORE; "
[ -n "${SCRATCH:-}" ] && DATA_ENV_EXPORT+="export SCRATCH=$SCRATCH; "

# Same cache locations as train_moe_climblab.sbatch
CACHE_ENV_EXPORT="
  export UV_CACHE_DIR=$WORKDIR/uv-cache
  export UV_PYTHON_INSTALL_DIR=$WORKDIR/uv-python
  export HF_HOME=$WORKDIR/hf_cache
  export TRITON_CACHE_DIR=$WORKDIR/.triton_cache
  export TORCHINDUCTOR_CACHE_DIR=$WORKDIR/.inductor_cache
  $DATA_ENV_EXPORT
"

# The venv it needs is the same one training builds, so this reuses train_moe_climblab.sbatch's
# environment-preparation step
srun --nodes=1 --ntasks=1 --environment=alps6 bash -c "
  set -euo pipefail
  $CACHE_ENV_EXPORT
  cd $WORKDIR
  if [ ! -f .venv/pyvenv.cfg ] || ! grep -q 'include-system-site-packages = true' .venv/pyvenv.cfg; then
    rm -rf .venv
    uv venv --system-site-packages --python \"\$(which python)\"
  fi
  # --reinstall-package huggingface-hub: the venv has system site-packages so the container's
  # torch/TE are visible, but that also exposes the container's older huggingface_hub, which
  # is incompatible with the datasets version uv installs. Reinstall is therefore required.
  uv sync --inexact --no-install-package torch --reinstall-package huggingface-hub
  set +u; source .venv/bin/activate; set -u
  # megatron-core 0.18 needs nvidia-resiliency-ext>=0.6; the NGC image ships 0.5.x. Newer wheel
  # into the venv (shadows the container copy), --no-deps so it can't pull a conflicting torch.
  uv pip install --no-deps 'nvidia-resiliency-ext>=0.6.0'
  echo '[env] build complete'
"

srun --nodes=1 --ntasks=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) ===\"
  $CACHE_ENV_EXPORT
  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  # Unbuffered: stdout here is a redirected log file, not a terminal, so Python block-buffers it
  # and a job that is working looks like a job that has hung. Children inherit the variable.
  export PYTHONUNBUFFERED=1
  python scripts/prepare_eval_cache.py

  echo \"=== done | \$(date) ===\"
"
