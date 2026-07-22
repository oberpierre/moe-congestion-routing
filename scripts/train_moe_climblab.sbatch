#!/bin/bash
#SBATCH --job-name=moe-train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/moe_train_%j.log
#SBATCH --error=logs/moe_train_%j.log

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

CONFIG="${1:-configs/train/switch_local.yaml}"
NPROC=${SLURM_GPUS_PER_NODE:-1}

DATA_ENV_EXPORT=""
[ -n "${DATA_STORE:-}" ] && DATA_ENV_EXPORT+="export DATA_STORE=$DATA_STORE; "
[ -n "${SCRATCH:-}" ] && DATA_ENV_EXPORT+="export SCRATCH=$SCRATCH; "

# Paths resolved on the host and embedded, so the container entrypoint can't drop them.
srun --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) ===\"

  # Redirect uv + HF caches off the small home quota onto scratch. uv itself ships
  # in the NGC image (on PATH); we just use it so the env matches uv.lock.
  export UV_CACHE_DIR=$WORKDIR/uv-cache
  export UV_PYTHON_INSTALL_DIR=$WORKDIR/uv-python
  export HF_HOME=$WORKDIR/hf_cache
  $DATA_ENV_EXPORT

  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))

  cd $WORKDIR
  uv sync --inexact
  uv run python scripts/run_moe_pretrain.py --config $CONFIG --nproc $NPROC

  echo \"=== done | \$(date) ===\"
"
