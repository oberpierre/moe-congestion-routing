#!/bin/bash
#SBATCH --job-name=climb-prep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1           # only so the EDF/NGC container's NVIDIA hook can start
#SBATCH --time=04:00:00
#SBATCH --output=logs/climb_prep_%j.log
#SBATCH --error=logs/climb_prep_%j.log

set -euo pipefail
mkdir -p logs

# From config.sh (exported into the job env by `source config.sh` before sbatch).
: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

# Data-prep config: first CLI arg
CONFIG="${1:-configs/data/climbmix_small.yaml}"

# Cluster data YAMLs reference ${DATA_STORE} (persistent Store output, set in config.sh) and
# ${SCRATCH} (system-provided per-cluster scratch, for the regenerable cache). Pass both into
# the container when set — the local smoke config uses relative artifacts/ paths and needs
# neither, and an unset var stays literal so the loader fails loud, not silently.
DATA_ENV_EXPORT=""
[ -n "${DATA_STORE:-}" ] && DATA_ENV_EXPORT+="export DATA_STORE=$DATA_STORE; "
[ -n "${SCRATCH:-}" ] && DATA_ENV_EXPORT+="export SCRATCH=$SCRATCH; "

# Paths are resolved here on the host and embedded into the command below, so the
# container's entrypoint (entrypoint = true) cannot drop them from the environment.
srun --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) | config=$CONFIG ===\"

  # Redirect uv + HF caches off the small home quota onto scratch. uv itself ships
  # in the NGC image (on PATH); we just use it so the env matches uv.lock.
  export UV_CACHE_DIR=$WORKDIR/uv-cache
  export UV_PYTHON_INSTALL_DIR=$WORKDIR/uv-python
  export HF_HOME=$WORKDIR/hf_cache
  $DATA_ENV_EXPORT

  # one thread per math lib: parallelism is process-level (convert_workers)
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

  cd $WORKDIR
  uv run python scripts/prepare_climb.py $CONFIG

  echo \"=== done | \$(date) ===\"
"
