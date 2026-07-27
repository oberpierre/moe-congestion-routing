#!/bin/bash
#SBATCH --job-name=moe-train
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/moe_train_%j.log
#SBATCH --error=logs/moe_train_%j.log

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

CONFIG="${1:-configs/train/switch_local.yaml}"

# Multi-node only: point every node's torch-elastic at a c10d rendezvous on node 0 of the
# allocation; ranks are assigned by the rendezvous. Single-node leaves DIST_ARGS empty, so the
# launcher takes its --standalone path.
DIST_ARGS=""
if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    DIST_ARGS="--nnodes $SLURM_NNODES --rdzv-endpoint $MASTER_ADDR:25678"
fi

# One shared run tag so all nodes write to the SAME <output_dir>/<tag> dir (see run_moe_pretrain).
RUN_TAG="$(date +%Y%m%d-%H%M%S)"

DATA_ENV_EXPORT=""
[ -n "${DATA_STORE:-}" ] && DATA_ENV_EXPORT+="export DATA_STORE=$DATA_STORE; "
[ -n "${SCRATCH:-}" ] && DATA_ENV_EXPORT+="export SCRATCH=$SCRATCH; "

# Caches redirected off the small home quota onto scratch. uv itself ships in the NGC image.
CACHE_ENV_EXPORT="
  export UV_CACHE_DIR=$WORKDIR/uv-cache
  export UV_PYTHON_INSTALL_DIR=$WORKDIR/uv-python
  export HF_HOME=$WORKDIR/hf_cache
  export TRITON_CACHE_DIR=$WORKDIR/.triton_cache
  export TORCHINDUCTOR_CACHE_DIR=$WORKDIR/.inductor_cache
  $DATA_ENV_EXPORT
"

# Prepare the environment ONCE for all nodes, before launchhing the training script on every node.
# Ensures the venv correctly uses the system torch/TE and uv manages other dependencies, while
# avoiding N nodes racing to write the same .venv/uv-cache concurrently.
srun --nodes=1 --ntasks=1 --environment=alps6 bash -c "
  set -euo pipefail
  $CACHE_ENV_EXPORT
  cd $WORKDIR
  if [ ! -f .venv/pyvenv.cfg ] || ! grep -q 'include-system-site-packages = true' .venv/pyvenv.cfg; then
    rm -rf .venv
    uv venv --system-site-packages --python \"\$(which python)\"
  fi
  uv sync --inexact --no-install-package torch
  set +u; source .venv/bin/activate; set -u
  # megatron-core 0.18 needs nvidia-resiliency-ext>=0.6; the NGC image ships 0.5.x. Newer wheel
  # into the venv (shadows the container copy), --no-deps so it can't pull a conflicting torch.
  uv pip install --no-deps 'nvidia-resiliency-ext>=0.6.0'
  echo '[env] build complete'
"

# Launch the training script on every node, one task per node. The launcher runs once per node and
# run_moe_pretrain.py launches via `python -m torch.distributed.run`, so the workers inherit the
# venv interpreter (sys.executable) and see the venv's own deps alongside the system torch/TE deps.
srun -l --mpi=pmix --network=disable_rdzv_get --wait 60 \
     --ntasks-per-node=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) ===\"
  $CACHE_ENV_EXPORT
  export MOE_RUN_TAG=$RUN_TAG

  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
  export TORCH_NCCL_AVOID_RECORD_STREAMS=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  NPROC=\${SLURM_GPUS_PER_NODE:-\$(nvidia-smi -L | wc -l)}
  echo \"=== \$NPROC GPU(s) x \${SLURM_NNODES:-1} node(s) ${DIST_ARGS:+| $DIST_ARGS} ===\"

  numactl --membind=0-3 python scripts/run_moe_pretrain.py --config $CONFIG \
    --nproc \$NPROC $DIST_ARGS \
    --no-capture

  echo \"=== done | \$(date) ===\"
"
