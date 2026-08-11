#!/bin/bash
#SBATCH --job-name=moe-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/moe_eval_%j.log
#SBATCH --error=logs/moe_eval_%j.log

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

# Positional args mirror train_moe_climblab.sbatch's shape: config, then where the checkpoint is,
# then which iteration of it to score. All three are required.
CONFIG="${1:?pass an eval config, e.g. configs/eval/flame_suite.yaml}"
CHECKPOINTS="${2:?pass a checkpoints DIR to evaluate, e.g. artifacts/<run>/checkpoints}"
CKPT_STEP="${3:?pass a checkpoint step (ckpt_step) to evaluate, e.g. 5500}"

DATA_ENV_EXPORT=""
[ -n "${DATA_STORE:-}" ] && DATA_ENV_EXPORT+="export DATA_STORE=$DATA_STORE; "
[ -n "${SCRATCH:-}" ] && DATA_ENV_EXPORT+="export SCRATCH=$SCRATCH; "

# Caches redirected off the small home quota onto scratch, identical to train_moe_climblab.sbatch
# $HF_HOME must be the same as the prepare_eval_cache job uses.
CACHE_ENV_EXPORT="
  export UV_CACHE_DIR=$WORKDIR/uv-cache
  export UV_PYTHON_INSTALL_DIR=$WORKDIR/uv-python
  export HF_HOME=$WORKDIR/hf_cache
  export TRITON_CACHE_DIR=$WORKDIR/.triton_cache
  export TORCHINDUCTOR_CACHE_DIR=$WORKDIR/.inductor_cache
  $DATA_ENV_EXPORT
"

# Prepare the environment once, before the eval step. Identical to train_moe_climblab.sbatch's
# env-prep step
srun --nodes=1 --ntasks=1 --environment=alps6 bash -c "
  set -euo pipefail
  $CACHE_ENV_EXPORT
  cd $WORKDIR
  # Rebuild the venv if it is missing, misconfigured, or damaged. Probe the dependencies the eval
  # path actually needs.
  if [ ! -f .venv/pyvenv.cfg ] \
     || ! grep -q 'include-system-site-packages = true' .venv/pyvenv.cfg \
     || ! .venv/bin/python -c 'import transformers, huggingface_hub, datasets' 2>/dev/null; then
    echo '[env] rebuilding venv (missing, misconfigured, or failed its import probe)'
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
  ./scripts/apply-patches.sh
  echo '[env] build complete'
"

# Launch the eval. run_lm_eval.py itself launches `torch.distributed.run --standalone
# --nproc-per-node NPROC`, one process per GPU, so this is still a single Slurm task (like
# training) that fans out into NPROC processes via torch-elastic
srun -l --mpi=pmix --wait 60 \
     --ntasks-per-node=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) ===\"
  $CACHE_ENV_EXPORT

  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export OMP_NUM_THREADS=\$((SLURM_CPUS_PER_TASK/NPROC))
  export TORCH_NCCL_AVOID_RECORD_STREAMS=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export PYTHONUNBUFFERED=1

  NPROC=\${SLURM_GPUS_PER_NODE:-\$(nvidia-smi -L | wc -l)}
  echo \"=== \$NPROC GPU(s) ===\"

  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  numactl --membind=0-3 python scripts/run_lm_eval.py --config $CONFIG --load $CHECKPOINTS --ckpt-step $CKPT_STEP \
    --nproc \$NPROC

  echo \"=== done | \$(date) ===\"
"
