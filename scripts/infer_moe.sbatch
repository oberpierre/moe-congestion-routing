#!/bin/bash
#SBATCH --job-name=moe-infer
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/moe_infer_%j.log
#SBATCH --error=logs/moe_infer_%j.log
#
# One-shot generation only: a batch job has no terminal, and run_moe_infer.py rejects
# --interactive above --nproc 1 anyway. For interactive use on the cluster, run
# `srun --pty ... scripts/run_moe_infer.py --load ... --ckpt-step ... --interactive`
# directly instead of this script.

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

# Positional args mirror eval_moe.sbatch's shape: a checkpoints dir, a step, then the
# prompt. Everything after that is forwarded to run_moe_infer.py verbatim, which is how
# FLAME's `-- --dist-ckpt-strictness log_all` reaches the launcher.
CHECKPOINTS="${1:?pass a checkpoints DIR to generate from, e.g. artifacts/<run>/checkpoints}"
CKPT_STEP="${2:?pass a checkpoint step (ckpt-step) to generate from, e.g. 5500}"
: "${3:?pass a prompt string, e.g. \"The capital of France is\"}"
# Shift past the two interpolated args so the prompt is $1 and any extras are $2 onward, which
# is the shape the trailing `_ "$@"` on the generate srun forwards into the inner shell.
shift 2

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

# Prepare the environment once, before the generate step. Identical to eval_moe.sbatch's
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

# Launch generation. One GPU: generation runs at TP/PP/EP 1, so extra ranks would each
# build the same model and print the same completion, and MegatronLLM (the dynamic
# engine) refuses direct mode above EP 1 anyway. --nproc 1 is passed literally rather
# than derived from SLURM_GPUS_PER_NODE.
srun -l --mpi=pmix --wait 60 \
     --ntasks-per-node=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) ===\"
  $CACHE_ENV_EXPORT

  export CUDA_DEVICE_MAX_CONNECTIONS=1
  # Guarded so an unset SLURM_CPUS_PER_TASK cannot abort the job under set -u; divides by
  # 1 (not by a derived GPU count) because this job always runs on exactly one GPU.
  export OMP_NUM_THREADS=\$((\${SLURM_CPUS_PER_TASK:-1}/1))
  export TORCH_NCCL_AVOID_RECORD_STREAMS=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export PYTHONUNBUFFERED=1

  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  # The prompt (\$1) and any extra launcher flags (\${@:2}) arrive as positional parameters
  # from the trailing `_ \"\$@\"` below, never interpolated into this string: they are
  # arbitrary user input whose quotes would otherwise be parsed by the outer login-node
  # shell, and passing them positionally also keeps each extra a separate argument.
  # --engine is deliberately unset, so 'auto' picks dynamic here, where flash_attn exists.
  numactl --membind=0-3 python scripts/run_moe_infer.py --load $CHECKPOINTS --ckpt-step $CKPT_STEP \
    --nproc 1 --prompt \"\$1\" \"\${@:2}\"

  echo \"=== done | \$(date) ===\"
" _ "$@"
