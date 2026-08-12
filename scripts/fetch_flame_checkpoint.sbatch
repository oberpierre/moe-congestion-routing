#!/bin/bash
#SBATCH --job-name=fetch-flame-checkpoint
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1           # only so the EDF/NGC container's NVIDIA hook can start
#SBATCH --time=04:00:00
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/fetch_flame_checkpoint_%j.log
#SBATCH --error=logs/fetch_flame_checkpoint_%j.log

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"
: "${DATA_STORE:?set DATA_STORE in config.sh and source it before sbatch}"

# This job only fetches weights into $DATA_STORE, it needs no patched Megatron-LM or
# lm-evaluation-harness tree, and running ./scripts/apply-patches.sh here would reset both
# submodules to their pin out from under any eval job that happens to be running concurrently.

# Same cache locations as train_moe_climblab.sbatch / prepare_eval_cache.sbatch.
CACHE_ENV_EXPORT="
  export UV_CACHE_DIR=$WORKDIR/uv-cache
  export UV_PYTHON_INSTALL_DIR=$WORKDIR/uv-python
  export HF_HOME=$WORKDIR/hf_cache
  export TRITON_CACHE_DIR=$WORKDIR/.triton_cache
  export TORCHINDUCTOR_CACHE_DIR=$WORKDIR/.inductor_cache
  export DATA_STORE=$DATA_STORE
"

# The venv it needs is the same one training and eval-cache-prep build, so this reuses
# prepare_eval_cache.sbatch's environment-preparation step.
srun --nodes=1 --ntasks=1 --environment=alps6 bash -c "
  set -euo pipefail
  $CACHE_ENV_EXPORT
  cd $WORKDIR
  # Rebuild the venv if it is missing, misconfigured, or damaged. Probe the dependency this job
  # actually needs.
  if [ ! -f .venv/pyvenv.cfg ] \
     || ! grep -q 'include-system-site-packages = true' .venv/pyvenv.cfg \
     || ! .venv/bin/python -c 'import huggingface_hub' 2>/dev/null; then
    echo '[env] rebuilding venv (missing, misconfigured, or failed its import probe)'
    rm -rf .venv
    uv venv --system-site-packages --python \"\$(which python)\"
  fi
  # --reinstall-package huggingface-hub: the venv has system site-packages so the container's
  # torch/TE are visible, but that also exposes the container's older huggingface_hub, which
  # is incompatible with the datasets version uv installs. Reinstall is therefore required.
  uv sync --inexact --no-install-package torch --reinstall-package huggingface-hub
  echo '[env] build complete'
"

srun --nodes=1 --ntasks=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) ===\"
  $CACHE_ENV_EXPORT
  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  # Unbuffered: stdout here is a redirected log file, not a terminal, so Python block-buffers it
  # and a job that is working looks like a job that has hung.
  export PYTHONUNBUFFERED=1
  python scripts/fetch_flame_checkpoint.py \"\$@\"

  echo \"=== done | \$(date) ===\"
" _ "$@"
