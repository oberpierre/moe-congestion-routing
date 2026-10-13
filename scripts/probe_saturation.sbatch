#!/bin/bash
#SBATCH --job-name=probe-saturation
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1           # only so the EDF/NGC container's NVIDIA hook can start
#SBATCH --time=00:30:00
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/probe_saturation_%j.log
#SBATCH --error=logs/probe_saturation_%j.log
#
# Reduce probe dumps to router-gate statistics, one CSV for every run named.
#
#   source config.sh
#   sbatch scripts/probe_saturation.sbatch out.csv <run_dir> [<run_dir> ...]
#   ASSET=standing_climbmix_small_strided_16x2048 sbatch scripts/probe_saturation.sbatch out.csv <run_dir>
#   EVERY=2 MAX_ITER=1500 sbatch scripts/probe_saturation.sbatch out.csv <run_dir>
#
# ASSET applies to EVERY run named in one call, so runs whose dumps carry different asset stems
# need one call each and the CSVs concatenated afterwards.
#
# **No environment build and no apply-patches step, unlike every other job here.** The reducer
# imports numpy and this package only, verified with an import-closure check that finds neither
# `megatron` nor `torch` in `sys.modules`. Building the venv would hard-reset both submodules,
# which is the hazard train_moe_climblab.sbatch made opt-in, and this job has no reason to pay it.

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

OUT="${1:?pass the output CSV path, e.g. router_saturation.csv}"
shift
[ "$#" -ge 1 ] || { echo "pass at least one run directory containing probes/" >&2; exit 1; }
RUN_DIRS="$*"

EXTRA=""
[ -n "${ASSET:-}" ] && EXTRA+=" --asset $ASSET"
[ -n "${EVERY:-}" ] && EXTRA+=" --every $EVERY"
[ -n "${MAX_ITER:-}" ] && EXTRA+=" --max-iter $MAX_ITER"

# Read-only preflight. The venv is the one training builds, and this job never writes to it.
if [ ! -f "$WORKDIR/.venv/pyvenv.cfg" ]; then
  echo "[env] no venv at $WORKDIR/.venv." >&2
  echo "  Build it once: MOE_PREPARE_ENV=1 sbatch scripts/train_moe_climblab.sbatch <config>" >&2
  exit 1
fi

srun --nodes=1 --ntasks=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) ===\"
  echo \"=== asset=${ASSET:-<one>} out=$OUT runs=$RUN_DIRS ===\"
  export UV_CACHE_DIR=$WORKDIR/uv-cache
  export UV_PYTHON_INSTALL_DIR=$WORKDIR/uv-python
  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  # Unbuffered, because stdout is a redirected log here and Python would block-buffer it, so a
  # job that is working would look like a job that has hung.
  export PYTHONUNBUFFERED=1
  python scripts/probe_saturation.py $RUN_DIRS --out \"$OUT\"$EXTRA

  echo \"=== rows written: \$(( \$(wc -l < \"$OUT\") - 1 )) ===\"
  echo \"=== done | \$(date) ===\"
"
