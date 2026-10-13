#!/bin/bash
#SBATCH --job-name=kappa
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/kappa_%j.log
#SBATCH --error=logs/kappa_%j.log

# Run either kappa driver under one sbatch, since both solve the same shape of LP (one
# capacity-constrained assignment at n=16,384 per cell) and differ only in which cells they cover.
#
#   sbatch scripts/run_kappa.sbatch scripts/run_kappa_trajectory.py \
#       --out assets/results/price-recovery/kappa-trajectory_corrected_21-steps.csv \
#       --project-code-axis
#
#   sbatch scripts/run_kappa.sbatch scripts/run_triad.py \
#       --out assets/results/price-recovery/triad_step500_corrected.csv --project-code-axis
#
# Neither driver resumes from a partial CSV, so a job that runs out of time is restarted from
# scratch against a fresh --out rather than continued.

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

DRIVER="${1:?pass a driver script, e.g. scripts/run_kappa_trajectory.py or scripts/run_triad.py}"
shift 1

# Anything after the driver is passed through to it, so a corrected run is `... --out X.csv
# --project-code-axis`. Requoted with %q rather than passed as $*, because the srun body below is
# one string the inner shell re-parses, which would otherwise split an argument containing a
# space into two.
EXTRA=""
for arg in "$@"; do
  EXTRA+=" $(printf '%q' "$arg")"
done

# One worker per core, unless memory says fewer. Measured peak resident set for one priced_unit
# LP at n=16,384 is ~1.09 GB, so 1.2 GB apiece keeps the node off the OOM killer, which would kill
# every in-flight solve at once and lose the run.
MEM_MB="${SLURM_MEM_PER_NODE:-460000}"
MEM_PER_WORKER_MB=1200
MEM_CAP=$(( MEM_MB / MEM_PER_WORKER_MB ))
WORKERS=$(( SLURM_CPUS_PER_TASK < MEM_CAP ? SLURM_CPUS_PER_TASK : MEM_CAP ))
echo "[kappa] $SLURM_CPUS_PER_TASK cores, ${MEM_MB} MB, memory allows $MEM_CAP -> $WORKERS workers"

# Append --jobs only when the caller did not already pass one, so an explicit --jobs from the
# caller is never overridden by the computed worker count.
HAS_JOBS=0
for arg in "$@"; do
  case "$arg" in
    --jobs|--jobs=*) HAS_JOBS=1 ;;
  esac
done
if [ "$HAS_JOBS" -eq 0 ]; then
  EXTRA+=" --jobs $(printf '%q' "$WORKERS")"
fi

if [ ! -f "$WORKDIR/.venv/pyvenv.cfg" ]; then
  echo "[kappa] no venv at $WORKDIR/.venv." >&2
  echo "  Initialize it once in the run directory." >&2
  exit 1
fi

srun --nodes=1 --ntasks=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  # One thread per worker. numpy and scipy each start a thread pool sized to the whole node by
  # default, so as many processes as cores would open far more threads than cores and spend the
  # run contending rather than solving.
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  python '$DRIVER'$EXTRA
"
echo "[kappa] done"
