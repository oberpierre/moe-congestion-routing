#!/bin/bash
#SBATCH --job-name=phi-gap-grid
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/phi_gap_grid_%j.log
#SBATCH --error=logs/phi_gap_grid_%j.log

# Score one arm's probe dumps against the shared reference game, one LP per cell.
#
#   sbatch scripts/run_phi_gap_grid.sbatch artifacts/exp1/control/control-trunk \
#          assets/results/phi-gap/control.csv
#
# Resuming is free and is the normal way to use this: the driver reads the CSV first and solves
# only what is missing, so a job that runs out of time is continued by submitting it again
# against the same output.
#
# To split an arm across nodes, give each job its own asset and its own output file, because two
# jobs appending to one CSV would interleave rows with no locking:
#
#   for a in standing_climbmix_small_16x2048 \
#            standing_climbmix_small_spread_off1_16x2048 \
#            standing_climbmix_small_strided_16x2048; do
#     sbatch scripts/run_phi_gap_grid.sbatch <run-dir> assets/results/phi-gap/<arm>.$a.csv \
#            --asset $a
#   done
#
# Then concatenate, keeping one header:
#   awk 'FNR==1 && NR!=1 {next} {print}' assets/results/phi-gap/<arm>.*.csv > assets/results/phi-gap/<arm>.csv
#
# Measured at 287 workers: 1208 cells/hour, so a standard arm's 5664 cells take 4.7 h and ALF-LB's
# 9120 take 7.6 h, which is why the limit is 6 h and ALF-LB must be split. Do not add --cpu-bind:
# it measured 3x slower, because one task's mask is inherited by all 287 children rather than
# spreading them over the node's four memory attachments.

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

RUN_DIR="${1:?pass a run directory, e.g. artifacts/exp1/control/control-trunk}"
OUT="${2:?pass an output CSV, e.g. assets/results/phi-gap/control.csv}"
shift 2

# Anything after the two required arguments is passed through to the driver, so a partial sweep
# is `... control.csv --asset standing_climbmix_small_16x2048 --layer 2`. Requoted with %q
# because the srun body is one string the inner shell re-parses, which would otherwise split an
# argument containing a space into two.
EXTRA=""
for arg in "$@"; do
  EXTRA+=" $(printf '%q' "$arg")"
done

# One worker per core, unless memory says fewer. Measured peak resident set for a worker at the
# real shape is 1.48 GB, so 1.6 GB apiece keeps the node off the OOM killer, which would kill
# every in-flight solve at once and lose an hour of work across the pool.
MEM_MB="${SLURM_MEM_PER_NODE:-460000}"
MEM_PER_WORKER_MB=1600
MEM_CAP=$(( MEM_MB / MEM_PER_WORKER_MB ))
WORKERS=$(( SLURM_CPUS_PER_TASK < MEM_CAP ? SLURM_CPUS_PER_TASK : MEM_CAP ))
echo "[grid] $SLURM_CPUS_PER_TASK cores, ${MEM_MB} MB, memory allows $MEM_CAP -> $WORKERS workers"

if [ ! -f "$WORKDIR/.venv/pyvenv.cfg" ]; then
  echo "[grid] no venv at $WORKDIR/.venv." >&2
  echo "  Initialize it once in the run directory." >&2
  exit 1
fi

srun --nodes=1 --ntasks=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  # One thread per worker. numpy and scipy each start a thread pool sized to the whole node by
  # default, so 288 processes would open tens of thousands of threads and spend the run
  # contending rather than solving.
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  python scripts/run_phi_gap_grid.py \
    --run-dir '$RUN_DIR' --out '$OUT' --workers $WORKERS$EXTRA
"
echo "[grid] done"
