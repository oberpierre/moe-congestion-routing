#!/bin/bash
#SBATCH --job-name=phi-gap-scaling
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/phi_gap_scaling_%j.log
#SBATCH --error=logs/phi_gap_scaling_%j.log

# Measure how the phi-gap grid's per-cell cost changes with the worker count, on this node.
#
#   sbatch scripts/phi_gap_scaling_probe.sbatch artifacts/exp1/control/control-trunk
#
# Why this exists. A full arm measured 856 s per cell per worker at 287 workers, against 77 s per
# cell per worker on a 4-worker desktop. That 11x mixes three causes that cannot be separated from
# one number: this node's per-core speed, the difficulty of the particular cells the grid reached,
# and contention between 287 concurrent simplex solves whose working sets are about 1.1 GB each
# against roughly 114 MB of last-level cache per Grace module. Only the third is fixable, so the
# ratio between legs A and D below is what decides whether the fleet's 54 h is reducible.
#
# Each leg solves two cells per worker, so every leg takes about the same wall clock whatever its
# worker count, and the reported seconds-per-cell-per-worker is directly comparable across legs.
# Legs write to separate scratch CSVs, because a shared one would let resume skip the later legs
# and measure nothing.

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"
RUN_DIR="${1:?pass a run directory, e.g. artifacts/exp1/control/control-trunk}"

SCRATCH="artifacts/phi_gap/scaling_probe_${SLURM_JOB_ID:-manual}"

run_leg () {
  local name="$1" workers="$2" bind="$3"
  local limit=$(( workers * 2 ))
  local out="$SCRATCH/${name}.csv"
  echo "[scaling] leg $name: $workers workers, $limit cells, cpu-bind=$bind"
  local start
  start=$(date +%s)
  srun --nodes=1 --ntasks=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" \
       --cpu-bind="$bind" bash -c "
    set -euo pipefail
    cd $WORKDIR
    set +u; source .venv/bin/activate; set -u
    mkdir -p $SCRATCH
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    python scripts/run_phi_gap_grid.py \
      --run-dir '$RUN_DIR' --out '$out' --workers $workers --limit $limit
  "
  local elapsed=$(( $(date +%s) - start ))
  # Seconds of worker time per cell, which is the quantity that is flat when scaling is perfect.
  echo "[scaling] RESULT leg=$name workers=$workers cells=$limit wall=${elapsed}s" \
       "per_cell_per_worker=$(( elapsed * workers / limit ))s"
}

# A is the uncontended baseline: one solve at a time on this hardware, on the same cells the
# other legs start from, so it isolates per-core speed and cell difficulty from contention.
run_leg A_serial     1   none
run_leg B_half     144   none
run_leg C_full     287   none
# D repeats C with processes pinned, which the production sbatch does not do today. On a node with
# four memory attachments an unpinned process is migrated away from the memory it allocated.
run_leg D_full_bound 287 cores

echo "[scaling] done. Compare per_cell_per_worker across legs:"
echo "[scaling]   A == C          -> no contention, the cost is hardware and cell difficulty"
echo "[scaling]   C >> A, D << C  -> binding is the fix, add --cpu-bind=cores to the grid sbatch"
echo "[scaling]   C >> A, D ~= C  -> memory-bound; run fewer workers per node and more nodes"
