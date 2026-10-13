#!/bin/bash
#SBATCH --job-name=price-probe-duals
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/price_probe_duals_%j.log
#SBATCH --error=logs/price_probe_duals_%j.log

# Price one run's probe dumps into a dual store and a bias store, one LP per cell.
#
#   sbatch scripts/run_price_probe_duals.sbatch artifacts/exp1/control/control-trunk \
#          artifacts/exp1/control/control-trunk/duals/duals.csv \
#          artifacts/exp1/control/control-trunk/duals/bias.csv
#
# Resuming is free: the driver reads both stores first and prices only what is missing, so a job
# that runs out of time is continued by submitting it again against the same two output files.
#
# The fleet cadence is three assets per run. Split by asset, one job per asset, and give each job
# its own --bias-out as well as its own --duals-out:
#
#   for a in standing_climbmix_small_16x2048 \
#            standing_climbmix_small_strided_16x2048 \
#            standing_climbmix_small_spread_off1_16x2048; do
#     sbatch scripts/run_price_probe_duals.sbatch <run-dir> \
#            <run-dir>/duals/duals.$a.csv <run-dir>/duals/bias.$a.csv --asset $a
#   done
#
# Each shard needs its own bias file because `BIAS_KEY_FIELDS` carries no `asset`: two assets'
# jobs sharing one --bias-out would write the same key from two processes with no lock.
#
#   scripts/run_triad_trajectory.py --duals .../duals.$a.csv ... --bias .../bias.$a.csv ...
#
# which cross-checks the shards against each other (`dual_store.read_bias_store`) and raises on
# disagreement, rather than an awk join silently picking one.

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

RUN_DIR="${1:?pass a run directory, e.g. artifacts/exp1/control/control-trunk}"
DUALS_OUT="${2:?pass a dual-store output CSV, e.g. <run-dir>/duals/duals.csv}"
BIAS_OUT="${3:?pass a bias-store output CSV, e.g. <run-dir>/duals/bias.csv}"
shift 3

# Anything after the three required arguments is passed through to the driver, so a per-asset
# shard is `... bias.csv --asset standing_climbmix_small_16x2048`. Requoted with %q because the
# srun body is one string the inner shell re-parses, which would otherwise split an argument
# containing a space into two.
EXTRA=""
for arg in "$@"; do
  EXTRA+=" $(printf '%q' "$arg")"
done

# One worker per core, unless memory requires fewer.
MEM_MB="${SLURM_MEM_PER_NODE:-460000}"
MEM_PER_WORKER_MB=1400
MEM_CAP=$(( MEM_MB / MEM_PER_WORKER_MB ))
WORKERS=$(( SLURM_CPUS_PER_TASK < MEM_CAP ? SLURM_CPUS_PER_TASK : MEM_CAP ))
echo "[price-probe-duals] $SLURM_CPUS_PER_TASK cores, ${MEM_MB} MB, memory allows $MEM_CAP -> $WORKERS workers"

if [ ! -f "$WORKDIR/.venv/pyvenv.cfg" ]; then
  echo "[price-probe-duals] no venv at $WORKDIR/.venv." >&2
  echo "  Initialize it once in the run directory." >&2
  exit 1
fi

# --gpus-per-node=4 is kept even though this pass reads dumps and solves LPs entirely on CPU,
# because the alps6 container environment refuses to start without at least one GPU allocated.
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

  python scripts/price_probe_duals.py \
    --run-dir '$RUN_DIR' --duals-out '$DUALS_OUT' --bias-out '$BIAS_OUT' \
    --workers $WORKERS$EXTRA
"
echo "[price-probe-duals] done"
