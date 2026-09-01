#!/bin/bash
#SBATCH --job-name=lp-contention
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/lp_contention_%j.log
#SBATCH --error=logs/lp_contention_%j.log

# How much does one LP solve slow down when 287 of them run at once on this node?
#
#   sbatch scripts/phi_gap_scaling_probe.sbatch <dump.npz>
#   CPU_BIND=cores sbatch scripts/phi_gap_scaling_probe.sbatch <dump.npz>
#
# Every worker solves the identical instance, so the only thing changing between legs is how many
# run at once. That is the one comparison the grid itself cannot make, because a different worker
# count there also means a different set of cells at a different difficulty.
#
# Submit it twice, once with CPU_BIND unset and once with CPU_BIND=cores, and compare. The
# production grid does no binding today, and on a node with four separate memory attachments an
# unpinned process is migrated away from the memory it allocated.
#
# ONE srun, deliberately. An earlier version ran a separate srun per leg and the second one died
# with "pyxis: Path for EDF image storage must be absolute", because the container environment
# does not survive into a second step. The worker count is a plain argument to the driver, so
# every leg fits inside a single step.

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"
DUMP="${1:?pass one probe dump .npz}"
LAYER="${2:-2}"

BIND_ARG=()
if [ -n "${CPU_BIND:-}" ]; then
  BIND_ARG=(--cpu-bind="$CPU_BIND")
fi
echo "[contention] dump=$DUMP layer=$LAYER cpu-bind=${CPU_BIND:-<none>}"

srun --nodes=1 --ntasks=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" \
     "${BIND_ARG[@]}" bash -c "
  set -euo pipefail
  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  python scripts/lp_contention_probe.py '$DUMP' --layer $LAYER --unit u0 \
    --workers 1 --workers 32 --workers 128 --workers 287
"

echo "[contention] read the slowdown_vs_first_leg column:"
echo "[contention]   flat            -> no contention; the grid's cost is hardware and cell difficulty"
echo "[contention]   rises with N    -> contention; rerun with CPU_BIND=cores to see if pinning fixes it"
