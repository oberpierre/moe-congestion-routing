#!/bin/bash
#SBATCH --job-name=probe-batch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1           # only so the EDF/NGC container's NVIDIA hook can start
#SBATCH --time=00:30:00
#SBATCH --network=disable_rdzv_get
#SBATCH --output=logs/probe_batch_%j.log
#SBATCH --error=logs/probe_batch_%j.log
#
# Extract a frozen probe-batch asset from an IndexedDataset blob.
#
#   source config.sh
#   sbatch scripts/make_probe_batch.sbatch                       # the standing ClimbMix asset
#   sbatch scripts/make_probe_batch.sbatch skewed 16 2048 <prefix>
#   SAMPLING=strided sbatch scripts/make_probe_batch.sbatch      # spread across the held-out split
#
# Args, all optional and positional: ROLE SEQUENCES SEQ_LENGTH DATA_PREFIX. --out is derived as
# assets/probe/<role>_<blob>[_<sampling>]_<S>x<seq_length>.npz, where <blob> is DATA_PREFIX's
# basename, because make_probe_batch.py refuses an --out basename that disagrees with --role.
#
# Set SAMPLING=strided to spread the sampled documents across the whole held-out split instead of
# taking the minimal run at the tail of the blob, which is a few dozen adjacent documents and so
# samples one corpus neighbourhood. The default stays `tail` so an asset extracted before the flag
# existed still re-extracts byte for byte. A strided asset is a DIFFERENT INSTRUMENT with its own
# token_sha256, never a replacement, which is why the name carries the mode and the default does
# not: renaming the existing assets would break every `moe_probe_batch` path pointing at them.
#
# Set FORCE=1 to overwrite an existing asset. Off by default: these are committed, frozen inputs
# that every arm's dumps are keyed to, so replacing one silently invalidates cross-run comparison.
#
# Two things to check with `--show` BEFORE committing the asset it writes:
#   - `data_prefix` in the provenance is the EXPANDED path, so a ${DATA_STORE} that carries a
#     username puts that username inside a committed .npz.
#   - The documents read must sit inside the valid slice that the arm's `split` carves
#     (base_cluster uses "99,1,0", so the last 1%). Read `tail_fraction` on a tail asset and
#     `span_fraction` on a strided one. The script checks against --max-tail-fraction, not against
#     the split, and a probe drawn from train contaminates every measurement built on it.

set -euo pipefail
mkdir -p logs

: "${WORKDIR:?set WORKDIR in config.sh and source it before sbatch}"

ROLE="${1:-standing}"
SEQUENCES="${2:-16}"
SEQ_LENGTH="${3:-2048}"

# Resolved here rather than as a ${4:-...} default: a literal ${DATA_STORE} inside a default ends
# the expansion at its own closing brace, which silently turns the rest into a bash substitution
# (${DATA_STORE/datasets/nvidia}) and reads a path nobody asked for.
if [ -n "${4:-}" ]; then
  DATA_PREFIX="$4"
else
  : "${DATA_STORE:?set DATA_STORE in config.sh, or pass the prefix as the 4th argument}"
  DATA_PREFIX="$DATA_STORE/datasets/nvidia/Nemotron-ClimbMix/climbmix_small/climbmix_small"
fi

SAMPLING="${SAMPLING:-tail}"
BLOB="$(basename "$DATA_PREFIX")"
# The mode is in the name only when it is not the default, so the assets extracted before the flag
# existed keep the names every config already points at, and a reader who has nothing but a
# filename can still tell two instruments apart.
SAMPLING_TAG=""
[ "$SAMPLING" != "tail" ] && SAMPLING_TAG="_${SAMPLING}"
OUT="assets/probe/${ROLE}_${BLOB}${SAMPLING_TAG}_${SEQUENCES}x${SEQ_LENGTH}.npz"
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

DATA_ENV_EXPORT=""
[ -n "${DATA_STORE:-}" ] && DATA_ENV_EXPORT+="export DATA_STORE=$DATA_STORE; "
[ -n "${SCRATCH:-}" ] && DATA_ENV_EXPORT+="export SCRATCH=$SCRATCH; "

# Same cache locations as train_moe_climblab.sbatch, to keep uv off the small home quota.
CACHE_ENV_EXPORT="
  export UV_CACHE_DIR=$WORKDIR/uv-cache
  export UV_PYTHON_INSTALL_DIR=$WORKDIR/uv-python
  export HF_HOME=$WORKDIR/hf_cache
  $DATA_ENV_EXPORT
"

# The venv it needs is the one training builds, so this mirrors train_moe_climblab.sbatch's
# preparation step rather than making a second kind of environment.
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
  # No apply-patches.sh here, unlike the training job since this reads IndexedDataset only
  echo '[env] build complete'
"

srun --nodes=1 --ntasks=1 --environment=alps6 --cpus-per-task "$SLURM_CPUS_PER_TASK" bash -c "
  set -euo pipefail
  echo \"=== node \$(hostname) | \$(date) ===\"
  echo \"=== role=$ROLE sampling=$SAMPLING S=$SEQUENCES seq_length=$SEQ_LENGTH out=$OUT ===\"
  $CACHE_ENV_EXPORT
  cd $WORKDIR
  set +u; source .venv/bin/activate; set -u

  # Unbuffered: stdout here is a redirected log file, not a terminal, so Python block-buffers it
  # and a job that is working looks like a job that has hung.
  export PYTHONUNBUFFERED=1
  python scripts/make_probe_batch.py \
    --data-prefix \"$DATA_PREFIX\" \
    --seq-length $SEQ_LENGTH \
    --sequences $SEQUENCES \
    --role $ROLE \
    --sampling $SAMPLING \
    --out \"$OUT\" $FORCE_FLAG

  echo '=== provenance ==='
  python scripts/make_probe_batch.py --show \"$OUT\"

  echo \"=== done | \$(date) ===\"
"
