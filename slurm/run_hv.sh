#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-eemcs-msc-cs

# Monte-Carlo hypervolume, cluster edition. This is the heavy (high --n-mc)
# counterpart of running hv_stats.py locally: a SLURM ARRAY where each task takes
# an interleaved shard of the result cells (--shard = array index) and runs its
# cells across --jobs worker processes. Submit via submit_hv.sh, which sizes the
# array and passes N_MC / N_SHARDS through --export.
#
# Because hv_stats.py seeds each cell deterministically from its name, the shard
# an array task happens to own never changes a cell's HV — sharding is pure
# parallelism. Concatenating the shard files reproduces a single-process run.
#
# Results are READ from ~/fitting/results (where the experiment jobs copy their
# JSONL back to; the repo's results/ is not synced to the cluster). Shard outputs
# are written to /scratch and copied back to ~/fitting/hv on exit.

set -eu

N_MC="${N_MC:-20000000}"
N_SHARDS="${N_SHARDS:-22}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/fitting/results}"
SEED="${SEED:-0}"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# Python + numpy. hv_stats.py / pareto_utils.py are numpy-only. The repo's .venv
# is not synced (sync.sh excludes it), so create it ONCE on the login node:
#   cd ~/rust-fitting-curvature && module load 2025 python && \
#     python -m venv .venv && .venv/bin/pip install numpy
# Compute nodes have no internet, but the venv built on the login node runs fine.
module load 2025
module load python
VENV="$HOME/rust-fitting-curvature/.venv"
if [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
else
  echo "no .venv found at $VENV; falling back to 'python' + module numpy" >&2
  module load py-numpy 2>/dev/null || true
  PY=python
fi

REPO="$HOME/rust-fitting-curvature"
SCRATCH_DIR=/scratch/"$USER"/fitting/hv
HOME_DIR="$HOME"/fitting/hv
mkdir -p "$SCRATCH_DIR" "$HOME_DIR"

OUT="$SCRATCH_DIR"/hv_shard_"$SHARD".jsonl

sync_back() {
  cp -f "$OUT" "$HOME_DIR"/ 2>/dev/null || true
}
trap sync_back EXIT

cd "$REPO"
"$PY" hv_stats.py \
  --results-dir "$RESULTS_DIR" \
  --out "$OUT" \
  --n-mc "$N_MC" \
  --seed "$SEED" \
  --shard "$SHARD" \
  --n-shards "$N_SHARDS" \
  --jobs "$SLURM_CPUS_PER_TASK"
