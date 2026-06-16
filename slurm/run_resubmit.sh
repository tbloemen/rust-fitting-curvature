#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-cs

# Per-geometry rerun of a single 1000-sample (dataset, experiment, geometry).
#
# Used by resubmit_failed.sh to re-run ONLY the geometries that FAILED, TIMED OUT,
# or were CANCELLED in the original all_off / loss-experiment sweeps -- without
# touching the geometries that already completed. Unlike run_one.sh /
# run_loss_experiment.sh (which loop every geometry inside one job), this runs
# exactly ONE geometry, passed via `sbatch --export`.
#
# It writes the SAME result filenames as the original 1000-sample runs
# (${EXPERIMENT}_${DATASET}_${GEOMETRY}.jsonl, with NO _n suffix), so the reruns
# drop straight back into the existing 1000-sample results set.
#
# A single geometry at 1000 samples is ~2.4h (sacct-measured) and peaks at ~2.2G
# RSS, so 5h / 3G-per-cpu is ample headroom.
#
# Relies on the committed total_cmp / objective-sanitisation fix: the euclidean
# runs that previously panicked now complete (the build step below picks it up).

set -eu

DATASET="${DATASET:-mnist}"
EXPERIMENT="${EXPERIMENT:-all_off}"
GEOMETRY="${GEOMETRY:-hyperbolic}"
N_SAMPLES="${N_SAMPLES:-1000}"

module load 2025
module load compiler
module load rust

cargo build --release --locked --offline -p fitting-optimizer

PREFIX=${EXPERIMENT}_${DATASET}
SCRATCH_DIR=/scratch/"$USER"/fitting/results
HOME_DIR="$HOME"/fitting/results

mkdir -p "$SCRATCH_DIR" "$HOME_DIR"

OUT="$SCRATCH_DIR"/"$PREFIX"_"$GEOMETRY".jsonl

# The optimizer APPENDS to its output (and writes a _pareto_*.json front). The
# original failed/cancelled run may have left a partial file for this geometry on
# scratch; remove it so this rerun starts clean instead of appending to stale rows.
rm -f "$SCRATCH_DIR"/"$PREFIX"_"$GEOMETRY"*

# Copy this geometry's fresh result files back to backed-up home, overwriting the
# partial copies left by the failed run. Runs on exit, on error, and on SIGTERM.
sync_back() {
  cp -f "$SCRATCH_DIR"/"$PREFIX"_"$GEOMETRY"* "$HOME_DIR"/ 2>/dev/null || true
}
trap sync_back EXIT

# Run in the background so the batch shell can catch the pre-timeout SIGTERM.
srun ./target/release/optimizer \
  --mode pareto --dataset "$DATASET" --experiment "$EXPERIMENT" \
  --n-trials 1000 --n-seeds 3 --n-samples "$N_SAMPLES" \
  --geometry "$GEOMETRY" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --data-path ./www/public/data \
  --output "$OUT" &
SRUN_PID=$!

trap 'sync_back; kill "$SRUN_PID" 2>/dev/null || true; exit' TERM

wait "$SRUN_PID" || echo "geometry $GEOMETRY failed (exit $?)"
