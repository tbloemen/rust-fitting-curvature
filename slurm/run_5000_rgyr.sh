#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=23:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3968MB
#SBATCH --account=education-eemcs-msc-cs

# One 24h chunk of one (dataset, setting, geometry) cell of the N=5000 qParEGO
# sweep the thesis reports. slurm/submit_rgyr_5000.sh chains several of these
# per cell; each continues the previous one with --resume, so a chunk that hits
# the wall-clock limit loses nothing. On SIGTERM the output is copied back from
# scratch to $HOME before the job ends.
#
# --threads is load-bearing: the qParEGO batch size equals the thread count, and
# a different batch size regroups the GP proposals. Keep --cpus-per-task at 48
# to reproduce the thesis results.

set -eu

DATASET="${DATASET:-mnist}"
EXPERIMENT="${EXPERIMENT:-all_off}"
GEOMETRY="${GEOMETRY:-spherical}"
N_SAMPLES=5000

module load 2026
module load compilers
module load rust

cargo build --release --locked --offline -p fitting-optimizer

PREFIX=${EXPERIMENT}_${DATASET}_n${N_SAMPLES}
# The stem the analysis parses is "<prefix>_<geometry>_rgyr"; crates/analysis
# strips the `_rgyr` marker when it reads the cell from the filename.
STEM=${PREFIX}_${GEOMETRY}_rgyr
SCRATCH_DIR=/scratch/"$USER"/fitting/results-rgyr
HOME_DIR="$HOME"/fitting/results-rgyr

mkdir -p "$SCRATCH_DIR" "$HOME_DIR"

OUT="$SCRATCH_DIR"/"$STEM".jsonl

# Restore the previous chunk's checkpoint to scratch so --resume can read it.
# Only fill in files scratch is missing (e.g. after a scratch purge); never
# overwrite an existing scratch file, which may hold trials newer than $HOME if
# the previous chunk was SIGKILLed before its sync_back ran.
for f in "$HOME_DIR"/"$STEM"*; do
  [ -e "$f" ] || continue
  dest="$SCRATCH_DIR"/$(basename "$f")
  [ -e "$dest" ] || cp -f "$f" "$dest"
done

# Copy this job's result files (the JSONL trial log and the _pareto_*.json front)
# from scratch back to backed-up home. Runs on normal exit, on error (set -e), and
# on the SIGTERM SLURM sends at the time limit. This is what lets the next chunk
# in the chain resume.
sync_back() {
  cp -f "$SCRATCH_DIR"/"$STEM"* "$HOME_DIR"/ 2>/dev/null || true
}
trap sync_back EXIT

# Run in the background so the batch shell can catch SIGTERM while it is still
# alive (a foreground srun would swallow the signal). --resume continues from the
# JSONL if it already has trials, or starts fresh if not (so the same invocation
# works for the first chunk and every continuation).
#
# NOTE: --resume must only ever see a file this fixed binary wrote. A reused
# trial re-observes but never rewrites its JSONL line, so resuming a checkpoint
# produced by an older build would leave those replayed trials without
# r_gyration and -- since the 2026-09-07 metric changes -- with their
# `shepard_goodness` still on the old scale, indistinguishable from the new one
# in the same file. Starting in a fresh directory is what guarantees that; if a
# results-rgyr/ from an earlier attempt exists on scratch or in $HOME, move it
# aside before submitting rather than resuming into it.
srun ./target/release/optimizer \
  --mode pareto --dataset "$DATASET" --experiment "$EXPERIMENT" \
  --n-trials 1000 --n-seeds 3 --n-samples "$N_SAMPLES" \
  --geometry "$GEOMETRY" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --data-path ./www/public/data \
  --resume \
  --output "$OUT" &
SRUN_PID=$!

# On the pre-timeout SIGTERM: save partial results, stop the run, exit.
trap 'sync_back; kill "$SRUN_PID" 2>/dev/null || true; exit' TERM

wait "$SRUN_PID" || echo "geometry $GEOMETRY failed (exit $?)"
