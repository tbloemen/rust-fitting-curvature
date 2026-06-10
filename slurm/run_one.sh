#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-cs

# One job per (dataset, experiment). The job runs every geometry in turn, each
# writing its own results file.
#
# Output is written to /scratch (fast, frequent JSONL appends) and copied back to
# $HOME (30 GB, backed up) on exit. /scratch is NOT backed up and may be purged a
# few days after a clean-up announcement, so the copy-back is what keeps results.
# Input data stays in the repo: it is read once at startup, so it gains nothing
# from scratch.
#
# DATASET/EXPERIMENT can be overridden per-job via `sbatch --export`
# (see submit_all.sh); the defaults below apply when running standalone.

set -eu

DATASET="${DATASET:-mnist}"
EXPERIMENT="${EXPERIMENT:-all_off}"
GEOMETRIES="hyperbolic euclidean spherical"

module load 2025
module load compiler
module load rust

# Build the binary (idempotent). Builds on the compute node, so it links against
# the cluster's glibc. Cargo file-locks target/ and the registry, so concurrent
# jobs serialize on the first build and then no-op. --offline avoids needing
# internet on compute nodes; run `cargo fetch --locked` once on the login node
# first to populate ~/.cargo so the deps are already downloaded.
cargo build --release --locked --offline -p fitting-optimizer

PREFIX=${EXPERIMENT}_${DATASET}
SCRATCH_DIR=/scratch/"$USER"/fitting/results
HOME_DIR="$HOME"/fitting/results

mkdir -p "$SCRATCH_DIR" "$HOME_DIR"

# Copy every result file for this job from scratch back to backed-up home. Runs on
# normal exit, on error (set -e), and on the SIGTERM that --signal delivers 120s
# before the time limit.
sync_back() {
  cp -f "$SCRATCH_DIR"/"$PREFIX"_*.jsonl "$HOME_DIR"/ 2>/dev/null || true
}
trap sync_back EXIT

for GEOMETRY in $GEOMETRIES; do
  OUT="$SCRATCH_DIR"/"$PREFIX"_"$GEOMETRY".jsonl

  # Run in the background so the batch shell can catch SIGTERM while it is still
  # alive (a foreground srun would swallow the signal).
  srun ./target/release/optimizer \
    --mode pareto --dataset "$DATASET" --experiment "$EXPERIMENT" \
    --n-trials 1000 --n-seeds 3 \
    --geometry "$GEOMETRY" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --data-path ./www/public/data \
    --output "$OUT" &
  SRUN_PID=$!

  # On the pre-timeout SIGTERM: save partial results, stop the run, exit.
  trap 'sync_back; kill "$SRUN_PID" 2>/dev/null || true; exit' TERM

  # Don't let one failing geometry abort the remaining ones (set -e would exit).
  wait "$SRUN_PID" || echo "geometry $GEOMETRY failed (exit $?), continuing"
done
