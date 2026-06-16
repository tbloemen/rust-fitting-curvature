#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=84:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-msc-cs

# 5000-sample variant of run_one.sh / run_loss_experiment.sh.
#
# At 5000 samples the dominant work is O(n^2) (pairwise distances, the t-SNE
# gradient step, the metrics), so cost is ~25-28x the 1000-sample jobs. To keep
# each job well under the 120h compute-partition wall-clock limit, the
# per-geometry loop has been split out to the submit scripts: this template runs
# a SINGLE geometry. DATASET/EXPERIMENT/GEOMETRY are passed per-job via
# `sbatch --export` (see submit_all_5000.sh / submit_main_5000.sh); the defaults
# below apply when running standalone.
#
# Sizing is calibrated from the 1000-sample sacct numbers: a single geometry
# (1000 trials x 3 seeds) took ~2.4h wall and ~2.2G peak RSS. Scaled ~25-28x:
#   time  ~60-72h  -> --time=84:00:00 (margin + good queue priority, << 120h cap)
#   mem   ~55G total (~1.2G/cpu) -> --mem-per-cpu=4G (192G) is a deliberate ~3.5x
#         cushion; 2-3G/cpu would also be safe.
#
# Output is written to /scratch (fast, frequent JSONL appends) and copied back to
# $HOME (30 GB, backed up) on exit. /scratch is NOT backed up and may be purged a
# few days after a clean-up announcement, so the copy-back is what keeps results.
# Input data stays in the repo: it is read once at startup, so it gains nothing
# from scratch.
#
# Result filenames carry an _n5000 marker so they never collide with the already
# submitted 1000-sample runs.

set -eu

DATASET="${DATASET:-mnist}"
EXPERIMENT="${EXPERIMENT:-all_off}"
GEOMETRY="${GEOMETRY:-hyperbolic}"
N_SAMPLES=5000

module load 2025
module load compiler
module load rust

# Build the binary (idempotent). Builds on the compute node, so it links against
# the cluster's glibc. Cargo file-locks target/ and the registry, so concurrent
# jobs serialize on the first build and then no-op. --offline avoids needing
# internet on compute nodes; run `cargo fetch --locked` once on the login node
# first to populate ~/.cargo so the deps are already downloaded.
cargo build --release --locked --offline -p fitting-optimizer

PREFIX=${EXPERIMENT}_${DATASET}_n${N_SAMPLES}
SCRATCH_DIR=/scratch/"$USER"/fitting/results
HOME_DIR="$HOME"/fitting/results

mkdir -p "$SCRATCH_DIR" "$HOME_DIR"

OUT="$SCRATCH_DIR"/"$PREFIX"_"$GEOMETRY".jsonl

# Copy this job's result files (the JSONL trial log and the _pareto_*.json front)
# from scratch back to backed-up home. Runs on normal exit, on error (set -e), and
# on the SIGTERM that --signal delivers before the time limit.
sync_back() {
  cp -f "$SCRATCH_DIR"/"$PREFIX"_"$GEOMETRY"* "$HOME_DIR"/ 2>/dev/null || true
}
trap sync_back EXIT

# Run in the background so the batch shell can catch SIGTERM while it is still
# alive (a foreground srun would swallow the signal).
srun ./target/release/optimizer \
  --mode pareto --dataset "$DATASET" --experiment "$EXPERIMENT" \
  --n-trials 1000 --n-seeds 3 --n-samples "$N_SAMPLES" \
  --geometry "$GEOMETRY" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --data-path ./www/public/data \
  --output "$OUT" &
SRUN_PID=$!

# On the pre-timeout SIGTERM: save partial results, stop the run, exit.
trap 'sync_back; kill "$SRUN_PID" 2>/dev/null || true; exit' TERM

wait "$SRUN_PID" || echo "geometry $GEOMETRY failed (exit $?)"
