#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=23:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3968MB
#SBATCH --account=education-eemcs-msc-cs

# 5000-sample variant of run_one.sh / run_loss_experiment.sh. Runs a SINGLE
# geometry; DATASET/EXPERIMENT/GEOMETRY are passed per-job via `sbatch --export`
# (see submit_all_5000.sh / submit_main_5000.sh). The defaults below apply when
# running standalone.
#
# WALL-CLOCK: our account caps compute-partition jobs at 24h (NOT the 120h that
# applies with full resource access). At 5000 samples the work is dominated by
# O(n^2) terms (pairwise distances, the t-SNE gradient step, the metrics), so a
# single (dataset, experiment, geometry) job is ~25-28x the 1000-sample run that
# took ~2.4h -> ~60-72h, which does NOT fit in one 24h job.
#
# CHUNKING + RESUME: the qParEGO search is sequential (each batch is proposed
# from a GP refit on all prior trials), so it cannot be split across parallel
# jobs without changing the result. Instead each job is one ~24h CHUNK and the
# submit scripts chain several of them with `--dependency=afterany`. Passing
# `--resume` makes the optimizer replay the trials already in the output JSONL
# (re-deriving their suggestions but reusing the recorded metrics, skipping the
# expensive embeddings) and then continue. The chained run is bit-identical to a
# single uninterrupted 60-72h run. A chunk that finds the run already complete
# exits immediately, so over-provisioning the chain is free.
#
# At the 24h limit SLURM sends SIGTERM (then SIGKILL after the grace period); the
# TERM trap below copies partial results back to $HOME so the next chunk resumes.
# Even without that, every trial is flushed to the JSONL as it finishes, so the
# next chunk loses at most the in-flight batch.
#
# Sizing: mem ~55G total (~1.2G/cpu); --mem-per-cpu=3968MB (~186G) is a generous
# cushion. Result filenames carry an _n5000 marker so they never collide with the
# already submitted 1000-sample runs.
#
# Output lives on /scratch (fast, frequent JSONL appends) and is copied back to
# $HOME (backed up) on exit; /scratch is NOT backed up and may be purged. Input
# data stays in the repo: it is read once at startup.

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

# Restore the previous chunk's checkpoint to scratch so --resume can read it.
# Only fill in files scratch is missing (e.g. after a scratch purge); never
# overwrite an existing scratch file, which may hold trials newer than $HOME if
# the previous chunk was SIGKILLed before its sync_back ran.
for f in "$HOME_DIR"/"$PREFIX"_"$GEOMETRY"*; do
  [ -e "$f" ] || continue
  dest="$SCRATCH_DIR"/$(basename "$f")
  [ -e "$dest" ] || cp -f "$f" "$dest"
done

# Copy this job's result files (the JSONL trial log and the _pareto_*.json front)
# from scratch back to backed-up home. Runs on normal exit, on error (set -e), and
# on the SIGTERM SLURM sends at the time limit. This is what lets the next chunk
# in the chain resume.
sync_back() {
  cp -f "$SCRATCH_DIR"/"$PREFIX"_"$GEOMETRY"* "$HOME_DIR"/ 2>/dev/null || true
}
trap sync_back EXIT

# Run in the background so the batch shell can catch SIGTERM while it is still
# alive (a foreground srun would swallow the signal). --resume continues from the
# JSONL if it already has trials, or starts fresh if not (so the same invocation
# works for the first chunk and every continuation).
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
