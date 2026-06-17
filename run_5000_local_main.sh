#!/bin/sh
# Launch the MAIN 5000-sample pareto runs locally, without SLURM.
#
# Local (non-SLURM) counterpart of slurm/submit_main_5000.sh: the all_off
# baseline across all three geometries, for each dataset:
#
#   all_off × {hyperbolic, euclidean, spherical} × 8 datasets = 24 runs.
#
# There is no 24h wall-clock cap locally, so there is no chunking: each run goes
# straight to completion. Every run still passes --resume against a stable output
# filename, so this script is safe to interrupt and re-run — finished runs exit
# immediately and a half-finished run continues bit-identically from its JSONL.
#
# Each run already uses all cores (the optimizer's batch size = thread count), so
# the runs are executed one at a time rather than in parallel. At 5000 samples a
# single run takes many hours; the whole matrix is a multi-day job. Trim DATASETS
# / MAIN_GEOMETRIES (or via env) to run a subset.
#
# Each run's combined stdout/stderr is tee'd to a per-run log under LOG_DIR (one
# <experiment>_<dataset>_n5000_<geometry>.log per cell), so progress is reviewable
# after the fact when running unattended.
#
# Usage (from anywhere; the script cd's to the repo root):
#   sh run_5000_local_main.sh
#   DATASETS="tree mnist" sh run_5000_local_main.sh   # subset of datasets
#   THREADS=8 sh run_5000_local_main.sh               # cap worker threads
#   RESULTS_DIR=/data/out sh run_5000_local_main.sh   # where the JSONL lands
#   LOG_DIR=/data/logs sh run_5000_local_main.sh      # where per-run logs land

set -eu

cd "$(dirname "$0")"

DATASETS="${DATASETS:-mnist fashion_mnist pbmc wordnet_mammals sphere antipodal_clusters tree hyperbolic_shells}"
MAIN_GEOMETRIES="${MAIN_GEOMETRIES:-hyperbolic euclidean spherical}"
N_SAMPLES=5000
DATA_PATH="${DATA_PATH:-./www/public/data}"
RESULTS_DIR="${RESULTS_DIR:-./results/n5000}"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Build once up front so the first run doesn't pay for it mid-loop.
echo "==> Building optimizer (release)..."
cargo build --release --locked -p fitting-optimizer
BIN=./target/release/optimizer

# Optional --threads flag; omitted => optimizer uses all logical CPUs.
THREADS_ARG=""
if [ -n "${THREADS:-}" ]; then
  THREADS_ARG="--threads $THREADS"
fi

# Run a single (experiment, dataset, geometry) cell to completion, teeing all
# output (stdout + the optimizer's indicatif progress on stderr) to a per-run
# log. Logs are appended across restarts and prefixed with a timestamp header so
# successive --resume continuations stay distinguishable.
run_cfg() {
  experiment=$1
  dataset=$2
  geometry=$3
  cell="${experiment}_${dataset}_n${N_SAMPLES}_${geometry}"
  out="$RESULTS_DIR/${cell}.jsonl"
  log="$LOG_DIR/${cell}.log"

  echo "==> [$experiment | $dataset | $geometry] -> $out  (log: $log)"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S')  start [$cell] =====" >>"$log"

  # Run in a group so we can capture the optimizer's own exit status (not tee's)
  # while still streaming combined output through tee. Plain `cmd | tee` would
  # report tee's status and mask a failed run.
  status_file=$(mktemp)
  # shellcheck disable=SC2086
  {
    "$BIN" \
      --mode pareto --dataset "$dataset" --experiment "$experiment" \
      --n-trials 1000 --n-seeds 3 --n-samples "$N_SAMPLES" \
      --geometry "$geometry" \
      $THREADS_ARG \
      --data-path "$DATA_PATH" \
      --resume \
      --output "$out"
    echo $? >"$status_file"
  } 2>&1 | tee -a "$log"
  status=$(cat "$status_file")
  rm -f "$status_file"

  if [ "$status" -ne 0 ]; then
    echo "   !! [$experiment | $dataset | $geometry] failed (exit $status)" | tee -a "$log"
  fi
}

for ds in $DATASETS; do
  for geo in $MAIN_GEOMETRIES; do
    run_cfg all_off "$ds" "$geo"
  done
done

echo "==> All main (all_off) 5000-sample runs complete. Results in $RESULTS_DIR"
