#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-msc-cs

# Curvature detection (κ_data export) for every dataset at one sample size.
# Runs `optimizer --mode detect --dataset all`, which fans the datasets out
# across the cores (one dataset per outer worker), each writing one JSONL line
# with its Wilson fits + Gromov δ(k) diagnostics + κ_data. No embedding is fit,
# but the Wilson power-iteration over the n×n Gram matrix is O(n²) per step, so
# n=5000 is ~25x the n=1000 cost — hence this runs on the cluster.
#
# NSAMPLES is passed per-job via `sbatch --export` (see submit_detect.sh); the
# default below applies when running standalone. Output goes to /scratch and is
# copied back to ~/fitting/results on exit, matching the sweep jobs.

set -eu

NSAMPLES="${NSAMPLES:-5000}"

module load 2025
module load compiler
module load rust

cargo build --release --locked --offline -p fitting-optimizer

SCRATCH_DIR=/scratch/"$USER"/fitting/results
HOME_DIR="$HOME"/fitting/results
mkdir -p "$SCRATCH_DIR" "$HOME_DIR"

OUT="$SCRATCH_DIR"/kappa_data_n"$NSAMPLES".jsonl
# Fresh file each run: detect appends, so remove any partial from a prior attempt
# to avoid duplicate dataset lines.
rm -f "$OUT"

sync_back() {
  cp -f "$OUT" "$HOME_DIR"/ 2>/dev/null || true
}
trap sync_back EXIT

srun ./target/release/optimizer \
  --mode detect --dataset all \
  --n-samples "$NSAMPLES" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --data-path ./www/public/data \
  --output "$OUT"
