#!/bin/sh
# Submit the curvature-detection (κ_data) export as one job per sample size.
# Each job runs every dataset at that N (see run_detect.sh) and writes
# kappa_data_n<N>.jsonl, copied back to ~/fitting/results.
#
# The n=1000 detection is cheap enough to run locally
#   ./target/release/optimizer --mode detect --dataset all --n-samples 1000 \
#     --output results/kappa_data.jsonl --threads 8
# so this is mainly for the n=5000 export. Both are submitted by default.
#
# Run from the repo root on the login node, after `cargo build --release --locked`:
#   sh slurm/submit_detect.sh
#   NSAMPLES="5000" sh slurm/submit_detect.sh   # just the n=5000 export

set -eu

NSAMPLES="${NSAMPLES:-1000 5000}"

for n in $NSAMPLES; do
  sbatch \
    --job-name="detect-n${n}" \
    --export=ALL,NSAMPLES="$n" \
    slurm/run_detect.sh
done
