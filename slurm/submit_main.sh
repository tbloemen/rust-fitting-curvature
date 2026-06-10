#!/bin/sh
# Submit one SLURM job per (dataset, experiment) combination -- 5 jobs per dataset.
# Each job loops over all geometries internally (see run_loss_experiments.sh).
# Run on the DelftBlue login node from the repo root, after `cargo build --release --locked`:
#   sh slurm/sumbit_loss_experiments.sh
#
# Each sbatch call queues a separate job (its own squeue entry, its own scratch
# results files, copied back to $HOME on exit). Per-job settings are passed via
# --export so run_one.sh stays a single template; SLURM's per-user running-job
# limit throttles concurrency automatically.

set -eu

DATASETS="mnist fashion_mnist pbmc wordnet_mammals sphere antipodal_clusters tree hyperbolic_shells"

for ds in $DATASETS; do
  sbatch \
    --job-name="${ds}-all_off" \
    --export=ALL,DATASET="$ds" \
    slurm/run_one.sh
done
