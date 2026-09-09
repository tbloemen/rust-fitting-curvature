#!/bin/sh
# Submit one SLURM job per (dataset, experiment) combination -- 4 jobs per dataset.
# Each job loops over all geometries internally (see run_loss_experiment.sh).
# Run on the DelftBlue login node from the repo root, after `cargo build --release --locked`:
#   sh slurm/submit_all.sh
#
# Each sbatch call queues a separate job (its own squeue entry, its own scratch
# results files, copied back to $HOME on exit). Per-job settings are passed via
# --export so run_one.sh stays a single template; SLURM's per-user running-job
# limit throttles concurrency automatically.

set -eu

. ./slurm/datasets.sh
DATASETS="${DATASETS:-$DATASETS_ALL}"
EXPERIMENTS="centering_only global_only norm_only all_free"

for ds in $DATASETS; do
  for ex in $EXPERIMENTS; do
    sbatch \
      --job-name="${ds}-${ex}" \
      --export=ALL,DATASET="$ds",EXPERIMENT="$ex" \
      slurm/run_loss_experiment.sh
  done
done
