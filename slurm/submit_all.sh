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
EXPERIMENTS="centering_only global_only norm_only all_free"
MODE="pareto"

for ds in $DATASETS; do
  for ex in $EXPERIMENTS; do
    sbatch \
      --job-name="${ds}-${ex}" \
      --export=ALL,MODE="$MODE",DATASET="$ds",EXPERIMENT="$ex" \
      slurm/run_loss_experiments.sh
  done
done
