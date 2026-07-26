#!/bin/sh
# Fill in the missing Euclidean synthetic dataset (DISCREPANCIES.md A4): "grid"
# (a lattice in R^2, thesis §datasets) was never wired into the optimizer's
# `Dataset::load_synthetic`/`--dataset all` list, so it has zero runs across
# every experiment/geometry -- unlike the other 7 datasets, which are already
# fully covered by submit_main.sh (all_off) + submit_all.sh (the other 4
# experiments, hyperbolic+euclidean) + submit_spherical.sh (spherical fill-in).
# This script is the grid-only equivalent of all three, combined.
#
# One sbatch call per experiment; each job loops over its geometries
# internally (see run_loss_experiment.sh). norm_only skips spherical, matching
# the spherical-incompatibility exclusion applied dataset-wide (see
# submit_spherical.sh) -- every other experiment gets all three geometries.
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_grid.sh

set -eu

DATASET="grid"
EXPERIMENTS="all_off centering_only global_only norm_only all_free"

for ex in $EXPERIMENTS; do
  if [ "$ex" = "norm_only" ]; then
    geometries="hyperbolic euclidean"
  else
    geometries="hyperbolic euclidean spherical"
  fi
  sbatch \
    --job-name="${DATASET}-${ex}" \
    --export=ALL,DATASET="$DATASET",EXPERIMENT="$ex",GEOMETRIES="$geometries" \
    slurm/run_loss_experiment.sh
done
