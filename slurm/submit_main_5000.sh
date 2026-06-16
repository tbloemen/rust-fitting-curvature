#!/bin/sh
# 5000-sample all_off sweep. Submit one SLURM job per (dataset, geometry) -- the
# geometry loop lives here (not inside the run template) so each 5000-sample job
# covers a single geometry and stays under the 120h compute-partition wall-clock
# limit (see run_5000.sh).
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_main_5000.sh
#
# The all_off baseline runs all three geometries (matching the original
# run_one.sh). Results carry an _n5000 marker, so these do not collide with the
# already submitted 1000-sample jobs.

set -eu

DATASETS="mnist fashion_mnist pbmc wordnet_mammals sphere antipodal_clusters tree hyperbolic_shells"
GEOMETRIES="hyperbolic euclidean spherical"

for ds in $DATASETS; do
  for geo in $GEOMETRIES; do
    sbatch \
      --job-name="${ds}-all_off-${geo}-n5000" \
      --export=ALL,DATASET="$ds",EXPERIMENT="all_off",GEOMETRY="$geo" \
      slurm/run_5000.sh
  done
done
