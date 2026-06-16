#!/bin/sh
# 5000-sample loss-experiment sweep. Submit one SLURM job per
# (dataset, experiment, geometry) -- the geometry loop lives here (not inside the
# run template) so each 5000-sample job covers a single geometry and stays under
# the 120h compute-partition wall-clock limit (see run_5000.sh).
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_all_5000.sh
#
# Loss experiments use the hyperbolic and euclidean geometries (matching the
# original run_loss_experiment.sh). Results carry an _n5000 marker, so these do
# not collide with the already submitted 1000-sample jobs.

set -eu

DATASETS="mnist fashion_mnist pbmc wordnet_mammals sphere antipodal_clusters tree hyperbolic_shells"
EXPERIMENTS="centering_only global_only norm_only all_free"
GEOMETRIES="hyperbolic euclidean"

for ds in $DATASETS; do
  for ex in $EXPERIMENTS; do
    for geo in $GEOMETRIES; do
      sbatch \
        --job-name="${ds}-${ex}-${geo}-n5000" \
        --export=ALL,DATASET="$ds",EXPERIMENT="$ex",GEOMETRY="$geo" \
        slurm/run_5000.sh
    done
  done
done
