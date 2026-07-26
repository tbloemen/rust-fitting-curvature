#!/bin/sh
# 5000-sample counterpart to submit_grid.sh (DISCREPANCIES.md A4): the "grid"
# Euclidean synthetic across every experiment/geometry cell, at N=5000. Chains
# CHUNKS jobs per cell via run_5000.sh, same as submit_main_5000.sh /
# submit_all_5000.sh / submit_spherical_5000.sh, since a single geometry at
# N=5000 does not fit the 24h wall-clock cap.
#
# norm_only skips spherical, matching the spherical-incompatibility exclusion
# applied dataset-wide -- every other experiment chains all three geometries.
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_grid_5000.sh
#   CHUNKS=5 sh slurm/submit_grid_5000.sh   # more chunks if jobs hit the wall

set -eu

DATASET="grid"
EXPERIMENTS="all_off centering_only global_only norm_only all_free"

# Number of chained ~24h chunks per (experiment, geometry). A full run needs ~3.
CHUNKS="${CHUNKS:-4}"

for ex in $EXPERIMENTS; do
  if [ "$ex" = "norm_only" ]; then
    geometries="hyperbolic euclidean"
  else
    geometries="hyperbolic euclidean spherical"
  fi
  for geo in $geometries; do
    prev=""
    chunk=1
    while [ "$chunk" -le "$CHUNKS" ]; do
      if [ -z "$prev" ]; then
        dep=""
      else
        dep="--dependency=afterany:$prev"
      fi
      jid=$(sbatch --parsable $dep \
        --job-name="${DATASET}-${ex}-${geo}-n5000-c${chunk}" \
        --export=ALL,DATASET="$DATASET",EXPERIMENT="$ex",GEOMETRY="$geo" \
        slurm/run_5000.sh)
      prev=${jid%%;*} # strip ";cluster" suffix if present
      chunk=$((chunk + 1))
    done
  done
done
