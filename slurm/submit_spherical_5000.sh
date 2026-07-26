#!/bin/sh
# 5000-sample counterpart to submit_spherical.sh (DISCREPANCIES.md A1): the 12
# missing spherical cells (centering_only/global_only/all_free x 4 real
# datasets), at N=5000. Chains CHUNKS jobs per cell via run_5000.sh, same as
# submit_all_5000.sh, since a single geometry at N=5000 does not fit the 24h
# wall-clock cap.
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_spherical_5000.sh
#   CHUNKS=5 sh slurm/submit_spherical_5000.sh   # more chunks if jobs hit the wall

set -eu

DATASETS="mnist fashion_mnist pbmc wordnet_mammals"
EXPERIMENTS="centering_only global_only all_free"
GEOMETRY="spherical"

# Number of chained ~24h chunks per (dataset, experiment). A full run needs ~3.
CHUNKS="${CHUNKS:-4}"

for ds in $DATASETS; do
  for ex in $EXPERIMENTS; do
    prev=""
    chunk=1
    while [ "$chunk" -le "$CHUNKS" ]; do
      if [ -z "$prev" ]; then
        dep=""
      else
        dep="--dependency=afterany:$prev"
      fi
      jid=$(sbatch --parsable $dep \
        --job-name="${ds}-${ex}-${GEOMETRY}-n5000-c${chunk}" \
        --export=ALL,DATASET="$ds",EXPERIMENT="$ex",GEOMETRY="$GEOMETRY" \
        slurm/run_5000.sh)
      prev=${jid%%;*} # strip ";cluster" suffix if present
      chunk=$((chunk + 1))
    done
  done
done
