#!/bin/sh
# 5000-sample counterpart to submit_rms_anchored.sh (DISCREPANCIES.md A2):
# rms_anchored, hyperbolic geometry, at N=5000. Chains CHUNKS jobs per dataset
# via run_5000.sh, same as submit_all_5000.sh, since a single geometry at
# N=5000 does not fit the 24h wall-clock cap.
#
# Defaults to the 4 real datasets; pass DATASETS to also cover the
# hyperbolic-verdict synthetics (tree, hyperbolic_shells).
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_rms_anchored_5000.sh
#   CHUNKS=5 sh slurm/submit_rms_anchored_5000.sh   # more chunks if jobs hit the wall

set -eu

DATASETS="${DATASETS:-mnist fashion_mnist pbmc wordnet_mammals}"
EXPERIMENT="rms_anchored"
GEOMETRY="hyperbolic"

# Number of chained ~24h chunks per dataset. A full run needs ~3.
CHUNKS="${CHUNKS:-4}"

for ds in $DATASETS; do
  prev=""
  chunk=1
  while [ "$chunk" -le "$CHUNKS" ]; do
    if [ -z "$prev" ]; then
      dep=""
    else
      dep="--dependency=afterany:$prev"
    fi
    jid=$(sbatch --parsable $dep \
      --job-name="${ds}-${EXPERIMENT}-${GEOMETRY}-n5000-c${chunk}" \
      --export=ALL,DATASET="$ds",EXPERIMENT="$EXPERIMENT",GEOMETRY="$GEOMETRY" \
      slurm/run_5000.sh)
    prev=${jid%%;*} # strip ";cluster" suffix if present
    chunk=$((chunk + 1))
  done
done
