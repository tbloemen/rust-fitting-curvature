#!/bin/sh
# 5000-sample all_off sweep. Submits one CHAIN of jobs per (dataset, geometry):
# the geometry loop lives here (not in the run template) so each job covers a
# single geometry, and a single geometry's ~60-72h of work is split across
# several chained ~24h chunks to fit our account's 24h wall-clock cap.
#
# The chunks are linked with `--dependency=afterany`, so each starts once the
# previous one ends (success, timeout, or kill). run_5000.sh passes --resume, so
# each chunk continues from the JSONL the previous chunk left in $HOME; the result
# is bit-identical to a single uninterrupted run. A chunk that finds the run
# already complete exits immediately, so over-provisioning CHUNKS is harmless.
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_main_5000.sh
#   CHUNKS=5 sh slurm/submit_main_5000.sh   # more chunks if jobs hit the wall
#
# The all_off baseline runs all three geometries (matching run_one.sh). Results
# carry an _n5000 marker, so these do not collide with the 1000-sample jobs.

set -eu

DATASETS="mnist fashion_mnist pbmc wordnet_mammals sphere antipodal_clusters tree hyperbolic_shells"
GEOMETRIES="hyperbolic euclidean spherical"

# Number of chained ~24h chunks per (dataset, geometry). A full run needs ~3.
CHUNKS="${CHUNKS:-4}"

for ds in $DATASETS; do
  for geo in $GEOMETRIES; do
    prev=""
    chunk=1
    while [ "$chunk" -le "$CHUNKS" ]; do
      if [ -z "$prev" ]; then
        dep=""
      else
        dep="--dependency=afterany:$prev"
      fi
      jid=$(sbatch --parsable $dep \
        --job-name="${ds}-all_off-${geo}-n5000-c${chunk}" \
        --export=ALL,DATASET="$ds",EXPERIMENT="all_off",GEOMETRY="$geo" \
        slurm/run_5000.sh)
      prev=${jid%%;*} # strip ";cluster" suffix if present
      chunk=$((chunk + 1))
    done
  done
done
