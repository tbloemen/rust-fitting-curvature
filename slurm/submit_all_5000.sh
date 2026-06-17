#!/bin/sh
# 5000-sample loss-experiment sweep. Submits one CHAIN of jobs per
# (dataset, experiment, geometry): the geometry loop lives here (not in the run
# template) so each job covers a single geometry, and a single geometry's
# ~60-72h of work is split across several chained ~24h chunks to fit our
# account's 24h wall-clock cap.
#
# The chunks are linked with `--dependency=afterany`, so each starts once the
# previous one ends (success, timeout, or kill). run_5000.sh passes --resume, so
# each chunk continues from the JSONL the previous chunk left in $HOME; the result
# is bit-identical to a single uninterrupted run. A chunk that finds the run
# already complete exits immediately, so over-provisioning CHUNKS is harmless.
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_all_5000.sh
#   CHUNKS=5 sh slurm/submit_all_5000.sh    # more chunks if jobs hit the wall
#
# Loss experiments use the hyperbolic and euclidean geometries (matching
# run_loss_experiment.sh). Results carry an _n5000 marker, so these do not
# collide with the 1000-sample jobs.

set -eu

DATASETS="mnist fashion_mnist pbmc wordnet_mammals sphere antipodal_clusters tree hyperbolic_shells"
EXPERIMENTS="centering_only global_only norm_only all_free"
GEOMETRIES="hyperbolic euclidean"

# Number of chained ~24h chunks per (dataset, experiment, geometry). ~3 needed.
CHUNKS="${CHUNKS:-4}"

for ds in $DATASETS; do
  for ex in $EXPERIMENTS; do
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
          --job-name="${ds}-${ex}-${geo}-n5000-c${chunk}" \
          --export=ALL,DATASET="$ds",EXPERIMENT="$ex",GEOMETRY="$geo" \
          slurm/run_5000.sh)
        prev=${jid%%;*} # strip ";cluster" suffix if present
        chunk=$((chunk + 1))
      done
    done
  done
done
