#!/bin/sh
# Resubmit the 8 results-rgyr/ cells that went stale: all_off x spherical, one
# per dataset. These are the only cells in results-rgyr/ missing the
# `distance_consistency` column every other cell there carries -- their
# checkpoint predates that objective being added, so a plain --resume would
# silently carry the old trials forward into a file the analysis then rejects
# as an older sweep without `distance_consistency`.
#
# Covers exactly:
#   all_off  spherical  x  mnist fashion_mnist pbmc wordnet_mammals
#                           sphere tree hyperbolic_shells grid   = 8 cells
#
# Everything else in results-rgyr/ (the other 3 spherical settings, all of
# hyperbolic and euclidean) already has distance_consistency and is untouched.
#
# Run from the repo root on the DelftBlue login node, after syncing and
# building:
#   sh slurm/sync.sh                        # from the local repo root
#   cargo build --release --locked          # on the login node
#   sh slurm/resubmit_rgyr_stale.sh
#
#   CHUNKS=5 sh slurm/resubmit_rgyr_stale.sh   # more chunks if jobs hit the wall
#
# WHAT THIS SCRIPT DOES FIRST, before submitting anything: for each of the 8
# stems it moves any existing checkpoint on scratch and in $HOME out of the
# way, into a `stale/` subdirectory of each -- not deleted, so nothing is lost,
# but out of --resume's sight; slurm/run_5000_rgyr.sh would otherwise resume
# from it. Only these 8 stems are touched; every other cell's checkpoint is
# left alone.

set -eu

DATASETS="${DATASETS:-mnist fashion_mnist pbmc wordnet_mammals sphere tree hyperbolic_shells grid}"
EXPERIMENT=all_off
GEOMETRY=spherical
N_SAMPLES=5000

# Number of chained ~24h chunks per cell, same default as submit_rgyr_5000.sh.
CHUNKS="${CHUNKS:-4}"

SCRATCH_DIR=/scratch/"$USER"/fitting/results-rgyr
HOME_DIR="$HOME"/fitting/results-rgyr
mkdir -p "$SCRATCH_DIR" "$HOME_DIR" "$SCRATCH_DIR"/stale "$HOME_DIR"/stale

echo "Archiving stale checkpoints for $EXPERIMENT x $GEOMETRY..."
for ds in $DATASETS; do
  STEM=${EXPERIMENT}_${ds}_n${N_SAMPLES}_${GEOMETRY}_rgyr
  for dir in "$SCRATCH_DIR" "$HOME_DIR"; do
    for f in "$dir"/"$STEM"*; do
      [ -e "$f" ] || continue
      mv -f "$f" "$dir"/stale/
      echo "  moved $f -> $dir/stale/"
    done
  done
done

submitted=0
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
      --job-name="${ds}-${EXPERIMENT}-${GEOMETRY}-n5000-rgyr-resub-c${chunk}" \
      --export=ALL,DATASET="$ds",EXPERIMENT="$EXPERIMENT",GEOMETRY="$GEOMETRY" \
      slurm/run_5000_rgyr.sh)
    prev=${jid%%;*} # strip ";cluster" suffix if present
    chunk=$((chunk + 1))
  done
  submitted=$((submitted + 1))
done

echo "submitted $submitted cell chains x $CHUNKS chunks for: $EXPERIMENT x $GEOMETRY"
