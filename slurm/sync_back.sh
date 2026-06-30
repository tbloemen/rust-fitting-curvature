#!/bin/sh
# Pull experiment outputs back from DelftBlue into the local repo's results/.
# Run from the repo root:
#   sh slurm/sync_back.sh
#
# Note the result path is NOT inside the repo on the cluster: the SLURM jobs
# write to ~/fitting/results (backed-up home), with the live working copy on
# /scratch/$USER/fitting/results. The home copy is the canonical, backed-up one,
# so that is what we pull. Pass REMOTE_USER/REMOTE_HOST or override REMOTE_RESULTS
# to point elsewhere (e.g. straight at scratch).

set -eu

REMOTE_USER="${REMOTE_USER:-tbloemen}"
REMOTE_HOST="${REMOTE_HOST:-login.delftblue.tudelft.nl}"
# Backed-up home copy. To grab the freshest in-flight data instead, run:
#   REMOTE_RESULTS=/scratch/$REMOTE_USER/fitting/results sh slurm/sync_back.sh
REMOTE_RESULTS="${REMOTE_RESULTS:-~/fitting/results}"

rsync -av --progress \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_RESULTS/" ./results/
