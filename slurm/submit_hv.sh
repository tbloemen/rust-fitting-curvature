#!/bin/sh
# Submit the Monte-Carlo hypervolume computation as a single SLURM ARRAY job.
# Each array task is one shard (see run_hv.sh); the array width is N_SHARDS.
#
# One-time setup on the login node (compute nodes have no internet):
#   cd ~/rust-fitting-curvature && module load 2025 python && \
#     python -m venv .venv && .venv/bin/pip install numpy
# and make sure the experiment results are in ~/fitting/results (the sweep jobs
# copy them there; or rsync your local results/ up).
#
# Run from the repo root on the login node:
#   sh slurm/submit_hv.sh
#   N_MC=50000000 N_SHARDS=44 sh slurm/submit_hv.sh   # tighter SE, wider array
#
# When every shard has finished, pull ~/fitting/hv back and aggregate locally:
#   rsync -av <user>@login.delftblue.tudelft.nl:~/fitting/hv/ ./hv/
#   uv run python hv_aggregate.py hv/hv_shard_*.jsonl --csv hv_delta.csv

set -eu

N_MC="${N_MC:-20000000}"
N_SHARDS="${N_SHARDS:-22}"
SEED="${SEED:-0}"

LAST=$((N_SHARDS - 1))

sbatch \
  --job-name="hv-mc" \
  --array=0-"$LAST" \
  --export=ALL,N_MC="$N_MC",N_SHARDS="$N_SHARDS",SEED="$SEED" \
  slurm/run_hv.sh
