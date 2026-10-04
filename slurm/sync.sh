#!/bin/sh
# Sync the repo to DelftBlue, excluding anything not needed to build or run the
# experiments. Run from the repo root:
#   sh slurm/sync.sh
#
# After syncing, build ON the cluster (login node) so the binary links against the
# cluster's glibc:
#   cargo build --release --locked
#
# Excluded (rebuilt/regenerated on the cluster, or local-only):
#   target/            1.8G  Rust build output -- MUST be built on the cluster
#   .git/               88M  version history, not needed to run
#   www/dist/          107M  compiled web frontend
#   www/node_modules/   25M  JS deps for the frontend
#   www/pkg/                 WASM output
#   results/ plots/         local experiment outputs / figures
#   .venv/                  Python virtualenv (~500M), analysis-only
#   __pycache__/ *.pyc      Python caches
# Kept: crates/ source, Cargo.toml, Cargo.lock, slurm/, www/public/data (datasets).

set -eu

# Edit this to your login-node target (user@host:path).
REMOTE="${REMOTE:-tbloemen@login.delftblue.tudelft.nl:~/rust-fitting-curvature/}"

rsync -av --progress \
  --exclude='/target/' \
  --exclude='/.git/' \
  --exclude='/results/' \
  --exclude='/plots/' \
  --exclude='/.venv/' \
  --exclude='/node_modules/' \
  --exclude='www/dist/' \
  --exclude='www/node_modules/' \
  --exclude='www/pkg/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  ./ "$REMOTE"
