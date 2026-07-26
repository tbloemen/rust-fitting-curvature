#!/bin/sh
# Run the never-executed rms_anchored setting (DISCREPANCIES.md A2): fixes
# R_rms=1 so the search runs directly over kappa. Exp 3's unanchored-vs-
# rms_anchored kappa overlay needs this per hyperbolic dataset; the
# TrialConfig::rms_anchored() path exists but was never swept. Only the
# hyperbolic geometry is meaningful here (rms_anchored gauge-fixes curvature).
#
# Defaults to the 4 real datasets per the DISCREPANCIES.md resolution; pass
# DATASETS to also cover the hyperbolic-verdict synthetics (tree,
# hyperbolic_shells), e.g.:
#   DATASETS="tree hyperbolic_shells" sh slurm/submit_rms_anchored.sh
#
# Reuses run_loss_experiment.sh unmodified via GEOMETRIES=hyperbolic.
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_rms_anchored.sh

set -eu

DATASETS="${DATASETS:-mnist fashion_mnist pbmc wordnet_mammals}"

for ds in $DATASETS; do
  sbatch \
    --job-name="${ds}-rms_anchored" \
    --export=ALL,DATASET="$ds",EXPERIMENT="rms_anchored",GEOMETRIES="hyperbolic" \
    slurm/run_loss_experiment.sh
done
