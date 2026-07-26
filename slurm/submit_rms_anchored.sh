#!/bin/sh
# Run the never-executed rms_anchored setting (DISCREPANCIES.md A2): fixes
# R_rms=1 so the search runs directly over kappa. Exp 3's unanchored-vs-
# rms_anchored kappa overlay needs this per hyperbolic dataset; the
# TrialConfig::rms_anchored() path exists but was never swept. Only the
# hyperbolic geometry is meaningful here (rms_anchored gauge-fixes curvature).
#
# Covers all 8 datasets (matching submit_all.sh/submit_main.sh), not just the
# 4 real ones -- the synthetic hyperbolic fixtures (tree, hyperbolic_shells)
# and the euclidean/spherical-verdict synthetics all feed the analysis too.
#
# Reuses run_loss_experiment.sh unmodified via GEOMETRIES=hyperbolic.
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_rms_anchored.sh

set -eu

DATASETS="${DATASETS:-mnist fashion_mnist pbmc wordnet_mammals sphere antipodal_clusters tree hyperbolic_shells}"

for ds in $DATASETS; do
  sbatch \
    --job-name="${ds}-rms_anchored" \
    --export=ALL,DATASET="$ds",EXPERIMENT="rms_anchored",GEOMETRIES="hyperbolic" \
    slurm/run_loss_experiment.sh
done
