#!/bin/sh
# Fill in the spherical cells missing from the main loss-ablation sweep
# (DISCREPANCIES.md A1): submit_all.sh only ran centering_only/global_only/
# norm_only/all_free for hyperbolic+euclidean (see run_loss_experiment.sh's
# default GEOMETRIES), so spherical exists only for all_off. norm_only is
# skipped on purpose -- it is spherical-incompatible (thesis tab:loss-ablations)
# -- leaving 3 experiments x 8 datasets = 24 missing cells. Covers all 8
# datasets (not just the 4 real ones behind Exp 2's headline count), since the
# synthetic spherical fixtures (sphere, antipodal_clusters) feed other analyses.
#
# Reuses run_loss_experiment.sh unmodified via GEOMETRIES=spherical, so this is
# a single extra sbatch call per cell rather than a duplicate runner.
#
# Run on the DelftBlue login node from the repo root, after
# `cargo build --release --locked`:
#   sh slurm/submit_spherical.sh

set -eu

DATASETS="mnist fashion_mnist pbmc wordnet_mammals sphere antipodal_clusters tree hyperbolic_shells"
EXPERIMENTS="centering_only global_only all_free"

for ds in $DATASETS; do
  for ex in $EXPERIMENTS; do
    sbatch \
      --job-name="${ds}-${ex}-spherical" \
      --export=ALL,DATASET="$ds",EXPERIMENT="$ex",GEOMETRIES="spherical" \
      slurm/run_loss_experiment.sh
  done
done
