#!/bin/sh
# Resubmit the 1000-sample (dataset, experiment, geometry) runs that did not
# finish in the original sweeps -- jobs that FAILED (euclidean total-order panic,
# now fixed), TIMED OUT, or were CANCELLED by mistake.
#
# Each entry below is one "dataset experiment geometry" triple submitted via
# run_resubmit.sh, which reruns just that geometry and writes back into the
# existing 1000-sample results set (no _n suffix). Run from the repo root on the
# DelftBlue login node, after `cargo build --release --locked`:
#   sh slurm/resubmit_failed.sh
#
# all_off failures were reported per-geometry, so only the named geometry is
# redone. The loss-experiment failures (all_free / centering_only / global_only)
# were reported at the (dataset, experiment) level, so BOTH geometries those
# experiments use -- hyperbolic and euclidean -- are redone. If hyperbolic
# actually survived a timeout there, just delete its line before submitting.

set -eu

# dataset experiment geometry
JOBS="
mnist all_off euclidean
fashion_mnist all_off spherical
pbmc all_off euclidean
wordnet_mammals all_off euclidean
sphere all_off spherical
antipodal_clusters all_off spherical
tree all_off spherical
hyperbolic_shells all_off spherical
antipodal_clusters all_free hyperbolic
antipodal_clusters all_free euclidean
tree centering_only hyperbolic
tree centering_only euclidean
tree global_only hyperbolic
tree global_only euclidean
"

echo "$JOBS" | while read -r ds ex geo; do
  [ -z "$ds" ] && continue
  sbatch \
    --job-name="${ds}-${ex}-${geo}-resub" \
    --export=ALL,DATASET="$ds",EXPERIMENT="$ex",GEOMETRY="$geo" \
    slurm/run_resubmit.sh
done
