#!/bin/sh
# Re-run the N=5000 sweeps to log the origin-free radius `r_gyration`, so kappa
# can be gauged the same way on every arm. See slurm/run_5000_rgyr.sh for why.
#
# Covers, into results-rgyr/:
#
#   spherical   8 datasets x 4 settings = 32   (the broken arm)
#   hyperbolic  8 datasets x 6 settings = 48
#                                       ----
#                                         80 cells
#
# TWO DELIBERATE DEPARTURES from the cell grid under results/:
#
#   - `antipodal_clusters` is skipped entirely (see DATASETS below), so its 10
#     existing cells get no re-run and keep the old gauge.
#   - rms_anchored x grid is NEW, with no twin under results/: no rms_anchored
#     run has ever included grid, because every earlier slurm DATASETS list
#     hardcoded the other eight. It is included so rms_anchored covers the same
#     datasets as every other setting.
#
# Everything else is a twin of an existing cell.
#
# Euclidean is deliberately absent too: `Euclidean::center` subtracts the
# coordinate mean every iteration, so r_gyration == r_rms there exactly, and
# euclidean kappa is 0 on any gauge. Re-running its cells would buy nothing.
#
# COST. Each cell is ~60-72h at N=5000, chunked into 24h jobs, so CHUNKS=4 means
# 80 x 4 = 320 queued jobs. Most sites cap jobs per user well below that. Submit
# in phases with GEOMETRIES, starting with the arm that is actually broken:
#
#   sh slurm/sync.sh                      # then, on the login node:
#   cargo build --release --locked
#   GEOMETRIES=spherical  sh slurm/submit_rgyr_5000.sh    # 32 cells, 128 jobs
#   GEOMETRIES=hyperbolic sh slurm/submit_rgyr_5000.sh    # 48 cells, 192 jobs
#
# Before either, run ONE cell by hand and check its output (see the plan's
# staged-submission section):
#   sbatch --export=ALL,DATASET=sphere,EXPERIMENT=all_off,GEOMETRY=spherical \
#     slurm/run_5000_rgyr.sh
#
#   CHUNKS=5 sh slurm/submit_rgyr_5000.sh   # more chunks if jobs hit the wall

set -eu

# `antipodal_clusters` is deliberately excluded -- it is skipped for this re-run,
# not forgotten. Do not "fix" it back in: that is how `grid` came to be missing
# from every rms_anchored run in the first place. Its cells under results/ keep
# the old r_rms gauge and have no results-rgyr/ counterpart.
DATASETS="${DATASETS:-mnist fashion_mnist pbmc wordnet_mammals sphere tree hyperbolic_shells grid}"
GEOMETRIES="${GEOMETRIES:-spherical hyperbolic}"

# Number of chained ~24h chunks per cell. A full run needs ~3.
CHUNKS="${CHUNKS:-4}"

# Settings per geometry. The sweep is not rectangular: norm_only was never run
# spherically, and rms_anchored is hyperbolic-only (it anchors the hyperboloid's
# radial spread, which has no spherical counterpart).
settings_for() {
  case "$1" in
  spherical) echo "all_off centering_only global_only all_free" ;;
  hyperbolic) echo "all_off centering_only global_only norm_only all_free rms_anchored" ;;
  *)
    echo "submit_rgyr_5000.sh: no settings for geometry '$1'" >&2
    exit 1
    ;;
  esac
}

submitted=0
for geo in $GEOMETRIES; do
  for ex in $(settings_for "$geo"); do
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
          --job-name="${ds}-${ex}-${geo}-n5000-rgyr-c${chunk}" \
          --export=ALL,DATASET="$ds",EXPERIMENT="$ex",GEOMETRY="$geo" \
          slurm/run_5000_rgyr.sh)
        prev=${jid%%;*} # strip ";cluster" suffix if present
        chunk=$((chunk + 1))
      done
      submitted=$((submitted + 1))
    done
  done
done

echo "submitted $submitted cell chains x $CHUNKS chunks for: $GEOMETRIES"
