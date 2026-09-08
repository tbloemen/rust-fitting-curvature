#!/bin/sh
# Re-run the N=5000 sweeps to log the origin-free radius `r_gyration`, so kappa
# can be gauged the same way on every arm, and to re-measure `shepard_goodness`
# on its current definition. See slurm/run_5000_rgyr.sh for why.
#
# Covers, into results-rgyr/:
#
#   spherical   8 datasets x 4 settings = 32   (the broken arm)
#   hyperbolic  8 datasets x 6 settings = 48
#   euclidean   8 datasets x 5 settings = 40
#                                       ----
#                                        120 cells
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
# EUCLIDEAN IS NOW INCLUDED, although the gauge never needed fixing there:
# `Euclidean::center` subtracts the coordinate mean every iteration, so
# r_gyration == r_rms exactly and euclidean kappa is 0 on any gauge. It is
# re-run for a second reason that arrived later and applies to every arm --
# `shepard_goodness` changed twice on 2026-09-07. Fractional ranks replaced the
# tie-free Spearman shortcut, and the score is now normalised onto [0, 1] as
# (r_s + 1) / 2 instead of being clipped at 0. Every shepard value under
# results/ is therefore on a scale no current build produces, and none of them
# can be converted after the fact: the clipped zeros lost their sign. Euclidean
# has to be re-fitted for the metric even though it needs nothing from the gauge.
#
# Consequence for reading the output: these cells are NO LONGER bit-for-bit
# twins of their results/ counterparts. `shepard_goodness` is one of the six
# qParEGO objectives, and it moved non-affinely, so the scalarisation, the GP
# proposals and hence the trial sequence all diverge from the original run.
# (The (r_s + 1) / 2 step alone would not have done that -- `scalarize_subset`
# min-max normalises each objective per batch, which absorbs any affine map --
# but the fractional-rank fix does.) Compare cells by their fronts and their
# indicator values, not trial by trial.
#
# COST. Each cell is ~60-72h at N=5000, chunked into 24h jobs, so CHUNKS=4 means
# 120 x 4 = 480 queued jobs. Most sites cap jobs per user well below that. Submit
# in phases with GEOMETRIES, starting with the arm that is actually broken:
#
#   sh slurm/sync.sh                      # then, on the login node:
#   cargo build --release --locked
#   GEOMETRIES=spherical  sh slurm/submit_rgyr_5000.sh    # 32 cells, 128 jobs
#   GEOMETRIES=hyperbolic sh slurm/submit_rgyr_5000.sh    # 48 cells, 192 jobs
#   GEOMETRIES=euclidean  sh slurm/submit_rgyr_5000.sh    # 40 cells, 160 jobs
#
# Before any of them, run ONE cell by hand and check its output (see the plan's
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
GEOMETRIES="${GEOMETRIES:-spherical hyperbolic euclidean}"

# Number of chained ~24h chunks per cell. A full run needs ~3.
CHUNKS="${CHUNKS:-4}"

# Settings per geometry. The sweep is not rectangular: norm_only was never run
# spherically, and rms_anchored is hyperbolic-only (it anchors the hyperboloid's
# radial spread, which has no spherical or euclidean counterpart). Each list is
# the set that geometry actually has under results/, so every cell here has a
# twin -- except rms_anchored x grid, noted above.
settings_for() {
  case "$1" in
  spherical) echo "all_off centering_only global_only all_free" ;;
  hyperbolic) echo "all_off centering_only global_only norm_only all_free rms_anchored" ;;
  euclidean) echo "all_off centering_only global_only norm_only all_free" ;;
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
