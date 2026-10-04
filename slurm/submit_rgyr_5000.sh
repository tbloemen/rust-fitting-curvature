#!/bin/sh
# Submit the N=5000 qParEGO sweep the thesis reports, into results-rgyr/.
#
#   spherical   8 datasets x 4 settings = 32 cells
#   hyperbolic  8 datasets x 6 settings = 48 cells
#   euclidean   8 datasets x 5 settings = 40 cells
#                                       ---
#                                       120 cells
#
# Each cell takes ~60-72h at N=5000, so it is split into CHUNKS chained ~24h
# jobs that continue with --resume (slurm/run_5000_rgyr.sh); CHUNKS=4 means
# 120 x 4 = 480 queued jobs. Most sites cap jobs per user well below that, so
# submit one geometry at a time:
#
#   sh slurm/sync.sh                      # then, on the login node:
#   cargo build --release --locked
#   GEOMETRIES=spherical  sh slurm/submit_rgyr_5000.sh    # 32 cells, 128 jobs
#   GEOMETRIES=hyperbolic sh slurm/submit_rgyr_5000.sh    # 48 cells, 192 jobs
#   GEOMETRIES=euclidean  sh slurm/submit_rgyr_5000.sh    # 40 cells, 160 jobs
#
# Before any of them, run ONE cell by hand and check its output:
#   sbatch --export=ALL,DATASET=sphere,EXPERIMENT=all_off,GEOMETRY=spherical \
#     slurm/run_5000_rgyr.sh
#
#   CHUNKS=5 sh slurm/submit_rgyr_5000.sh   # more chunks if jobs hit the wall

set -eu

DATASETS="${DATASETS:-mnist fashion_mnist pbmc wordnet_mammals sphere tree hyperbolic_shells grid}"
GEOMETRIES="${GEOMETRIES:-spherical hyperbolic euclidean}"

# Number of chained ~24h chunks per cell. A full run needs ~3.
CHUNKS="${CHUNKS:-4}"

# Settings per geometry. The sweep is not rectangular: norm_only is not run on
# the sphere, and rms_anchored is hyperbolic-only (it anchors the hyperboloid's
# radial spread, which has no spherical or euclidean counterpart).
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
