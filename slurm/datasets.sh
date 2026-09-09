#!/bin/sh
# The dataset lists every sweep driver shares. Source it, do not copy it:
#
#   . slurm/datasets.sh
#   DATASETS="${DATASETS:-$DATASETS_ALL}"
#
# This file exists because the same eight-name string used to be pasted into ten
# separate scripts, and `grid` was added to the optimizer and to the analysis
# ground truth without any of them noticing -- so no rms_anchored run has ever
# included it. One list, sourced, cannot desync that way.
#
# Keep in step with `crates/optimizer/src/main.rs::SYNTHETIC_DATASETS` and
# `crates/analysis/src/cell.rs::SYNTH_TRUTH`; a test in the analysis crate pins
# those two to each other, but nothing can pin a shell string to Rust.
#
# `slurm/submit_rgyr_5000.sh` deliberately keeps its own list (it skips
# antipodal_clusters and adds grid) -- leave it alone.

DATASETS_REAL="mnist fashion_mnist pbmc wordnet_mammals"

# The original five synthetics. Their generators are unchanged, so results
# already under results/ stay comparable for these.
DATASETS_SYNTH_ORIGINAL="sphere antipodal_clusters tree hyperbolic_shells grid"

# Added for the geometry-matching redesign: a real tree metric, and the matched
# geodesic balls (one sampling scheme mapped into three geometries, at a 2-D and
# a 9-D source tier).
DATASETS_SYNTH_MATCHED="tree_graph ball2_euclidean ball2_spherical ball2_hyperbolic ball9_euclidean ball9_spherical ball9_hyperbolic"

DATASETS_SYNTH="$DATASETS_SYNTH_ORIGINAL $DATASETS_SYNTH_MATCHED"
DATASETS_ALL="$DATASETS_REAL $DATASETS_SYNTH"
