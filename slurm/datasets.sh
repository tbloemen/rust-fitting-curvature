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
# `slurm/submit_rgyr_5000.sh` deliberately keeps its own list -- leave it alone.

DATASETS_REAL="mnist fashion_mnist pbmc wordnet_mammals"

# The synthetic suite the thesis reports.
DATASETS_SYNTH="sphere tree hyperbolic_shells grid"
DATASETS_ALL="$DATASETS_REAL $DATASETS_SYNTH"
