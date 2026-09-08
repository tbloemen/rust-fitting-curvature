#!/bin/sh
#SBATCH --partition=compute
#SBATCH --time=23:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3968MB
#SBATCH --account=education-eemcs-msc-cs

# run_5000.sh, re-run to log the origin-free radius `r_gyration`.
#
# WHY THIS EXISTS. The sweeps gauge curvature as kappa = |K| * R^2, with R read
# off `distances_from_origin` -- the geodesic radius from the manifold's fixed
# origin. That is sound on the hyperboloid and in Euclidean space, where
# `center()` puts the configuration's mean at that origin every iteration. It is
# vacuous on the sphere: `Sphere::center` is a no-op, and `lift_pca_to_manifold`
# writes the constrained coordinate to the LAST ambient slot while
# `Sphere::distances_from_origin` reads the FIRST, so PCA init lands every point
# ~90 degrees from the pole kappa is gauged against. |K|*r_rms^2 then sits at
# pi^2/4 = 2.4674 however curved the space is -- 68% of the 71,839 spherical
# trials in results/ are at exactly that value.
#
# The fix is an origin-free measure: the radius of gyration over the pairwise
# geodesics, logged as `r_gyration` alongside the untouched `r_rms`. Embeddings
# are not persisted, only the scalar columns, so the only way to obtain it for
# past trials is to re-fit them. Hence this job.
#
# WHY HYPERBOLIC IS RE-RUN TOO, not just the broken arm: the two measures agree
# only in FLAT space. On the hyperboloid, centred or not, r_gyration/r_rms grows
# with radius -- 1.04 at geodesic radius 1, 1.11 at 2, 1.25 at 5 -- and kappa
# squares that. Gauging the spherical arm one way and the hyperbolic arm the
# other would put two different quantities on one axis, which is exactly what
# "one kappa, one gauge" exists to prevent.
#
# EUCLIDEAN IS RE-RUN, but not for the gauge. `Euclidean::center` subtracts the
# coordinate mean every iteration and is the last thing `step()` does, so the
# centroid is the origin when metrics are taken and r_gyration == r_rms exactly.
# Euclidean kappa is 0 on any gauge besides (curvature: 0.0), so it never enters
# a kappa axis, and on the original rationale its cells were skipped.
#
# It is in the sweep because `shepard_goodness` changed on 2026-09-07, on every
# arm at once: fractional ranks replaced the tie-free `1 - 6*sum(d^2)` shortcut,
# which is only valid on tie-free input and scored a fully collapsed embedding
# 0.63 on tree_structured, and the score is now normalised onto [0, 1] by the
# order-preserving map (r_s + 1) / 2 rather than clipped at 0 -- so 0.5, not 0,
# is now the score of an embedding that preserves no rank structure. Every
# shepard value under results/ predates both changes and cannot be rescaled to
# match (the clipped zeros lost their sign), so those cells have to be re-fitted
# whatever their geometry.
#
# OUTPUT LIVES SOMEWHERE ELSE. results-rgyr, not results, both on scratch and in
# $HOME, and every filename carries an _rgyr marker before the extension. The
# directory is the working separator; the marker means even a mis-targeted rsync
# cannot overwrite the original sweeps. Pull them with
#   REMOTE_RESULTS=~/fitting/results-rgyr LOCAL_RESULTS=./results-rgyr \
#     sh slurm/sync_back.sh
#
# DETERMINISM -- WEAKER THAN IT WAS. `r_rms`/`r_gyration` are still logged but
# never read by the acquisition function or the scalarisation, and run_pareto
# still seeds its RNG with a fixed constant. On the gauge change alone this run
# reproduced the original trials bit-for-bit, differing only in the added column.
# It no longer does: `shepard_goodness` is one of the six qParEGO objectives, and
# the fractional-rank fix moved it non-affinely, so the scalarised values, the GP
# proposals and hence the whole trial sequence diverge from the original run.
# (The (r_s + 1) / 2 normalisation alone would not have done that -- gp.rs
# `scalarize_subset` min-max normalises every objective per batch, which absorbs
# any positive affine map.) These cells are re-measurements, not replays;
# compare them to results/ by front and indicator, never trial by trial.
#
# The settings that must still match the original for the cells to be comparable
# at all are unchanged: --n-trials, --n-seeds, --n-samples and --threads.
# --threads is load-bearing: batch_size = n_threads, and a different batch size
# regroups the GP proposals. Keep --cpus-per-task at 48.
#
# Everything else -- the 24h chunking, --resume, the SIGTERM trap, the sizing --
# is unchanged from run_5000.sh; see that file for the reasoning.

set -eu

DATASET="${DATASET:-mnist}"
EXPERIMENT="${EXPERIMENT:-all_off}"
GEOMETRY="${GEOMETRY:-spherical}"
N_SAMPLES=5000

module load 2025
module load compiler
module load rust

cargo build --release --locked --offline -p fitting-optimizer

PREFIX=${EXPERIMENT}_${DATASET}_n${N_SAMPLES}
# The stem the analysis parses is "<prefix>_<geometry>_rgyr"; crates/analysis
# strips the marker so these cells key identically to the originals, which is
# what lets `--results-dir results-rgyr` render every figure unchanged.
STEM=${PREFIX}_${GEOMETRY}_rgyr
SCRATCH_DIR=/scratch/"$USER"/fitting/results-rgyr
HOME_DIR="$HOME"/fitting/results-rgyr

mkdir -p "$SCRATCH_DIR" "$HOME_DIR"

OUT="$SCRATCH_DIR"/"$STEM".jsonl

# Restore the previous chunk's checkpoint to scratch so --resume can read it.
# Only fill in files scratch is missing (e.g. after a scratch purge); never
# overwrite an existing scratch file, which may hold trials newer than $HOME if
# the previous chunk was SIGKILLed before its sync_back ran.
for f in "$HOME_DIR"/"$STEM"*; do
  [ -e "$f" ] || continue
  dest="$SCRATCH_DIR"/$(basename "$f")
  [ -e "$dest" ] || cp -f "$f" "$dest"
done

# Copy this job's result files (the JSONL trial log and the _pareto_*.json front)
# from scratch back to backed-up home. Runs on normal exit, on error (set -e), and
# on the SIGTERM SLURM sends at the time limit. This is what lets the next chunk
# in the chain resume.
sync_back() {
  cp -f "$SCRATCH_DIR"/"$STEM"* "$HOME_DIR"/ 2>/dev/null || true
}
trap sync_back EXIT

# Run in the background so the batch shell can catch SIGTERM while it is still
# alive (a foreground srun would swallow the signal). --resume continues from the
# JSONL if it already has trials, or starts fresh if not (so the same invocation
# works for the first chunk and every continuation).
#
# NOTE: --resume must only ever see a file this fixed binary wrote. A reused
# trial re-observes but never rewrites its JSONL line, so resuming a checkpoint
# produced by an older build would leave those replayed trials without
# r_gyration and -- since the 2026-09-07 metric changes -- with their
# `shepard_goodness` still on the old scale, indistinguishable from the new one
# in the same file. Starting in a fresh directory is what guarantees that; if a
# results-rgyr/ from an earlier attempt exists on scratch or in $HOME, move it
# aside before submitting rather than resuming into it.
srun ./target/release/optimizer \
  --mode pareto --dataset "$DATASET" --experiment "$EXPERIMENT" \
  --n-trials 1000 --n-seeds 3 --n-samples "$N_SAMPLES" \
  --geometry "$GEOMETRY" \
  --threads "$SLURM_CPUS_PER_TASK" \
  --data-path ./www/public/data \
  --resume \
  --output "$OUT" &
SRUN_PID=$!

# On the pre-timeout SIGTERM: save partial results, stop the run, exit.
trap 'sync_back; kill "$SRUN_PID" 2>/dev/null || true; exit' TERM

wait "$SRUN_PID" || echo "geometry $GEOMETRY failed (exit $?)"
