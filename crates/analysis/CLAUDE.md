## Analysis crate (`crates/analysis`)

Post-hoc analysis of the qParEGO sweeps: Pareto fronts, hypervolume, the ΔH
statistics, and the thesis result figures. Replaces what used to be
`pareto_utils.py`, `hv_stats.py`, `hv_aggregate.py` and `analyze_experiments.py`
at the repo root. (`analyze_hyperparams.py` is still Python — it is the
exploratory tool, uses sklearn, and never runs on the cluster.)

**Everything here runs locally — there is no cluster stage.** The Python version
needed a 22-way SLURM array plus a hand-built numpy venv to get through the table
in a couple of hours; `hv stats` does the full 176-cell table in ~1s at
`--n-mc 2e6` and ~12s at the default 2e7 on 16 cores, which is shorter than a
queue wait. `slurm/run_hv.sh` and `submit_hv.sh` were deleted along with the
Python (recoverable from git if a much larger sweep ever changes the arithmetic).

plotters is still an optional dependency behind the `plots` feature, with the
`figures` binary declaring `required-features = ["plots"]` — that keeps
font-kit/fontconfig out of `cargo build --release` and off the dependency graph
of anything that only needs numbers.

```bash
# Stage 1: per-cell Monte-Carlo hypervolume
cargo run --release -p fitting-analysis --bin hv -- \
  stats --results-dir results --out hv_local.jsonl

# Stage 2: ΔH over the all_off baseline + cross-dataset Wilcoxon
cargo run --release -p fitting-analysis --bin hv -- \
  aggregate hv_local.jsonl --csv hv_delta.csv

# Recompute *_pareto_*.json fronts for cells whose sweep predates front writing
cargo run --release -p fitting-analysis --bin hv -- front --results-dir results

# Thesis figures (local only; needs fontconfig)
cargo run --release -p fitting-analysis --features plots --bin figures
```

### Orientation and the hypervolume

Everything works in an *oriented* space: each of the 10 qParEGO objectives
(`objectives.rs`, same order as `pareto.rs::default_pareto_metrics`) is mapped
into `[0, 1]` with higher = better — normalised stress becomes `1 − x`, and a
missing/null/non-finite value becomes `0.0`, matching the optimizer's
`metrics_to_vec` substitution so diverged trials score badly instead of vanishing.
The reference point is then the origin and the reference box has volume 1, so HV
is literally the fraction of the unit box the front dominates, and Monte-Carlo
integration needs no reference-point bookkeeping.

`hypervolume.rs` reduces to the Pareto front first (HV-invariant), then samples.
Two exact accelerations over the numpy original — bounding-box rejection and
first-dominator early exit — make it ~300x cheaper in CPU time; neither changes
the estimator. `cell_seed` reproduces Python's `crc32(stem) ^ seed`, so a cell's
estimate depends only on its name and the base seed, not on `--jobs` or on which
other cells were in the run: **re-running one cell reproduces it exactly.**

`--n-mc` defaults to 2e7, which puts the per-cell standard error near 1e-4 — an
order of magnitude below the smallest ΔH the aggregate stage calls significant
(~4e-4). Raising it is cheap and linear: 2e8 is about two minutes.

### float_roundtrip is load-bearing

`serde_json` is pulled with the `float_roundtrip` feature. Its default float
parser can land one ulp off the nearest double, and the Pareto sort is a chain of
exact `<=` comparisons — in this repo's own results, one ulp was enough to drop a
real front point in 8 of 176 cells. Do not remove the feature.

(The same hazard exists in `crates/optimizer/src/resume.rs`, which parses trial
metrics back from JSONL on `--resume` without that feature; 34 of the 128 saved
`*_pareto_*.json` fronts disagree with their own JSONL by one ulp, which is what
you would expect from a resumed chunk.)

### Statistics

`stats.rs` reimplements the three scipy functions the analysis used, matching
scipy's defaults: Spearman ρ with the t-approximation p-value, and the Wilcoxon
signed-rank test (`zero_method="wilcox"`, two-sided, statistic `min(W⁺, W⁻)`,
exact distribution for ≤50 untied non-zero differences and the tie-corrected
normal approximation otherwise). `tests/test_stats.rs` pins every one of them to
values produced by scipy 1.17.1.

### Figures

`figures/` renders each figure through a `Figure` trait that is generic over the
backend, so one definition writes both the SVG (for the thesis) and the PNG.
Keep axis text inside Latin-1 + Greek: the bitmap backend renders through the
system "sans-serif", which on this machine has no arrows (U+2190/2192) or
geometric shapes — they come out as tofu. κ and ρ are fine.

Two deliberate departures from the Python:

- **Exp 2 fronts are actual fronts.** `_slice_front` in `analyze_experiments.py`
  had its domination test inverted (it dropped the points that dominate `i`
  rather than the ones `i` dominates), so it drew the *worst-case* boundary.
  `pareto::slice_front_2d` computes the real front; `test_pareto.rs` guards it.
- **Exp 5 marginals share bin edges across geometries** so the overlaid
  histograms compare like with like; matplotlib binned each series separately.
