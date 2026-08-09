## Analysis crate (`crates/analysis`)

Post-hoc analysis of the qParEGO sweeps: Pareto fronts, the R2 indicator, the
ΔR2 statistics, and the thesis result figures. Replaces what used to be
`pareto_utils.py` and `analyze_experiments.py` at the repo root.
(`analyze_hyperparams.py` is still Python — it is the exploratory tool, uses
sklearn, and never runs on the cluster.)

**Everything here runs locally — there is no cluster stage.** The full table is
~0.2s on 16 cores: the indicator is an exact mean over an enumerated weight set,
so unlike the Monte-Carlo hypervolume it replaced there is no sampling budget to
trade against precision, no seed, and no standard error to report.

plotters is an optional dependency behind the `plots` feature, with the
`figures` binary declaring `required-features = ["plots"]` — that keeps
font-kit/fontconfig out of `cargo build --release` and off the dependency graph
of anything that only needs numbers.

```bash
# Stage 1: per-cell R2 indicator under every preference region
cargo run --release -p fitting-analysis --bin r2 -- \
  stats --results-dir results --out r2_local.jsonl

# Stage 2: Friedman + Holm over the blocks, per preference region
cargo run --release -p fitting-analysis --bin r2 -- \
  aggregate r2_local.jsonl --csv r2_delta.csv          # --region all to filter
#   --settings all_off,centering_only,global_only,all_free   includes spherical
#   --descriptive                                            adds the mean/median ΔR2 table

# The per-preference recommendation table (hyperparameters + all 10 objectives)
cargo run --release -p fitting-analysis --bin r2 -- \
  recommend --results-dir results --csv recommendations.csv

# Recompute *_pareto_*.json fronts for cells whose sweep predates front writing
cargo run --release -p fitting-analysis --bin r2 -- front --results-dir results

# Thesis figures (local only; needs fontconfig)
cargo run --release -p fitting-analysis --features plots --bin figures
```

### Orientation and the indicator

Everything works in an *oriented* space: each of the 10 qParEGO objectives
(`objectives.rs`, same order as `pareto.rs::default_pareto_metrics`) is mapped
into `[0, 1]` with higher = better — normalised stress becomes `1 − x`, and a
missing/null/non-finite value becomes `0.0`, matching the optimizer's
`metrics_to_vec` substitution so diverged trials score badly instead of
vanishing. The ideal point is then `(1, …, 1)`.

`r2.rs` computes

```
R2(A; W) = mean over λ ∈ W of  min over a ∈ A of  max_j λ_j (1 − a_j)
```

**Smaller is better**, so `aggregate.rs` forms `ΔR2 = R2(baseline) − R2(setting)`
and positive means improvement. Getting that subtraction backwards is the easy
mistake here; it runs the opposite way from the hypervolume gain it replaced.

Two facts keep it cheap:

- The inner minimum is always attained on the Pareto front (a dominating point
  can only shrink the max), so reducing to the front first is exact, not an
  approximation. `test_r2.rs` pins this.
- Every preference region is a **subset of one enumeration** of the simplex, so
  `front_utilities` minimises once per weight vector and each region is a mean
  over its own slice. Eight regions cost barely more than one.

### The weight simplex and preference regions

`Weights::new()` enumerates all vectors with `λ_j = l/S`, `Σλ_j = 1` at `S = 5`
— `C(14, 9) = 2002` vectors for ten objectives, the same set
`ParEgoOptimizer::sample_discrete_simplex` draws from. Vectors are stored as
**integer counts**, and every region test is integer arithmetic: `l/5` is not
representable in binary, so `0.2 + 0.4 > 0.6` and a float `λ_a + λ_b >= 0.5`
would be a coin flip.

Eight regions, in report order:

| region | membership | size |
|---|---|---|
| `all` | everything | 2002 |
| one per metric (`trustworthiness`, …) | `l_proj + l_man >= 3` for that metric's pair | 190 |
| `manifold` | supported on the odd (manifold) objectives | 126 |
| `projected` | supported on the even (2D) objectives | 126 |

`OBJECTIVES` interleaves the pair as `[metric, metric_manifold, …]`, so metric
`i` owns objectives `2i` and `2i+1`. `metric_layout_matches_the_objective_order`
in `test_r2.rs` fails loudly if that array is ever reordered.

### Recommendations

`recommendation()` returns the front point most often chosen by a region's
weight vectors, with the share that chose it. `front_index` indexes into
`CellSummary::front`, which holds indices into the *records* — the `recommend`
subcommand relies on that indirection to report hyperparameters, and a test
covers it. Ties resolve to the lowest front index, so nothing depends on
iteration order.

### float_roundtrip is load-bearing

`serde_json` is pulled with the `float_roundtrip` feature. Its default float
parser can land one ulp off the nearest double, and the Pareto sort is a chain of
exact `<=` comparisons — in this repo's own results, one ulp was enough to drop a
real front point in 8 of 176 cells. Do not remove the feature.

(The same hazard exists in `crates/optimizer/src/resume.rs`, which parses trial
metrics back from JSONL on `--resume` without that feature.)

### Statistics

`stats.rs` reimplements the scipy functions the analysis uses, matching scipy's
defaults: Spearman ρ with the t-approximation p-value; the Wilcoxon signed-rank
test (`zero_method="wilcox"`, two-sided, statistic `min(W⁺, W⁻)`, exact
distribution for ≤50 untied non-zero differences and the tie-corrected normal
approximation otherwise); and the Friedman test with the same tie correction
scipy applies. `tests/test_stats.rs` pins the scipy ports to values from scipy
1.17.1, and the Friedman/χ² additions to hand-computed statistics and the
closed-form χ² tail for even degrees of freedom — those are checked against
analysis rather than against the library they replace.

**Friedman + Holm is what the results tables use.** `aggregate::rank_tests` ranks
settings within each (dataset, geometry, N) block, runs Friedman on the ranks per
preference region, and reports Holm-adjusted post-hoc p-values against the
`all_off` control (Demšar 2006 §3.2.2). Two things to know:

- `friedman` takes **larger = better** and ranks descending, so mean rank 1 is the
  best treatment. ΔR2 is already oriented that way; R2 itself is not.
- Blocks must be **complete**. The sweep grid is not rectangular: `norm_only` has
  no spherical runs, `rms_anchored` only hyperbolic. `rank_tests` fixes the
  treatment list first, keeps only blocks carrying every treatment, and reports
  `dropped_blocks`. With `norm_only` in the list there are zero complete spherical
  blocks, so that geometry is simply absent from the table — pass
  `--settings all_off,centering_only,global_only,all_free` to get it back.

`stats::wilcoxon` is retained (tested, and the right test for a two-method
comparison) but is no longer used by `aggregate.rs`.

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
