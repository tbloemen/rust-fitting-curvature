## Analysis crate (`crates/analysis`)

Post-hoc analysis of the qParEGO sweeps: Pareto fronts, the R2 indicator, the
ΔR2 statistics, the ε-indicator cross-check, and the thesis result figures.
Replaces what used to be
`pareto_utils.py` and `analyze_experiments.py` at the repo root.
(`analyze_hyperparams.py` is still Python — it is the exploratory tool, uses
sklearn, and never runs on the cluster.)

**Terminology.** A `Cell` is one `(setting, dataset, N, geometry)` sweep run —
one `results/*.jsonl`, ~1000 trials, one Pareto front — and `setting` is the
loss-weight variant (`all_off`, `centering_only`, …). Both names come from the
thesis. The optimizer and `slurm/` call that same variant `--experiment` /
`$EXPERIMENT`, so translate at the boundary: **`cf.cell.setting` here is the
value the optimizer was given as `--experiment`.** "Experiment N" means a
numbered research question (`figures/exp2.rs` … `exp5.rs`), never a setting.
Root `CLAUDE.md` has the full table.

**Everything here runs locally — there is no cluster stage.** The full table is
~1.3s single-threaded over 269 cells / 350 MB of JSONL, nearly all of it parse
time: the indicator is an exact mean over an enumerated weight set, so unlike
the Monte-Carlo hypervolume it replaced there is no sampling budget to trade
against precision, no seed, and no standard error to report.

plotters is an optional dependency behind the `plots` feature, with the
`figures` binary declaring `required-features = ["plots"]` — that keeps
font-kit/fontconfig out of `cargo build --release` and off the dependency graph
of anything that only needs numbers.

```bash
# Stage 1: per-cell R2 indicator under every preference region
cargo run --release -p fitting-analysis --bin r2 -- \
  stats --results-dir results --out r2_local.jsonl

# Stage 2: Friedman + Holm over the blocks, per preference region.
# Writes the test table to --tests (default r2_tests.jsonl); --deltas and
# --descriptive are the two optional extra tables.
cargo run --release -p fitting-analysis --bin r2 -- \
  aggregate r2_local.jsonl --deltas r2_delta.jsonl     # --region all to filter
#   --tests r2_tests.jsonl                                   Friedman + Holm, one object per test
#   --settings all_off,centering_only,global_only,all_free   includes spherical
#   --descriptive r2_descriptive.jsonl                       mean/median ΔR2 per group

# The parameter-free cross-check: binary additive ε-indicator vs the baseline,
# plus how well its verdict agrees with the ΔR2 one (needs --r2-table)
cargo run --release -p fitting-analysis --bin r2 -- \
  compare --results-dir results --out r2_epsilon.jsonl \
  --r2-table r2_local.jsonl --agreement r2_agreement.jsonl
#   --settings centering_only,global_only,norm_only,all_free   the default

# The per-preference recommendation table (hyperparameters + all 10 objectives)
cargo run --release -p fitting-analysis --bin r2 -- \
  recommend --results-dir results --out recommendations.jsonl

# Recompute *_pareto_*.json fronts for cells whose sweep predates front writing
cargo run --release -p fitting-analysis --bin r2 -- front --results-dir results

# Thesis figures (local only; needs fontconfig)
cargo run --release -p fitting-analysis --features plots --bin figures
```

### Output goes to files; failures go through `Result`

Nothing here prints. Every subcommand's product is a file (`--out`, `--deltas`,
`--tests`, `--descriptive`, the SVG/PNG pairs), and the only thing that reaches
the terminal is a failure, rendered once by `main` returning `Err`. `error.rs`
holds the crate's `Error` — `Io`/`Parse` carry the path (and 1-based line),
`NoCells`/`UnknownRegion`/`TooFewSettings`/`MissingBaseline` replace what used
to be an `eprintln!` followed by `process::exit(1)`. Its `Debug` forwards to
`Display`, because `Termination` renders a `main` error with `Debug` and the
derived form is unreadable. `Error::Plot` flattens plotters' backend-generic
error to its message, which keeps `Error: Send + Sync` for the sweep workers
and plotters out of the non-`plots` build.

Every table is **JSONL**, written through `records::write_jsonl` — same format
in as out, so a stage-2 table loads with the same one-line `serde_json::from_str`
the sweeps do and `jq` works on all of it. Nested structure survives: a
`RankTest` is one object with its `settings`/`mean_ranks`/`holm_p` arrays intact,
and `recommend` writes name-keyed `params` / `objectives` objects. A value that
does not exist is `null`, not an empty field.

**Loading is strict.** `records::load_jsonl` is generic over the record type
(trial records, stage-1 cells, κ_data all go through it) and errors on a file it
cannot open and on any line that will not deserialise; only blank lines are
skipped. A missing *field* is still fine — every `TrialRecord` column is
`Option` — so old results files keep loading. The trade: a sweep killed mid-write leaves a truncated last line,
and that now fails the whole run instead of quietly computing a front over one
trial fewer. Truncate the offending line (the error names the file and line) and
re-run. The κ_data table is the one deliberate exception: *absent* is not an
error (it is a separate optimizer run, and Exp 3 skips its scatter without it),
*present but malformed* is.

Figures with no data behind them are skipped silently — the sweep grid is not
rectangular, so that is the normal case, and what is missing is visible as an
absent file in the output directory.

`stats`, `compare`, `recommend` and the figures walk the cells **serially**, in
`cell::discover_cells` order (sorted by stem), so every table is byte-identical
across runs — they get diffed between runs, and a second of wall time is not
worth losing that. A scoped worker pool got the two stages to 0.15s but made the
output order completion order; it was deleted for that reason. If the sweep grid
ever grows an order of magnitude, parallelise the *parse* (that is where the time
goes) and keep writing in cell order.

### Orientation and the indicator

Everything works in an _oriented_ space: each of the 10 qParEGO objectives
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

### The ε-indicator, and why there are two indicators

`indicators.rs` is the parameter-free cross-check on everything the section above
depends on. `S = 5`, the region definitions and the `>= 3` mass threshold are all
*choices*; the binary additive ε-indicator (Zitzler et al. 2003, already in
`references.bib`) has none, so agreement between the two says the ΔR2 conclusions
are not an artefact of the preference model.

```
I_ε+(A, B) = max over b ∈ B of  min over a ∈ A of  max_j (b_j − a_j)
```

in the same oriented space, so an ε reads directly as objective units and
`I ≤ 0` means A covers B outright. Three rules the `compare` subcommand enforces:

- **Both directions, always.** It is asymmetric, and when fronts cross neither
  direction alone settles anything. `epsilon_pair` returns both plus
  `Δε = I(baseline, setting) − I(setting, baseline)` — baseline minus setting, the
  same way round as ΔR2, because both indicators are costs.
- **No region dimension.** Carrying no parameters is the whole point; a
  region-weighted ε would give that away. The agreement table is per region only
  because ΔR2 is.
- **Report the front sizes.** ε is blind to cardinality: a 12-point and a
  230-point front can score identically, and the reader has to see which is which.

Where R2 is only *weakly* Pareto compliant and averages over 2002 weight vectors,
ε is fully compliant and is a worst case — it catches a regression confined to one
corner of objective space that the mean buries. That is the reason for reporting
both rather than picking one.

The merged-front dominance ranking that `4methods.typ` used to promise alongside
it was **deliberately dropped**, not forgotten: no citation for that specific
statistic could be verified. Don't add it back without one.

### The weight simplex and preference regions

`Weights::new()` enumerates all vectors with `λ_j = l/S`, `Σλ_j = 1` at `S = 5`
— `C(14, 9) = 2002` vectors for ten objectives, the same set
`ParEgoOptimizer::sample_discrete_simplex` draws from. Vectors are stored as
**integer counts**, and every region test is integer arithmetic: `l/5` is not
representable in binary, so `0.2 + 0.4 > 0.6` and a float `λ_a + λ_b >= 0.5`
would be a coin flip.

Eight regions, in report order:

| region                                | membership                                   | size |
| ------------------------------------- | -------------------------------------------- | ---- |
| `all`                                 | everything                                   | 2002 |
| one per metric (`trustworthiness`, …) | `l_proj + l_man >= 3` for that metric's pair | 190  |
| `manifold`                            | supported on the odd (manifold) objectives   | 126  |
| `projected`                           | supported on the even (2D) objectives        | 126  |

`objectives.rs` writes the projected/manifold pairing down **once**, in
`METRIC_PAIRS`. `OBJECTIVES` (interleaved as `[metric, metric_manifold, …]`),
`METRICS`, `N_OBJECTIVES`, `metric_objectives(i) = (2i, 2i+1)` and
`is_manifold(j) = j % 2 == 1` are all derived from it — the first two by `const
fn`, so the layout cannot desync and no test has to pin it. Add or reorder a
metric there and everything downstream follows.

Row order is a free choice: the indicators are invariant under a relabelling of
the axes (the simplex is enumerated symmetrically), nothing on disk is
positional (`oriented_row` resolves by name, every table is name-keyed), and the
only visible effect is the left-to-right panel order of Exp 4. What order *does*
have to match is the optimizer's `default_pareto_metrics`, which is what the
`OBJECTIVES` doc comment means — that alignment is by hand across crates.

Two things the compiler still cannot check, so `test_r2.rs` covers them: that
every pair's second name is the first plus `_manifold`, and that every name in
`OBJECTIVES` resolves through `TrialRecord::objective` (a typo there would read
as permanently missing, i.e. silently worst-case).

### Recommendations

`recommendation()` returns the front point most often chosen by a region's
weight vectors, with the share that chose it. `front_index` indexes into
`CellSummary::front`, which holds indices into the _records_ — the `recommend`
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
defaults: Spearman ρ with the t-approximation p-value, and the Friedman test with
the same tie correction scipy applies. `tests/test_stats.rs` pins the scipy ports
to values from scipy 1.17.1, and the Friedman/χ²/Holm additions to hand-computed
statistics and the closed-form χ² tail for even degrees of freedom — those are
checked against analysis rather than against the library they replace.

There is no Wilcoxon here. It was ported and tested, then deleted once
`aggregate.rs` moved to Friedman + Holm and nothing called it; the two-method
comparison it would serve is not a test this analysis runs.

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

### Figures

`figures/` renders each figure through a `Figure` trait that is generic over the
backend, so one definition writes both the SVG (for the thesis) and the PNG.
Keep axis text inside Latin-1 + Greek: the bitmap backend renders through the
system "sans-serif", which on this machine has no arrows (U+2190/2192) or
geometric shapes — they come out as tofu. κ and ρ are fine.

Two deliberate departures from the Python:

- **Exp 2 fronts are actual fronts.** `_slice_front` in `analyze_experiments.py`
  had its domination test inverted (it dropped the points that dominate `i`
  rather than the ones `i` dominates), so it drew the _worst-case_ boundary.
  `pareto::slice_front_2d` computes the real front; `test_pareto.rs` guards it.
- **Exp 5 marginals share bin edges across geometries** so the overlaid
  histograms compare like with like; matplotlib binned each series separately.
