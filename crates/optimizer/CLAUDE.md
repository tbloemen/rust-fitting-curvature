## Optimizer crate (`crates/optimizer`)

CLI hyperparameter search over the embedding. All output is appended JSONL (`TrialResult` per line, `trial_result.rs`); diverged trials serialize non-finite metrics as `null`. Run modes (`--mode`, see `cli.rs` for all flags):

- **`random`** (`random.rs`) — sample random configs over a continuous curvature range, log all metrics.
- **`scan`** (`scan.rs`) — sweep each free parameter individually from a base config; requires `--metric`.
- **`bayes`** (`bayes.rs`) — single-objective Bayesian optimization (GP + Expected Improvement); requires `--metric`. Round-based parallel batches: each round the GP proposes `batch_size` (= `--threads`) configs, they're evaluated in parallel, then the GP is refit before the next round.
- **`pareto`** (`pareto.rs`) — qParEGO multi-objective optimization over 10 objectives (no `--metric`). LHS init phase then GP phase. This is the mode used for the main 5000-sample sweeps.

Key cross-cutting pieces:

- **`gp.rs`** — the GP surrogate, EI acquisition, and both `GpOptimizer` (bayes) and `ParEgoOptimizer` (pareto). Notation follows Frazier (2018), cited in-file. `suggest_batch(n, rng)` is **deterministic given `(trials, rng)`** — no threads/clock/hashmap in the proposal path. This is what makes resume possible.
- **`evaluate.rs`** — `Evaluator` wraps a dataset, precomputes the high-dim distance matrix once, and runs `fit_embedding` + all quality metrics for a config. `metrics.rs` defines `AllMetrics` / the `Metric` enum (2D-projected and pre-projection "manifold" variants of trustworthiness, continuity, stress, Shepard goodness, neighborhood hit).
- **`search_space.rs`** — `TrialConfig` and `ParamSpec` (`Fixed` vs `Optimize{lo,hi,log}`). Parameter bounds are loaded at startup from `config/params.json` via `include_str!` (`param_bounds()`); change ranges there, not in code. `--experiment` (`common.rs::parse_experiment`) selects which loss weights are free vs fixed at 0: `all_off`, `centering_only`, `global_only`, `norm_only`, `all_free`, `rms_anchored`. In all variants lr/perplexity/early-exaggeration are optimized and momentum is fixed at 0.8.
- **Geometry resolution** (`bayes.rs::resolve_geometry`) — `--geometry hyperbolic|euclidean|spherical` forces the curvature sign; omitting it auto-detects via `curvature_detection`. For curved geometries `curvature_magnitude` becomes an extra search dimension.

### Resume & chunked sweeps (`resume.rs`)

The 5000-sample sweeps exceed the cluster's **24h wall-clock cap**, so each `(dataset, experiment, geometry)` job is split into **chained chunks** that resume from the JSONL the previous chunk wrote. Resume (`--mode pareto --resume`) works by **pure replay**: start a fresh optimizer + fresh rng and re-run the exact same `suggest_batch` calls, but for any trial already on disk substitute its recorded evaluation instead of re-running the expensive embedding fit. Because `suggest_batch` is deterministic, proposed configs come out bit-identical and the rng advances identically — a chunked run reproduces a single uninterrupted run exactly (verified by `tests/test_resume.rs`). Implications:

- **Batch size must be constant across chunks** (it's part of the qParEGO algorithm) — all chunks must use the same `--threads`.
- `--resume` on an absent/empty file just starts fresh, so it can be passed unconditionally. A chunk that finds the run already complete exits immediately, so over-provisioning chunks is harmless.
- Without `--resume`, every mode behaves exactly as before (no checkpoint read, all trials evaluated fresh, results **appended**).

See `../../slurm/CLAUDE.md` for how these chunks get orchestrated on the cluster.
