# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rust/WASM port of a Python+JS thesis project implementing t-SNE embedding in constant curvature spaces (hyperbolic, Euclidean, spherical). The original Python project is at `../fitting-curvature`.

## Build & Test Commands

```bash
# Run all tests
cargo test

# Run a single test file
cargo test --test test_manifolds

# Run a single test
cargo test --test test_manifolds hyperboloid_constraint

# Build the optimizer in release (SLURM sweeps run this binary; LTO + 1
# codegen-unit make the cold compile slow but the O(n²) inner loops fast)
cargo build --release --locked

# Run a hyperparameter search locally (see "Optimizer crate" below for modes)
cargo run --release -p fitting-optimizer -- \
  --mode pareto --dataset tree --geometry hyperbolic --experiment all_off --n-samples 200

# Build WASM package
wasm-pack build crates/web --target web --out-dir ../../www/pkg

# Full build + dev server (requires wasm-pack, Node.js)
./build.sh

# Dev server only (after WASM is built)
cd www && npm run dev
```

No linter is configured. Rust edition 2024.

## Architecture

**Workspace** with three crates:
- `crates/core` (`fitting-core`) — Pure Rust library, zero dependencies. All embedding algorithms live here.
- `crates/web` (`fitting-web`) — WASM bindings via wasm-bindgen. Uses plotters + plotters-canvas for HTML5 canvas rendering and lol_alloc as WASM allocator.
- `crates/optimizer` (`fitting-optimizer`) — Native-only CLI binary (bin name `optimizer`) that drives hyperparameter search over `fitting-core`. Has dependencies (clap, indicatif, serde) — the "zero dependencies" rule applies to `core` only.

**Web frontend** (`www/`): Vite + vite-plugin-wasm. JS imports WASM pkg as `"fitting-web": "file:./pkg"` with `await init()` pattern.

### Core data layout

All point data is **flat row-major `Vec<f64>`** of shape `(n_points, ambient_dim)`. No matrix types — this is intentional for cache efficiency and WASM compatibility. The `manifolds::Points` type alias captures this.

### Key abstractions

- **`Manifold` trait** (`manifolds.rs`): Defines `init_points`, `pairwise_distances`, `project_to_tangent`, `exp_map`, `center`, `scaling_loss`. Three implementations: `Euclidean` (k=0), `Hyperboloid` (k<0), `Sphere` (k>0). Factory: `create_manifold(curvature)`.
- **`TrainingConfig`** (`config.rs`): All hyperparameters. Curvature sign selects geometry.
- **`fit_embedding`** (`embedding.rs`): Main training loop. Takes data, config, and optional `FnMut(StepResult) -> bool` callback for per-iteration rendering/logging.
- **`RiemannianSGDMomentum`** (`optimizer.rs`): Optimizer with parallel transport for momentum on curved spaces.

### Additional modules

- **`curvature_detection/`**: Geometry detection from a pairwise distance matrix (native-only, no WASM). Five submodules; `signature` + `gromov_ball_curve` items are re-exported at `curvature_detection::` for convenience. `detect_geometry`/`GeometryDetection` resolve to the **signature** detector (the histogram module has its own, reached via the `histogram::` path):
  - `signature.rs` — Wilson et al. (2014) radius-of-curvature fit: a constant-curvature Gram matrix `Z(r)` for a target embedding dimension, minimising a normalised signature residual (`fit_spherical`/`fit_hyperbolic`), eigenvalues via power iteration + deflation. Top-level `detect_geometry` returns a `GeometryDetection` (Wilson fits + tail slope) whose `verdict: GeometryVerdict` (label + signed curvature) is the decision consumers act on — spherical via the angular-extent test, hyperbolic via the `gromov_ball_curve` saturation test.
  - `gromov.rs` — shared low-level primitives only: `quad_delta`, `median_pairwise_distance`, `four_distinct`.
  - `old_detection.rs` — legacy global 90th-percentile Gromov δ scalar (`gromov_hyperbolicity`); kept for the histogram tests.
  - `gromov_ball_curve.rs` — NeTS-proposal growing-ball method: `gromov_delta_curve` builds δ(k) over growing k-NN balls and `detect_hyperbolic` classifies hyperbolic-vs-not by whether δ(k) saturates (log–log tail slope < `SATURATION_SLOPE_THRESHOLD`). Saturated δ → curvature via `δ = ln(1+√2)/√(−K)`. See `examples/gromov_delta_curve.rs` (plotters dev-dep) for the δ-vs-k plot.
  - `histogram.rs` — the deliberately naive baseline: `fit_geometries` fits all three shell-density curves to the rising-regime profile and `detect_geometry` returns the `GeometryVerdict` of the best-R² one (ties → euclidean). Exists to show curvature *sign* can't be read off the residuals (they're near-degenerate), motivating the growing-ball test.
- **`kl_divergence.rs`**: Global t-SNE similarity matrix (`compute_global_similarities`) using `(1 + d²)` kernel (Zhou & Sharpee loss variant that emphasizes large distances).
- **`scaling_loss.rs`**: Radial regularization for hyperbolic embeddings — penalizes geodesic spread from origin. Returns `(loss, ambient_gradient)`; caller must project gradient to tangent space.
- **`data.rs`**: MNIST loader from IDX binary format. `#[cfg(not(target_arch = "wasm32"))]` — not compiled for WASM.

### Pipeline flow

Input data → `compute_perplexity_affinities` (affinities.rs) → `fit_embedding` loop: `compute_q_matrix` (kernels.rs) → `kl_gradient` → optimizer step (exp_map + project) → optional callback with `StepResult`.

### Ambient dimension

For curved spaces, `ambient_dim = embed_dim + 1` (hyperboloid in R^{d+1}, sphere in R^{d+1}). Euclidean: `ambient_dim = embed_dim`.

### Built-in PRNG

`synthetic_data::Rng` implements xoshiro256** to avoid external dependencies. Used for data generation and initialization.

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

### SLURM orchestration (`slurm/`)

`run_*.sh` are job templates; `submit_*.sh` enumerate `(dataset, geometry[, experiment])` and submit jobs. The `*_5000.sh` variants chain `CHUNKS` (default 4, override via env) jobs per cell with `--dependency=afterany` and `--time=23:55:00`, restoring the prior chunk's checkpoint from `$HOME` to scratch at startup and passing `--resume`. Results carry an `_n5000` marker so they don't collide with the 1000-sample runs.
