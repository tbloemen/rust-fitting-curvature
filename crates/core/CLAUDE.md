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
  - `gromov_ball_curve.rs` — NeTS-proposal growing-ball method: `gromov_delta_curve` builds δ(k) over growing k-NN balls and `detect_hyperbolic` classifies hyperbolic-vs-not by whether δ(k) saturates (log–log tail slope < `SATURATION_SLOPE_THRESHOLD`). Saturated δ → curvature via `δ = ln(1+√2)/√(−K)`. See `examples/gromov_delta_curve.rs` for the δ-vs-k plot (run with `cargo run -p fitting-core --features plot-examples --example gromov_delta_curve`; plotters is an optional dep gated behind the `plot-examples` feature so core's tests/clippy stay dependency-free and CI doesn't need fontconfig).
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
