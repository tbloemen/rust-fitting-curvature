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

# Build WASM package
wasm-pack build crates/web --target web --out-dir ../../www/pkg

# Full build + dev server (requires wasm-pack, Node.js)
./build.sh

# Dev server only (after WASM is built)
cd www && npm run dev
```

No linter is configured. Rust edition 2024.

## Architecture

**Workspace** with two crates:
- `crates/core` (`fitting-core`) — Pure Rust library, zero dependencies. All algorithms live here.
- `crates/web` (`fitting-web`) — WASM bindings via wasm-bindgen. Uses plotters + plotters-canvas for HTML5 canvas rendering and lol_alloc as WASM allocator.

**Web frontend** (`www/`): Vite + vite-plugin-wasm. JS imports WASM pkg as `"fitting-web": "file:./pkg"` with `await init()` pattern.

### Core data layout

All point data is **flat row-major `Vec<f64>`** of shape `(n_points, ambient_dim)`. No matrix types — this is intentional for cache efficiency and WASM compatibility. The `manifolds::Points` type alias captures this.

### Key abstractions

- **`Manifold` trait** (`manifolds.rs`): Defines `init_points`, `pairwise_distances`, `project_to_tangent`, `exp_map`, `center`, `scaling_loss`. Three implementations: `Euclidean` (k=0), `Hyperboloid` (k<0), `Sphere` (k>0). Factory: `create_manifold(curvature)`.
- **`TrainingConfig`** (`config.rs`): All hyperparameters. Curvature sign selects geometry.
- **`fit_embedding`** (`embedding.rs`): Main training loop. Takes data, config, and optional `FnMut(StepResult) -> bool` callback for per-iteration rendering/logging.
- **`RiemannianSGDMomentum`** (`optimizer.rs`): Optimizer with parallel transport for momentum on curved spaces.

### Additional modules

- **`curvature_detection.rs`**: Detects geometry (Euclidean/spherical/hyperbolic) and intrinsic dimension from a pairwise distance matrix using shell density histograms + OLS fitting. Native-only (no WASM).
- **`kl_divergence.rs`**: Global t-SNE similarity matrix (`compute_global_similarities`) using `(1 + d²)` kernel (Zhou & Sharpee loss variant that emphasizes large distances).
- **`scaling_loss.rs`**: Radial regularization for hyperbolic embeddings — penalizes geodesic spread from origin. Returns `(loss, ambient_gradient)`; caller must project gradient to tangent space.
- **`data.rs`**: MNIST loader from IDX binary format. `#[cfg(not(target_arch = "wasm32"))]` — not compiled for WASM.

### Pipeline flow

Input data → `compute_perplexity_affinities` (affinities.rs) → `fit_embedding` loop: `compute_q_matrix` (kernels.rs) → `kl_gradient` → optimizer step (exp_map + project) → optional callback with `StepResult`.

### Ambient dimension

For curved spaces, `ambient_dim = embed_dim + 1` (hyperboloid in R^{d+1}, sphere in R^{d+1}). Euclidean: `ambient_dim = embed_dim`.

### Built-in PRNG

`synthetic_data::Rng` implements xoshiro256** to avoid external dependencies. Used for data generation and initialization.
