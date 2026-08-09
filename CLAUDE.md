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

# Run a hyperparameter search locally (see crates/optimizer/CLAUDE.md for modes)
cargo run --release -p fitting-optimizer -- \
  --mode pareto --dataset tree --geometry hyperbolic --experiment all_off --n-samples 200

# Analyse the sweeps: R2 indicator (~0.2s for the whole table), then ΔR2 + the test
cargo run --release -p fitting-analysis --bin r2 -- stats --out r2_local.jsonl
cargo run --release -p fitting-analysis --bin r2 -- aggregate r2_local.jsonl --csv r2_delta.csv   # Friedman + Holm -> r2_tests.csv
cargo run --release -p fitting-analysis --bin r2 -- recommend --csv recommendations.csv

# Thesis result figures (local only — needs fontconfig, hence the feature gate)
cargo run --release -p fitting-analysis --features plots --bin figures

# Build WASM package
wasm-pack build crates/web --target web --out-dir ../../www/pkg

# Full build + dev server (requires wasm-pack, Node.js)
./build.sh

# Dev server only (after WASM is built)
cd www && npm run dev
```

No linter is configured. Rust edition 2024.

## Architecture

**Workspace** with four crates:
- `crates/core` (`fitting-core`) — Pure Rust library, zero dependencies. All embedding algorithms live here.
- `crates/web` (`fitting-web`) — WASM bindings via wasm-bindgen. Uses plotters + plotters-canvas for HTML5 canvas rendering and lol_alloc as WASM allocator.
- `crates/optimizer` (`fitting-optimizer`) — Native-only CLI binary (bin name `optimizer`) that drives hyperparameter search over `fitting-core`. Has dependencies (clap, indicatif, serde) — the "zero dependencies" rule applies to `core` only.
- `crates/analysis` (`fitting-analysis`) — Post-hoc analysis of the sweeps, all of it local: Pareto fronts, the R2 indicator over preference regions and the ΔR2 statistics (`r2` binary; the full table takes ~0.2s, so this needs no cluster job) plus the thesis figures (`figures` binary, gated behind the `plots` feature so plotters/fontconfig stay off the default dependency graph).

**Web frontend** (`www/`): Vite + vite-plugin-wasm. JS imports WASM pkg as `"fitting-web": "file:./pkg"` with `await init()` pattern.

**Python** is down to one file: `analyze_hyperparams.py`, the exploratory
hyperparameter tool (sklearn/scipy/matplotlib, local only). The indicator and
thesis-figure stages were ported to `crates/analysis`.

Crate-specific detail lives in nested CLAUDE.md files and loads only when you work in that directory: `crates/core/CLAUDE.md` (data layout, manifold abstractions, curvature detection submodules, pipeline flow), `crates/optimizer/CLAUDE.md` (run modes, GP/qParEGO internals, resume semantics), `crates/analysis/CLAUDE.md` (orientation, the weight simplex and preference regions, the scipy ports), `slurm/CLAUDE.md` (cluster job orchestration).
