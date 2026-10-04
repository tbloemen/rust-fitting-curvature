# Visualising High-Dimensional Data using Non-Euclidean t-SNE

This is the code accompanying the MSc thesis _Visualising High-Dimensional Data using Non-Euclidean t-SNE_.
It implements t-SNE on constant-curvature manifolds (hyperbolic, Euclidean and spherical), searches each dataset's hyperparameters with multi-objective Bayesian optimisation (qParEGO), and turns the resulting Pareto fronts into the thesis figures.

## Requirements

- Rust 1.82 or newer (`cargo`).
- For the figures only: a system font stack (`libfontconfig1-dev` on Debian/Ubuntu).
- Optional: [uv](https://docs.astral.sh/uv/) for the dataset preparation scripts, and Node.js plus `wasm-pack` for the web viewer.

The real datasets are included under `www/public/data/`.
`scripts/download_mnist.sh`, `scripts/generate_pbmc_pca.py` and `scripts/generate_wordnet_mammals.py` recreate them from source.

## 1. Run the optimizer

Build once:

```bash
cargo build --release -p fitting-optimizer
```

One run optimises a single **(setting, dataset, geometry)** cell and appends one JSON line per trial to its output file:

```bash
./target/release/optimizer \
  --mode pareto --dataset tree --geometry hyperbolic --experiment all_off \
  --n-trials 1000 --n-seeds 3 --n-samples 5000 \
  --data-path ./www/public/data \
  --resume \
  --output results/all_off_tree_n5000_hyperbolic.jsonl
```

- `--dataset`: `grid`, `sphere`, `hyperbolic_shells`, `tree` (synthetic) or `mnist`, `fashion_mnist`, `pbmc`, `wordnet_mammals` (real).
- `--geometry`: `hyperbolic`, `euclidean` or `spherical`.
- `--experiment` (the loss setting): `all_off` (KL only), `centering_only`, `global_only`, `norm_only` (not spherical) or `all_free`.
- `--resume` continues an interrupted run from its JSONL, so the same command can simply be re-run.
- Name the output `<setting>_<dataset>_n<N>_<geometry>.jsonl`:
  the analysis reads the cell from the filename.
  Next to it the optimizer writes the run's Pareto front as `*_pareto_*.json`.

A run uses every core by default (`--threads` caps it).
At N = 5000 one run takes many hours, so the full grid is a multi-day job locally.
Two scripts run the thesis grid one cell at a time, each safe to interrupt and restart:

```bash
sh run_5000_local_main.sh   # all_off × {hyperbolic, euclidean, spherical} × all datasets
sh run_5000_local_all.sh    # the four loss settings × {hyperbolic, euclidean} × all datasets

# Subsets via environment variables, e.g.:
DATASETS="tree mnist" THREADS=8 sh run_5000_local_main.sh
EXPERIMENTS="centering_only global_only all_free" LOSS_GEOMETRIES=spherical sh run_5000_local_all.sh  # the spherical loss settings
```

Both write to `results/n5000/` (override with `RESULTS_DIR`) with a log per cell.

## 2. Analyse the results and make the figures

Point `RES` at a directory of result files.
The thesis figures were made from the cluster run in `results-rgyr/` (see section 3);
for your own local run use `results/n5000`.

```bash
RES=results-rgyr
cargo build --release -p fitting-analysis --features plots

# R2 indicator of every cell's Pareto front, under each preference region
./target/release/r2 stats --results-dir $RES --out $RES/r2_local_obj6.jsonl

# ΔR2 of each loss setting over the all_off baseline (Experiment 4)
./target/release/r2 aggregate $RES/r2_local_obj6.jsonl \
  --deltas $RES/r2_delta_obj6.jsonl

# The parameter-free cross-check: additive ε-indicator against the baseline
./target/release/r2 compare --results-dir $RES --out $RES/r2_epsilon_obj6.jsonl

# Experiment 1: matched against mismatched geometry
./target/release/exp1 --results-dir $RES --out $RES/exp1_geometry_match_obj6.jsonl

# All figures (Experiments 1 to 4) as SVG
./target/release/figures --results-dir $RES --out-dir plots \
  --exp1 $RES/exp1_geometry_match_obj6.jsonl \
  --r2-local $RES/r2_local_obj6.jsonl \
  --r2-delta $RES/r2_delta_obj6.jsonl \
  --r2-epsilon $RES/r2_epsilon_obj6.jsonl
```

The figures land in `plots/`;
for `RES=results-rgyr` they are identical to the ones in `plots-rgyr/`.
Add `--exp 2` to render a single experiment, and `--n 1000 5000` for other sample sizes.

The curvature-fit table of the exploratory chapter comes from an example binary:

```bash
cargo run --release -p fitting-core --example three_arm_residuals \
  -- --jsonl results/three_arm_residuals.jsonl
```

Its four Wilson residual figures come from two more, which need the `plot-examples` feature and write to `plots/`:

```bash
cargo run --release -p fitting-core --features plot-examples --example wilson_residual_curve
cargo run --release -p fitting-core --features plot-examples --example wilson_residual_curve_real
```

## 3. Run on a SLURM cluster (DelftBlue)

The thesis results come from the N = 5000 sweep in `slurm/submit_rgyr_5000.sh`.
A single run takes about 60 to 72 hours, so each cell is split into chained 24-hour jobs that pick up with `--resume`.

Before submitting, set your account, partition and resources in the `#SBATCH` header of `slurm/run_5000_rgyr.sh` and your login in `slurm/sync.sh` and `slurm/sync_back.sh`.

```bash
# 1. Locally: copy the code and data to the cluster
sh slurm/sync.sh

# 2. On the login node, in the synced directory: build there, so the binary
#    links against the cluster's libraries
cargo build --release --locked

# 3. Test one cell first, and check its output
sbatch --export=ALL,DATASET=sphere,EXPERIMENT=all_off,GEOMETRY=spherical \
  slurm/run_5000_rgyr.sh

# 4. Submit the grid, one geometry at a time (each cell is CHUNKS chained jobs)
GEOMETRIES=spherical  sh slurm/submit_rgyr_5000.sh
GEOMETRIES=hyperbolic sh slurm/submit_rgyr_5000.sh
GEOMETRIES=euclidean  sh slurm/submit_rgyr_5000.sh
#   DATASETS="tree mnist" or CHUNKS=5 narrow or extend a submission

# 5. Locally, once the jobs are done: pull the results back
REMOTE_RESULTS=~/fitting/results-rgyr LOCAL_RESULTS=./results-rgyr \
  sh slurm/sync_back.sh
```

Jobs write to `/scratch/$USER/fitting/results-rgyr` and copy each file to `~/fitting/results-rgyr` when they stop.
Then continue with section 2 using `RES=results-rgyr`.

## Web viewer

```bash
sh build.sh   # builds the WebAssembly package and starts the dev server
```

## Tests

```bash
cargo test --workspace
cd www && npm test
```
