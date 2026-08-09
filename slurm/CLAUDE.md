### SLURM orchestration (`slurm/`)

`run_*.sh` are job templates; `submit_*.sh` enumerate `(dataset, geometry[, experiment])` and submit jobs. The `*_5000.sh` variants chain `CHUNKS` (default 4, override via env) jobs per cell with `--dependency=afterany` and `--time=23:55:00`, restoring the prior chunk's checkpoint from `$HOME` to scratch at startup and passing `--resume`. Results carry an `_n5000` marker so they don't collide with the 1000-sample runs.

See `../crates/optimizer/CLAUDE.md` for what `--resume` and chunking mean on the optimizer side.

Everything here submits **embedding** work — that is the only thing that still needs a cluster. The `run_hv.sh` / `submit_hv.sh` pair was deleted once the front-summary stage was ported to Rust: the 22-way array it used to need now runs in ~1.3s locally, single-threaded (`crates/analysis`, `r2 stats`). Don't reintroduce a SLURM job for post-hoc analysis of the results JSONL without timing it locally first — the sweeps are O(n²) per iteration and genuinely need the cluster, the analysis is a few thousand weighted maxima over a few hundred front points and does not.
