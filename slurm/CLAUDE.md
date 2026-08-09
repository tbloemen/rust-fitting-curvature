### SLURM orchestration (`slurm/`)

`run_*.sh` are job templates; `submit_*.sh` enumerate `(dataset, geometry[, experiment])` and submit jobs. The `*_5000.sh` variants chain `CHUNKS` (default 4, override via env) jobs per cell with `--dependency=afterany` and `--time=23:55:00`, restoring the prior chunk's checkpoint from `$HOME` to scratch at startup and passing `--resume`. Results carry an `_n5000` marker so they don't collide with the 1000-sample runs.

See `../crates/optimizer/CLAUDE.md` for what `--resume` and chunking mean on the optimizer side.
