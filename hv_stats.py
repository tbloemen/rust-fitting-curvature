#!/usr/bin/env python3
"""Stage 1 of the hypervolume analysis: per-cell Monte-Carlo hypervolume.

For every experiment cell (one results ``.jsonl`` file = one
(setting, dataset, N, geometry) run) this computes the dominated hypervolume of
its 10-objective Pareto front in the oriented unit box ``[0, 1]^10`` — i.e. the
fraction of the box the front dominates — by Monte-Carlo integration.  The output
is one JSON object per cell (see ``--out``), consumed by ``hv_aggregate.py`` which
forms ΔH over the ``all_off`` baseline and runs the cross-dataset test.

This module is deliberately **numpy-only** (via ``pareto_utils``) so it runs on a
bare cluster compute node.  It has two run modes for the same code path:

* **Local** — ``python hv_stats.py --results-dir results --out hv_local.jsonl``
  processes every cell, optionally across worker processes (``--jobs``).  Use a
  modest ``--n-mc`` (default 2e6) for a quick turnaround on your machine.

* **Cluster** — a SLURM array (see ``slurm/run_hv.sh``) launches one task per
  shard; each task passes ``--shard i --n-shards N`` to take an interleaved slice
  of the cells and a high ``--n-mc`` (e.g. 5e7) for a tight standard error, and
  writes its own shard file.  Concatenating the shard files gives the full table.

The per-cell RNG seed is derived deterministically from the cell name and
``--seed`` (``crc32(stem) ^ seed``), so a cell's HV estimate is identical whether
it is computed locally, in a shard, or re-run later — sharding never changes a
result, only which task computes it.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pareto_utils as pu


def discover_cells(results_dir: Path) -> list[tuple[Path, pu.Cell]]:
    """Every ``(path, Cell)`` for trial-results JSONL files under *results_dir*.

    Front files (``*_pareto_*.json``) and anything whose stem doesn't parse as a
    cell are skipped.  Sorted by stem so shard assignment is stable.
    """
    out: list[tuple[Path, pu.Cell]] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        cell = pu.parse_cell_stem(path.stem)
        if cell is not None:
            out.append((path, cell))
    return out


def _cell_seed(stem: str, base_seed: int) -> int:
    """Deterministic per-cell RNG seed, independent of shard/worker layout."""
    return (zlib.crc32(stem.encode()) ^ (base_seed & 0xFFFFFFFF)) & 0xFFFFFFFF


def compute_cell(path: Path, cell: pu.Cell, n_mc: int, base_seed: int) -> dict:
    """Full HV record for one cell: identity + counts + HV and its MC error."""
    records = pu.trial_records(path)
    summary = pu.cell_hypervolume(records, n_mc=n_mc, seed=_cell_seed(path.stem, base_seed))
    return {
        "stem": path.stem,
        "setting": cell.setting,
        "dataset": cell.dataset,
        "n": cell.n,
        "geometry": cell.geometry,
        "n_mc": n_mc,
        **summary,
    }


def _worker(args: tuple[str, tuple, int, int]) -> dict:
    """ProcessPool entry point (picklable): rebuild Cell from its key tuple."""
    path_str, cell_key, n_mc, base_seed = args
    return compute_cell(Path(path_str), pu.Cell(*cell_key), n_mc, base_seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory of *.jsonl result files.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path (one line per cell).")
    ap.add_argument("--n-mc", type=int, default=2_000_000, help="Monte-Carlo samples per cell (default 2e6).")
    ap.add_argument("--seed", type=int, default=0, help="Base seed; combined with each cell's name.")
    ap.add_argument("--shard", type=int, default=0, help="This shard's index (0-based), for SLURM arrays.")
    ap.add_argument("--n-shards", type=int, default=1, help="Total number of shards; cells are split round-robin.")
    ap.add_argument("--jobs", type=int, default=1, help="Local worker processes (ignored effect if 1).")
    args = ap.parse_args(argv)

    if args.shard < 0 or args.shard >= args.n_shards:
        ap.error(f"--shard must be in [0, {args.n_shards}); got {args.shard}")

    cells = discover_cells(args.results_dir)
    if not cells:
        print(f"No result cells found under {args.results_dir}", file=sys.stderr)
        return 1

    # Round-robin shard slice: interleaving balances load better than contiguous
    # blocks because adjacent stems (same setting) have similar front sizes.
    mine = [(p, c) for i, (p, c) in enumerate(cells) if i % args.n_shards == args.shard]
    if args.n_shards > 1:
        print(f"shard {args.shard}/{args.n_shards}: {len(mine)}/{len(cells)} cells, n_mc={args.n_mc}", file=sys.stderr)
    else:
        print(f"{len(mine)} cells, n_mc={args.n_mc}, jobs={args.jobs}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        if args.jobs > 1:
            payload = [(str(p), c.key(), args.n_mc, args.seed) for p, c in mine]
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                for rec in ex.map(_worker, payload):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"  {rec['stem']}: hv={rec['hv']:.5f} ± {rec['hv_se']:.5f} (front={rec['n_front']})", file=sys.stderr)
        else:
            for p, c in mine:
                rec = compute_cell(p, c, args.n_mc, args.seed)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  {rec['stem']}: hv={rec['hv']:.5f} ± {rec['hv_se']:.5f} (front={rec['n_front']})", file=sys.stderr)

    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
