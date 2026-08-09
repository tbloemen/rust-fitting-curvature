#!/usr/bin/env python3
"""Stage 2 of the hypervolume analysis: ΔH over baseline + cross-dataset test.

Reads the per-cell HV table(s) produced by ``hv_stats.py`` (pass one or more
files or shard files; they are concatenated) and, for each
(N, geometry, setting) group, reports:

* ΔH_d = HV(setting, dataset d) − HV(all_off, dataset d) for every dataset d that
  has both the setting and the ``all_off`` baseline — the per-dataset gain a loss
  term buys over the unregularised run.
* the mean/median ΔH across datasets, and a **Wilcoxon signed-rank test** of
  ΔH ≠ 0 across datasets (Demšar 2006's recommended paired cross-dataset test).

``all_off`` itself is the baseline, so it is reported with ΔH = 0 and no test.
Unlike stage 1, this uses scipy and is meant to run locally on the (small) HV
table — never on the cluster.

Usage::

    python hv_aggregate.py hv_local.jsonl
    python hv_aggregate.py hv_shard_*.jsonl --csv hv_delta.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - scipy is a stage-2-only dep
    wilcoxon = None

BASELINE = "all_off"


def load_hv_table(paths: list[Path]) -> list[dict]:
    """Concatenate per-cell HV records from one or more stage-1 JSONL files.

    A later file wins on a duplicate ``stem`` (e.g. a re-run shard), so mixing a
    coarse local table with a few refined shards keeps the refined values.
    """
    by_stem: dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_stem[rec["stem"]] = rec
    return list(by_stem.values())


def compute_deltas(table: list[dict]) -> list[dict]:
    """ΔH rows for every (n, geometry, setting, dataset) against the baseline."""
    # hv[(n, geometry, setting, dataset)] = hv value
    hv: dict[tuple, float] = {}
    for r in table:
        hv[(r["n"], r["geometry"], r["setting"], r["dataset"])] = r["hv"]

    rows: list[dict] = []
    for (n, geom, setting, dataset), value in hv.items():
        base = hv.get((n, geom, BASELINE, dataset))
        rows.append(
            {
                "n": n,
                "geometry": geom,
                "setting": setting,
                "dataset": dataset,
                "hv": value,
                "hv_baseline": base,
                "delta_hv": (value - base) if base is not None else None,
            }
        )
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Per-(n, geometry, setting) ΔH summary with the cross-dataset signed test."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["n"], r["geometry"], r["setting"])].append(r)

    out: list[dict] = []
    for (n, geom, setting), grp in sorted(groups.items()):
        deltas = [g["delta_hv"] for g in grp if g["delta_hv"] is not None]
        summary = {
            "n": n,
            "geometry": geom,
            "setting": setting,
            "n_datasets": len(deltas),
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "median_delta": float(np.median(deltas)) if deltas else None,
            "n_positive": int(sum(d > 0 for d in deltas)),
            "wilcoxon_p": None,
        }
        # The baseline vs itself is all-zero; a signed-rank test is undefined and
        # meaningless there, so skip it.
        if setting != BASELINE and len(deltas) >= 1 and any(d != 0 for d in deltas):
            if wilcoxon is not None:
                try:
                    summary["wilcoxon_p"] = float(wilcoxon(deltas).pvalue)
                except ValueError:
                    summary["wilcoxon_p"] = None
        out.append(summary)
    return out


def print_report(summaries: list[dict]) -> None:
    hdr = f"{'N':>5} {'geometry':<11} {'setting':<15} {'#ds':>4} {'mean ΔH':>10} {'median ΔH':>10} {'#pos':>5} {'wilcoxon p':>11}"
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        mean = "n/a" if s["mean_delta"] is None else f"{s['mean_delta']:+.5f}"
        med = "n/a" if s["median_delta"] is None else f"{s['median_delta']:+.5f}"
        p = "n/a" if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.4f}"
        star = " *" if (s["wilcoxon_p"] is not None and s["wilcoxon_p"] < 0.05) else ""
        print(f"{s['n']:>5} {s['geometry']:<11} {s['setting']:<15} {s['n_datasets']:>4} {mean:>10} {med:>10} {s['n_positive']:>5} {p:>11}{star}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", type=Path, nargs="+", help="Stage-1 HV JSONL file(s) / shard files.")
    ap.add_argument("--csv", type=Path, default=None, help="Optional path to write the per-dataset ΔH rows as CSV.")
    args = ap.parse_args(argv)

    if wilcoxon is None:
        print("warning: scipy not available; Wilcoxon p-values will be n/a", file=sys.stderr)

    table = load_hv_table(args.tables)
    if not table:
        print("empty HV table", file=sys.stderr)
        return 1
    rows = compute_deltas(table)
    summaries = summarise(rows)
    print_report(summaries)

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["n", "geometry", "setting", "dataset", "hv", "hv_baseline", "delta_hv"])
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["n"], r["geometry"], r["setting"], r["dataset"])))
        print(f"\nwrote per-dataset ΔH rows to {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
