#!/usr/bin/env python3
"""One-time exact correction of ``results/kappa_data.jsonl`` for the d_rms fix.

The first ``--mode detect`` run computed ``d_rms`` as the RMS distance from the
centroid, ``sqrt(S / 2n²)`` with ``S = Σ_ij d_ij²``. The thesis (methods
§gauge-fixing) defines ``d_rms`` as the input RMS *pairwise* distance,
``sqrt(S / n(n-1))``. Both come from the same S, so the correction is exact and
needs no re-run of the detector:

    d_rms_pairwise = d_rms_centroid · sqrt(2n / (n-1))
    every κ field  = |curvature| · d_rms²  →  scales by  2n / (n-1)

The signed curvatures (r*, δ) are independent of d_rms and stay put. This script
rewrites the file in place (with a .bak backup) so it matches both the thesis
definition and the corrected detect.rs. Idempotent guard via a ``d_rms_convention``
marker it stamps on each row.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

KAPPA_FIELDS = ["kappa_data", "sph_kappa", "hyp_kappa", "delta_kappa"]


def main(path_str: str = "results/kappa_data.jsonl") -> int:
    path = Path(path_str)
    if not path.exists():
        print(f"{path} not found", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in open(path) if line.strip()]

    changed = 0
    for r in rows:
        if r.get("d_rms_convention") == "pairwise":
            continue  # already corrected
        n = int(r["n_samples"])
        ratio = math.sqrt(2.0 * n / (n - 1))  # d_rms scale
        r["d_rms"] = r["d_rms"] * ratio
        for f in KAPPA_FIELDS:
            if f in r and r[f] is not None:
                r[f] = r[f] * (ratio * ratio)
        r["d_rms_convention"] = "pairwise"
        changed += 1

    if changed == 0:
        print("nothing to correct (all rows already 'pairwise')")
        return 0

    shutil.copy(path, path.with_suffix(path.suffix + ".bak"))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"corrected {changed} rows in {path} (backup at {path}.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
