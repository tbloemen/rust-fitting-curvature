"""
Turn the ΔR2 rows emitted by `crates/analysis/src/bin/r2.rs aggregate --deltas`
into a Typst `#figure(table(...))` fragment for the thesis.

Usage:
    uv run python scripts/r2_delta_typst.py --datasets real --n 1000

The input defaults to `results/r2_delta.jsonl`, where `r2 aggregate --deltas`
writes it, and the output to `tables/r2_delta_<datasets>_n<N>.typ`, alongside
the other generated Typst tables -- so both flags are only needed to override
those:

    uv run python scripts/r2_delta_typst.py \
        --input results/r2_delta.jsonl \
        --output tables/r2_delta_real_n1000.typ \
        --datasets real --n 1000

The fragment is self-contained: numbers are pre-rendered as Typst math, so it
needs no package imports beyond what `docs/thesis/main.typ` already provides,
and it can be `#include`d wherever `<tab:r2-delta-real-n1000>` is referenced.

One row per (dataset, geometry, setting), one column per preference region of
@preference-regions.  The `all_off` row that opens each geometry block carries
the baseline *level* $R_2$; the rows under it carry the *gain* over it.  Both
halves are scaled by $10^3$, so the block reads as one column of numbers rather
than two scales stacked on top of each other.

Stdlib only.
"""

import argparse
import json
import math
import os


def _ensure_parent(path: str) -> None:
    """Create the output's directory, so the default `tables/` works on a fresh
    clone (the directory holds only generated fragments, so it may not exist)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

# The five synthetic generators of known intrinsic geometry, flat-first and then
# grouped by curvature sign, and the four real datasets of Experiment 2.  Rows
# absent from the JSONL are skipped; rows present but not listed are appended at
# the end, so a partial run still produces output.
DATASET_GROUPS = {
    "synthetic": ["grid", "sphere", "antipodal_clusters", "tree", "hyperbolic_shells"],
    "real": ["mnist", "fashion_mnist", "pbmc", "wordnet_mammals"],
}

DATASET_LABEL = {
    "grid": "grid",
    "sphere": "sphere",
    "antipodal_clusters": "antipodal",
    "tree": "tree",
    "hyperbolic_shells": "hyp. shells",
    "mnist": "MNIST",
    "fashion_mnist": "F-MNIST",
    "pbmc": "PBMC",
    "wordnet_mammals": "WordNet",
}

# Curvature-sign order, so the flat arm sits between the two curved ones.
GEOMETRY_ORDER = ["spherical", "euclidean", "hyperbolic"]
GEOMETRY_LABEL = {"spherical": "sph", "euclidean": "euc", "hyperbolic": "hyp"}

BASELINE = "all_off"

# @tab:loss-ablations order: the baseline, the three single-loss settings, then
# the joint one.  `rms_anchored` is deliberately absent: it fixes the gauge for
# the curvature search of Experiment 3 rather than ablating a loss term, and it
# exists for hyperbolic only.  `--settings` puts it back.
SETTING_ORDER = [BASELINE, "centering_only", "global_only", "norm_only", "all_free"]

# Region order follows `METRIC_PAIRS` in crates/analysis/src/objectives.rs, not
# the JSONL's alphabetical key order: the five per-metric regions read in the
# same order as every other per-metric table in the thesis.
REGIONS = [
    ("all", '$W_"all"$'),
    ("trustworthiness", '$W_"trust"$'),
    ("continuity", '$W_"cont"$'),
    ("normalized_stress", '$W_"stress"$'),
    ("shepard_goodness", '$W_"shep"$'),
    ("neighborhood_hit", '$W_"nh"$'),
    ("manifold", '$W_"man"$'),
    ("projected", '$W_"proj"$'),
]

# Every entry in the table is scaled by this.  |ΔR2| spans 8e-5 to 0.13, so the
# unscaled numbers would need four decimals to say anything and would then read
# as a column of zeros.
SCALE = 1000
SCALE_LABEL = "10^(-3)"
# Smallest scaled magnitude `fixed` prints; anything under half of it is a gain
# the table cannot resolve.
RESOLUTION = 0.01


def fixed(x, signed):
    """Render `x` (already scaled) as Typst math at a common decimal scale.

    Not significant figures: the entries here span three decades and the point
    of the column is to be compared *down*, so decimals are dropped as the
    magnitude grows rather than held constant.  The result is 3 to 4 significant
    figures everywhere and a uniform column width of at most six characters.
    """
    if x is None or not math.isfinite(x):
        return "---"
    magnitude = abs(x)
    # Below half the last printed digit the sign is not information: 16 of the
    # 1712 gains sit here, the smallest at 2e-11, which is a front that did not
    # move rather than a gain too small to see. Printing `+0.00` would claim a
    # direction the number does not carry.
    if magnitude < 0.5 * RESOLUTION:
        return "$approx 0$"
    decimals = 2 if magnitude < 10 else (1 if magnitude < 100 else 0)
    return f"${x:{'+' if signed else ''}.{decimals}f}$"


def cell(text, bold=False):
    """One table cell.

    Bold goes *inside* the math block: Typst's `*...*` around a math block is
    unreliable, `bold(...)` within it is not.
    """
    if bold and text.startswith("$") and text.endswith("$"):
        return f"[$bold({text[1:-1]})$]"
    return f"[{'*' + text + '*' if bold else text}]"


def load(path, n, datasets, settings):
    """Rows for one sample size, indexed as dataset → geometry → setting → region."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")

    wanted = set(DATASET_GROUPS[datasets])
    table = {}
    for r in rows:
        if r["n"] != n or r["dataset"] not in wanted or r["setting"] not in settings:
            continue
        by_setting = table.setdefault(r["dataset"], {}).setdefault(r["geometry"], {})
        by_setting.setdefault(r["setting"], {})[r["region"]] = r

    if not table:
        raise SystemExit(f"no {datasets} rows with n = {n} in {path}")

    # A stale or half-regenerated table is the failure mode worth catching here:
    # `delta_r2` is written by the same pass that writes `r2` and `r2_baseline`,
    # so if the identity does not hold, the file mixes runs and every number
    # below it is suspect.
    names = [name for name, _ in REGIONS]
    for dataset, arms in table.items():
        for geometry, by_setting in arms.items():
            for setting, by_region in by_setting.items():
                missing = [m for m in names if m not in by_region]
                if missing:
                    raise SystemExit(
                        f"{dataset}/{geometry}/{setting} at n={n} is missing "
                        f"region(s) {', '.join(missing)}"
                    )
                for region, r in by_region.items():
                    delta, base = r["delta_r2"], r["r2_baseline"]
                    if delta is None or base is None:
                        raise SystemExit(
                            f"{dataset}/{geometry}/{setting}/{region} at n={n} "
                            "has no baseline to difference against"
                        )
                    if abs((base - r["r2"]) - delta) > 1e-12:
                        raise SystemExit(
                            f"{dataset}/{geometry}/{setting}/{region} at n={n}: "
                            f"delta_r2 {delta} != r2_baseline - r2 {base - r['r2']}"
                        )
    return table


def ordered(present, order):
    """`order` first, then anything present but unlisted, so a partial run still
    produces output rather than dropping rows silently."""
    out = [k for k in order if k in present]
    return out + [k for k in present if k not in order]


def build(table, n, datasets, settings):
    dataset_order = ordered(table, DATASET_GROUPS[datasets])

    # Rows per (dataset, geometry) block, needed up front for the rowspans.
    block_rows = {}
    for dataset in dataset_order:
        for geometry in ordered(table[dataset], GEOMETRY_ORDER):
            block_rows[dataset, geometry] = ordered(
                table[dataset][geometry], settings
            )

    n_cols = 3 + len(REGIONS)
    align = ["left", "left", "left"] + ["right"] * len(REGIONS)

    # Row 1 groups the regions by the question they ask; the three identity
    # columns span both header rows instead.
    row1 = [
        "table.cell(rowspan: 2, align: horizon)[*Dataset*]",
        "table.cell(rowspan: 2, align: horizon)[*Geom.*]",
        "table.cell(rowspan: 2, align: horizon)[*Setting*]",
        'table.cell(rowspan: 2, align: horizon + right)[$W_"all"$]',
        "table.cell(colspan: 5, align: center)[*Per-metric*]",
        "table.cell(colspan: 2, align: center)[*Surface*]",
    ]
    row2 = [f"[{label}]" for _, label in REGIONS[1:]]

    lines = [
        "// Generated by scripts/r2_delta_typst.py -- do not edit by hand.",
        # Eleven columns of numbers do not fit a page, and a `#figure` is not
        # breakable by default.  The rule is scoped to this content block so it
        # does not leak into the rest of the chapter.
        "#[",
        "  #show figure: set block(breakable: true)",
        "  #figure(",
        "    table(",
        f"      columns: {n_cols},",
        f"      align: ({', '.join(align)}),",
        "      stroke: none,",
        # Eleven columns at the default 5pt side inset spend ~4cm of the text
        # width on padding alone. Same treatment as `exp1_common.py`'s `render`.
        "      inset: (x: 2.5pt, y: 3pt),",
        "      table.hline(stroke: 0.8pt),",
        "      table.header(",
        "        " + ", ".join(row1) + ",",
        "        " + ", ".join(row2) + ",",
        "      ),",
        "      table.hline(stroke: 0.5pt),",
    ]

    for di, dataset in enumerate(dataset_order):
        if di:
            lines.append("      table.hline(stroke: 0.5pt),")
        geometries = ordered(table[dataset], GEOMETRY_ORDER)

        for gi, geometry in enumerate(geometries):
            if gi:
                lines.append("      table.hline(stroke: 0.2pt, start: 1),")
            rows = block_rows[dataset, geometry]
            by_setting = table[dataset][geometry]

            # Per region, the best gain in this block -- bolded below.  Only a
            # positive gain can win: bolding the least bad of a block of losses
            # would read as an endorsement.
            best = {}
            for region, _ in REGIONS:
                gains = [
                    (by_setting[s][region]["delta_r2"], s)
                    for s in rows
                    if s != BASELINE
                ]
                # A gain the table renders as `≈ 0` is not a winner: bolding it
                # would point at a number the reader cannot read a sign off.
                gains = [
                    (d, s)
                    for d, s in gains
                    if d is not None and d * SCALE >= 0.5 * RESOLUTION
                ]
                if gains:
                    best[region] = max(gains)[1]

            for si, setting in enumerate(rows):
                cells = []
                if si == 0:
                    # The dataset label spans one geometry block rather than all
                    # three: a `table.cell` rowspan is *not* repeated when the
                    # table breaks across a page, so a 14-row span leaves the
                    # rows above the break unlabelled. Repeating it per block
                    # bounds that loss at a few rows, and the heavier rule
                    # between datasets still groups the three blocks by eye.
                    # The label is dimmed on repeats so the grouping reads as
                    # one dataset rather than three.
                    label = DATASET_LABEL.get(dataset, dataset)
                    if gi:
                        label = f"#text(fill: luma(40%))[{label}]"
                    cells.append(f"table.cell(rowspan: {len(rows)})[{label}]")
                    cells.append(
                        f"table.cell(rowspan: {len(rows)})"
                        f"[{GEOMETRY_LABEL.get(geometry, geometry)}]"
                    )
                cells.append(f"[`{setting}`]")

                by_region = by_setting[setting]
                for region, _ in REGIONS:
                    r = by_region[region]
                    if setting == BASELINE:
                        # The level, not a gain: unsigned, and never bolded --
                        # it is what the rows below are measured against.
                        cells.append(cell(fixed(r["r2"] * SCALE, signed=False)))
                    else:
                        delta = r["delta_r2"]
                        cells.append(
                            cell(
                                fixed(None if delta is None else delta * SCALE, True),
                                bold=best.get(region) == setting,
                            )
                        )
                lines.append("      " + ", ".join(cells) + ",")

        # Settings absent for this dataset entirely (spherical `norm_only`) are
        # not rows at all; the caption says why rather than the table carrying a
        # line of dashes.

    lines.append("      table.hline(stroke: 0.8pt),")
    lines.append("    ),")

    kind = "synthetic datasets of known intrinsic geometry" if datasets == "synthetic" else "real datasets"
    lines += [
        "    caption: [Experiment 2: R2 indicator gain $Delta R_2$ of each loss-weight setting over the",
        f"    `all_off` baseline on the {kind}, under each preference region of @preference-regions",
        f"    ($N = {n}$).",
        "",
        f"    Every entry is multiplied by ${SCALE}$, so a table entry of $1$ is an indicator value",
        f"    of ${SCALE_LABEL}$.",
        "    The `all_off` row opening each geometry block is the baseline *level* $R_2$ of that front",
        "    (@eq:r2), a cost: smaller is better.  Every row below it is the *gain*",
        '    $Delta R_2 = R_2("all_off") - R_2("setting")$ of @eq:r2-gain, formed',
        "    baseline-minus-setting so that a positive value means the setting improved on the",
        "    baseline.  Levels are therefore unsigned and gains always carry a sign.",
        "    Within each geometry block the largest positive gain per region is set in bold; a block",
        "    with no positive gain has no bold entry.  An entry of $approx 0$ is a gain below the",
        "    resolution printed here, where the front barely moved and the sign carries nothing.",
        "",
        '    The columns are the weight sets of @preference-regions: $W_"all"$ is the full simplex and',
        "    is the only one under which $Delta R_2$ is a claim about overall front quality; the five",
        "    per-metric regions hold the weight vectors placing at least half their mass on that",
        '    metric\'s two objectives; $W_"man"$ and $W_"proj"$ hold those supported entirely on the',
        "    five manifold, respectively the five projected, objectives.  A setting that helps under",
        "    one priority and costs under another is visible as a sign change along its row.",
        "",
        "    `norm_only` is absent from every spherical block because the depth-norm loss is",
        "    undefined on the sphere (@norm-loss).  `rms_anchored` is not an ablation of a loss term",
        "    and is reported with the curvature results (@curvature-magnitude-results) instead.",
        "  ],",
    ]

    label = f"tab:r2-delta-{datasets}-n{n}"
    lines.append(f"  ) <{label}>")
    lines.append("]")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/r2_delta.jsonl")
    ap.add_argument("--output", help="default: tables/r2_delta_<datasets>_n<N>.typ")
    ap.add_argument("--n", type=int, default=1000, help="sample size to tabulate")
    ap.add_argument(
        "--datasets",
        default="real",
        choices=sorted(DATASET_GROUPS),
        help="which dataset group to tabulate",
    )
    ap.add_argument(
        "--settings",
        default=",".join(SETTING_ORDER),
        help="comma-separated settings, in row order; must include the baseline",
    )
    a = ap.parse_args()

    settings = [s for s in a.settings.split(",") if s]
    if BASELINE not in settings:
        raise SystemExit(f"--settings must include the {BASELINE!r} baseline")
    out = a.output or f"tables/r2_delta_{a.datasets}_n{a.n}.typ"

    table = load(a.input, a.n, a.datasets, settings)
    _ensure_parent(out)
    with open(out, "w") as f:
        f.write(build(table, a.n, a.datasets, settings))
    print(f"wrote {out} (n={a.n}, datasets={a.datasets})")


if __name__ == "__main__":
    main()
