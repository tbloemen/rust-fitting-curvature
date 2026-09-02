"""
Turn the JSONL emitted by `crates/analysis/src/bin/exp1.rs` into a Typst
`#figure(table(...))` fragment for the thesis.

Usage:
    uv run python scripts/exp1_typst.py \
        --input exp1_geometry_match.jsonl \
        --output exp1_geometry_match.typ \
        --n 1000 --region all

The fragment is self-contained: numbers are pre-rendered as Typst math, so it
needs no package imports beyond what `docs/thesis/main.typ` already provides,
and it can be `#include`d wherever `<tab:geometry-match>` is referenced.

Long format, one row per (dataset, geometry): fifteen rows grouped by dataset
rather than five rows of twelve columns. The wide form hides the answer, which
is a single column here -- whether the matched geometry's Delta-R2 beats the
mismatched one's.

Stdlib only.
"""

import argparse
import json
import math

# Datasets flat-first, then grouped by the geometry they are built to have.
# Rows absent from the JSONL are skipped; rows present but not listed here are
# appended at the end, so a partial run still produces output.
ROW_ORDER = [
    "grid",
    "sphere",
    "antipodal_clusters",
    "tree",
    "hyperbolic_shells",
]

# Curvature-sign order, so the flat arm sits between the two curved ones.
GEOMETRY_ORDER = ["spherical", "euclidean", "hyperbolic"]

# Marks a Wilson fit that came to rest on the flat-ward edge of its search
# window. Such a value is the bound, not a measurement, and the caption says so.
PINNED_MARK = "#super[†]"

# Prettier row labels than the raw dataset keys.
DATASET_LABEL = {
    "grid": "grid",
    "sphere": "sphere",
    "antipodal_clusters": "antipodal",
    "tree": "tree",
    "hyperbolic_shells": "hyp. shells",
}

GEOMETRY_LABEL = {"spherical": "sph", "euclidean": "euc", "hyperbolic": "hyp"}

TRUTH_LABEL = {"spherical": "spherical", "euclidean": "euclidean", "hyperbolic": "hyperbolic"}


def sci(x, sig):
    """Render `x` as Typst math in scientific notation with `sig` significant
    figures.  Exponents in [-2, 2] are written plainly instead, since
    `0.0100` reads better than `1.00 dot 10^(-2)` in a table cell."""
    if x is None or not math.isfinite(x):
        return "---"
    if x == 0.0:
        return "$0$"
    exp = math.floor(math.log10(abs(x)))
    # Rounding to `sig` figures can carry into the next decade; renormalise so
    # the mantissa stays in [1, 10).
    if abs(round(x / (10.0**exp), sig - 1)) >= 10.0:
        exp += 1
    if -2 <= exp <= 2:
        decimals = max(sig - 1 - exp, 0)
        return f"${x:.{decimals}f}$"
    mantissa = x / (10.0**exp)
    return f"${mantissa:.{sig - 1}f} dot 10^({exp})$"


def signed(x, decimals=4):
    """A signed fixed-point number, for the two gain columns.

    Deliberately not `sci`: these columns exist to be compared *down*, and both
    span barely two decades (roughly 4e-4 to 4e-2), so a common decimal scale
    lines the magnitudes up by eye. Scientific notation also wraps to a second
    line at this column width, which makes the rows uneven.
    """
    if x is None or not math.isfinite(x):
        return "---"
    if x == 0.0:
        return "$0$"
    return f"${x:+.{decimals}f}$"


def cell(text, pinned=False, bold=False):
    """One table cell.

    `bold` is only ever applied to plain-text labels. Numbers are left alone:
    Typst's `*...*` around a math block is unreliable, and a bolded `---` is
    nonsense. The matched row is already marked twice, by the bold geometry
    label and the checkmark, which is enough.
    """
    if pinned:
        text += PINNED_MARK
    return f"[{'*' + text + '*' if bold else text}]"


def build(rows, n, region):
    rows = [r for r in rows if r["n"] == n]
    if not rows:
        raise SystemExit(f"no rows with n = {n}")

    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], {})[r["geometry"]] = r

    order = [d for d in ROW_ORDER if d in by_dataset]
    order += [d for d in by_dataset if d not in ROW_ORDER]

    setting = sorted({r["setting"] for r in rows})[0]

    # The Wilson half splits by what survives `--wilson-fallback`: the fit's own
    # statistics always, the front comparison only when the fit was run at this
    # N. Drop the columns that would be empty rather than printing dashes.
    wilson_n = sorted({r["wilson"]["wilson_n"] for r in rows if r["wilson"]})
    comparable = any(
        r["wilson"] and r["wilson"]["dominated_by_front"] is not None for r in rows
    )
    any_pinned = any(r["wilson"] and r["wilson"]["pinned"] for r in rows)

    # Three groups under a spanning header: what the detector fitted, what the
    # sweep achieved, and how the one sits against the other.
    groups = [
        ('[*Wilson fit*]', ['[$rho$]', '[$kappa$]'], ["right", "right"]),
        (
            '[*Pareto front*]',
            ['[$R_2$]', '[$Delta R_2$]', '[$tilde(kappa)$]', '[$n_"front"$]'],
            ["right", "right", "right", "right"],
        ),
    ]
    if comparable:
        groups.append(
            (
                '[*Wilson point vs. front*]',
                ['[dom.]', '[$I_epsilon (F, w)$]', '[$I_epsilon (w, F)$]'],
                ["center", "right", "right"],
            )
        )

    n_cols = 2 + sum(len(g[1]) for g in groups)
    align = ["left", "left"] + [a for g in groups for a in g[2]]

    # Row 1 spans each group; the two identity columns span both rows instead.
    row1 = ["table.cell(rowspan: 2)[*Dataset*]", "table.cell(rowspan: 2)[*Geom.*]"]
    row1 += [f"table.cell(colspan: {len(cols)}){title}" for title, cols, _ in groups]
    row2 = [c for _, cols, _ in groups for c in cols]

    lines = [
        "// Generated by scripts/exp1_typst.py -- do not edit by hand.",
        "#figure(",
        "  table(",
        f"    columns: {n_cols},",
        f"    align: ({', '.join(align)}),",
        "    stroke: none,",
        # Eleven columns at the default 5pt side inset spend ~4cm of the text
        # width on padding alone, which pushes the signed columns onto a second
        # line. Same treatment as the hand-written table in `5results.typ`.
        "    inset: (x: 2.5pt, y: 4pt),",
        "    table.hline(stroke: 0.8pt),",
        "    table.header(",
        "      " + ", ".join(row1) + ",",
        "      " + ", ".join(row2) + ",",
        "    ),",
        "    table.hline(stroke: 0.5pt),",
    ]

    for di, dataset in enumerate(order):
        if di:
            lines.append("    table.hline(stroke: 0.5pt),")
        arms = by_dataset[dataset]
        present = [g for g in GEOMETRY_ORDER if g in arms]
        present += [g for g in arms if g not in GEOMETRY_ORDER]

        for gi, geometry in enumerate(present):
            r = arms[geometry]
            w = r["wilson"]
            matched = r["matched"]

            if gi == 0:
                label = DATASET_LABEL.get(dataset, dataset)
                truth = TRUTH_LABEL.get(r["truth"], r["truth"])
                name = (
                    f"table.cell(rowspan: {len(present)})"
                    f"[{label}\\ #text(size: 0.8em)[({truth})]]"
                )
            else:
                name = None

            delta = r["delta_r2"][region] if r["delta_r2"] else None
            pinned = bool(w and w["pinned"])

            # The matched arm is marked by bolding its geometry label, which
            # costs no width; a marker column would spend a whole column on one
            # bit.
            geom = cell(GEOMETRY_LABEL.get(geometry, geometry), bold=matched)

            cells = []
            if name is not None:
                cells.append(name)
            cells += [
                geom,
                # Wilson fit
                cell(sci(w["rho"], 3) if w else "---", pinned=pinned),
                cell(sci(w["kappa"], 3) if w else "---", pinned=pinned),
                # Pareto front
                cell(sci(r["r2"][region], 3)),
                cell(signed(delta)),
                cell(sci(r["kappa_median"], 3)),
                cell(f"${r['n_front']}$"),
            ]
            if comparable:
                cells += [
                    cell("yes" if w["dominated_by_front"] else "no"),
                    cell(signed(w["eps_front_vs_wilson"])),
                    cell(signed(w["eps_wilson_vs_front"])),
                ]
            lines.append("    " + ", ".join(cells) + ",")

    lines.append("    table.hline(stroke: 0.8pt),")
    lines.append("  ),")

    n_desc = ", ".join(str(v) for v in wilson_n)
    caption = [
        "  caption: [Experiment 1: matched embedding geometry against the Euclidean baseline,",
        f"  on the synthetic datasets of known intrinsic geometry ($N = {n}$, `{setting}`,",
        f'  preference region $W_"{region}"$).',
        "  Each dataset is embedded under all three geometries; the row whose geometry matches",
        "  the data's intrinsic curvature sign is set in bold.",
        "",
    ]
    caption += [
        "  $R_2$ is the R2 indicator of the cell's Pareto front (@eq:r2), a *cost*: smaller is",
        '  better.  $Delta R_2 = R_2("euc") - R_2("geom")$ is therefore formed baseline-minus-row,',
        "  so a positive value means the row's geometry improved on the Euclidean baseline ---",
        "  the same reading direction as @eq:r2-gain.",
        "  $n_"
        + '"front"'
        + "$ is the front's size, reported because $R_2$ improves monotonically",
        "  with it: fronts here range over an order of magnitude, and the Euclidean fronts are",
        "  consistently among the smallest, which biases $Delta R_2$ *toward* the curved arms.",
    ]
    # The same column answers a second objection once the epsilon columns are
    # present, and for the opposite reason: R2 is *too* sensitive to front size,
    # the epsilon indicator is blind to it.
    if comparable:
        caption += [
            "  It also guards the $epsilon$ columns, which carry the opposite bias: being blind",
            "  to cardinality is the price of carrying no parameters, so a 46-point front can",
            "  score as a 580-point one.",
        ]
    caption += [
        "",
        "  Both $kappa$ columns are the same quantity, $|K| dot R_"
        + '"rms"'
        + "^2$ of @eq:kappa: the",
        "  Wilson arm's is measured on the configuration its fit implies, $tilde(kappa)$ is the",
        "  median over the front.  They share a gauge, so the pair reads as a comparison ---",
        "  what the detector infers about the data against what the optimiser actually chose.",
        "  Both are $0$ for the Euclidean arm, where $K = 0$ exactly.",
        "",
        # The Wilson half of the caption tracks the Wilson half of the table:
        # with `--wilson-fallback` at another N the front comparison is absent,
        # and a caption explaining three columns that are not there is worse
        # than a terse one.
        "  The Wilson columns bring the curvature detector into the same objective space.",
        "  Each arm of @tab:geometry-inference fits a constant-curvature Gram matrix $Z(r)$ to",
        "  the data; because that $Z$ *is* the Gram matrix of the model it tests for, its",
        "  retained eigen-block is an embedding.  Reconstructing it and scoring it with the",
        "  same metric code every t-SNE trial goes through puts the fit at a single point $w$ of",
        "  the ten-objective space, so it can be compared with the front rather than merely set",
        "  beside it.",
        "  $rho$ is that arm's normalised signature residual, gauged by $n dot d_"
        + '"max"'
        + "^2$;",
        "  it is also exactly the eigenvalue mass the reconstruction discarded, so it is the",
        "  reconstruction's own error and not a separate misfit score.",
    ]
    if comparable:
        caption += [
            "",
            "  The last three columns place $w$ against the front $F$ without any preference",
            "  model at all.  `dom.` says whether some front point dominates $w$ outright: `no`",
            "  means a single eigendecomposition reached a part of objective space that a",
            "  thousand qParEGO trials did not.",
            "  $I_epsilon (A, B)$ is the binary additive $epsilon$-indicator, the smallest shift",
            "  that makes $A$ cover $B$; $I_epsilon (A, B) <= 0$ iff $A$ covers $B$ outright.",
            "  It is asymmetric and both directions are reported, since neither settles anything",
            "  alone when the two cross: $I_epsilon (F, w)$ is how far the front falls short of",
            "  covering the Wilson point --- its sign is the `dom.` verdict and its magnitude the",
            "  margin --- while $I_epsilon (w, F)$ is how far that one point falls short of",
            "  covering the whole front.",
        ]
    if any_pinned:
        caption += [
            "",
            "  A dagger marks a fit pinned on the flat-ward edge of its search window, where the",
            "  reported value is the bound and not a measurement.  The hyperbolic search is capped",
            "  at the radius where $kappa$ falls to $0.01$, so a pinned hyperbolic arm reports",
            "  exactly that --- the cap is a bound on the same $kappa$ the column shows.  Every",
            "  hyperbolic arm here is pinned, so no hyperbolic reconstruction in this table is",
            "  meaningfully curved: the column tests the Euclidean null rather than a hyperbolic",
            "  alternative, which is why its $rho$ tracks the Euclidean arm's so closely.",
        ]
    if comparable and len(wilson_n) == 1 and wilson_n[0] != n:
        caption += [
            "",
            f"  The Wilson fits are run at $N = {n_desc}$ and reused here: $rho$ and $kappa$ are",
            "  dimensionless by construction, so they characterise the generator rather than the",
            "  sample.",
        ]
    elif not comparable:
        caption += [
            "",
            f"  The Wilson fits are run at $N = {n_desc}$ only --- the radius search is $O(n^3)$",
            "  per candidate radius --- so $rho$ and $kappa$ are carried over, both being",
            "  dimensionless by construction, while the columns comparing the Wilson point",
            "  against the front are omitted: that comparison would set a reconstruction of one",
            "  sample against a front built from another.",
        ]
    caption.append("  ],")
    lines += caption

    lines.append(") <tab:geometry-match>")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="exp1_geometry_match.jsonl")
    ap.add_argument("--output", default="exp1_geometry_match.typ")
    ap.add_argument("--n", type=int, default=1000, help="sample size to tabulate")
    ap.add_argument("--region", default="all", help="preference region to report")
    a = ap.parse_args()

    with open(a.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"{a.input} is empty")

    regions = sorted(rows[0]["r2"])
    if a.region not in regions:
        raise SystemExit(f"unknown region {a.region!r}; have {', '.join(regions)}")

    with open(a.output, "w") as f:
        f.write(build(rows, a.n, a.region))
    print(f"wrote {a.output} (n={a.n}, region={a.region})")


if __name__ == "__main__":
    main()
