"""
The Experiment 1 Pareto-front table: what each embedding geometry achieved,
against the Euclidean baseline.

Reads the JSONL emitted by `crates/analysis/src/bin/exp1.rs` and writes a Typst
`#figure(table(...))` fragment labelled `<tab:geometry-match-r2>`.

Usage:
    uv run python scripts/exp1_r2_typst.py --n 1000 --region all

The input and output default to the paths the pipeline already uses, so the
flags are only needed to override them:

    uv run python scripts/exp1_r2_typst.py --n 5000 --region all \
        --input results/exp1_geometry_match.jsonl \
        --output tables/exp1_geometry_match_n5000.typ

Columns: R2, Delta-R2, kappa-tilde, n_front. The companion table of rho / kappa
is `exp1_wilson_typst.py`; the two share row order and kappa gauge, and their
captions cross-reference each other. Shared machinery lives in
`exp1_common.py`.

Long format, one row per (dataset, geometry): fifteen rows grouped by dataset
rather than five wide ones. The wide form hides the answer, which is a single
column here -- whether the matched geometry's Delta-R2 beats the mismatched
one's.

The fragment is self-contained: numbers are pre-rendered as Typst math, so it
needs no package imports beyond what `docs/thesis/main.typ` already provides.

Stdlib only.
"""

import argparse

from exp1_common import cell, ensure_parent, load_rows, prepare, render, sci, signed

GENERATOR = "exp1_r2_typst.py"
LABEL = "tab:geometry-match-r2"
DEFAULT_OUTPUT = "tables/exp1_geometry_match.typ"


def build(rows, n, region):
    rows, by_dataset, order, setting = prepare(rows, n)

    def data_cells(r):
        delta = r["delta_r2"][region] if r["delta_r2"] else None
        return [
            cell(sci(r["r2"][region], 3)),
            cell(signed(delta)),
            cell(sci(r["kappa_median"], 3)),
            cell(f"${r['n_front']}$"),
        ]

    caption = [
        "  caption: [Experiment 1: matched embedding geometry against the Euclidean baseline,",
        f"  on the synthetic datasets of known intrinsic geometry ($N = {n}$, `{setting}`,",
        f'  preference region $W_"{region}"$).',
        "  Each dataset is embedded under all three geometries; the row whose geometry matches",
        "  the data's intrinsic curvature sign is set in bold.",
        "",
        "  $R_2$ is the R2 indicator of the cell's Pareto front (@eq:r2), a *cost*: smaller is",
        '  better.  $Delta R_2 = R_2("euc") - R_2("geom")$ is therefore formed baseline-minus-row,',
        "  so a positive value means the row's geometry improved on the Euclidean baseline ---",
        "  the same reading direction as @eq:r2-gain.",
        "  $n_"
        + '"front"'
        + "$ is the front's size, reported because $R_2$ improves monotonically",
        "  with it: fronts here range over an order of magnitude, and the Euclidean fronts are",
        "  consistently among the smallest, which biases $Delta R_2$ *toward* the curved arms.",
        "",
        "  $tilde(kappa)$ is the median $|K| dot R_"
        + '"rms"'
        + "^2$ of @eq:kappa over the front --- what",
        "  the optimiser actually chose.  It shares its gauge with the $kappa$ the detector infers",
        "  in @tab:wilson-fit, so the two columns may be set side by side.  Both are $0$ for the",
        "  Euclidean arm, where $K = 0$ exactly.",
    ]

    return render(
        by_dataset,
        order,
        ["[$R_2$]", "[$Delta R_2$]", "[$tilde(kappa)$]", '[$n_"front"$]'],
        ["right", "right", "right", "right"],
        data_cells,
        caption,
        LABEL,
        GENERATOR,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/exp1_geometry_match.jsonl")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--n", type=int, default=1000, help="sample size to tabulate")
    ap.add_argument("--region", default="all", help="preference region to report")
    a = ap.parse_args()

    rows = load_rows(a.input)

    regions = sorted(rows[0]["r2"])
    if a.region not in regions:
        raise SystemExit(f"unknown region {a.region!r}; have {', '.join(regions)}")

    ensure_parent(a.output)
    with open(a.output, "w") as f:
        f.write(build(rows, a.n, a.region))
    print(f"wrote {a.output} (n={a.n}, region={a.region})")


if __name__ == "__main__":
    main()
