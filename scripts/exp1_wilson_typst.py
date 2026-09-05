"""
The Experiment 1 Wilson-fit table: what the curvature detector infers about each
synthetic dataset, per constant-curvature arm.

Reads the JSONL emitted by `crates/analysis/src/bin/exp1.rs` and writes a Typst
`#figure(table(...))` fragment labelled `<tab:wilson-fit>`.

Usage:
    uv run python scripts/exp1_wilson_typst.py --n 1000

The input and output default to the paths the pipeline already uses, so the
flags are only needed to override them:

    uv run python scripts/exp1_wilson_typst.py --n 5000 \
        --input results/exp1_geometry_match.jsonl \
        --output tables/exp1_wilson_fit_n5000.typ

Columns: rho, kappa. The companion table of R2 / Delta-R2 / kappa-tilde /
n_front is `exp1_r2_typst.py`; the two share row order and kappa gauge, and
their captions cross-reference each other. Shared machinery lives in
`exp1_common.py`.

The fragment is self-contained: numbers are pre-rendered as Typst math, so it
needs no package imports beyond what `docs/thesis/main.typ` already provides.

Stdlib only.
"""

import argparse

from exp1_common import cell, ensure_parent, load_rows, prepare, render, sci

GENERATOR = "exp1_wilson_typst.py"
LABEL = "tab:wilson-fit"
DEFAULT_OUTPUT = "tables/exp1_wilson_fit.typ"


def build(rows, n):
    """Carries no preference region -- `rho` and `kappa` are properties of the
    fit and the reconstruction, so unlike the R2 table nothing here depends on
    which weights the front is scored under -- nor on the loss-weight `setting`,
    which is why the caption does not name one."""
    rows, by_dataset, order, _ = prepare(rows, n)

    # Both columns survive `--wilson-fallback`, so nothing here is conditional on
    # the sample size the fit was run at; `wilson_n` only feeds the caption.
    wilson_n = sorted({r["wilson"]["wilson_n"] for r in rows if r["wilson"]})
    any_pinned = any(r["wilson"] and r["wilson"]["pinned"] for r in rows)
    n_desc = ", ".join(str(v) for v in wilson_n)

    def data_cells(r):
        w = r["wilson"]
        pinned = bool(w and w["pinned"])
        return [
            cell(sci(w["rho"], 3) if w else "---", pinned=pinned),
            cell(sci(w["kappa"], 3) if w else "---", pinned=pinned),
        ]

    caption = [
        "  caption: [The constant-curvature fit of each synthetic dataset of known intrinsic",
        f"  geometry ($N = {n}$), scored as an embedding.",
        "  Each dataset is fitted under all three geometries; the row whose geometry matches",
        "  the data's intrinsic curvature sign is set in bold.",
        "",
        "  Each arm of @tab:geometry-inference fits a constant-curvature Gram matrix $Z(r)$ to",
        "  the data; because that $Z$ *is* the Gram matrix of the model it tests for, its",
        "  retained eigen-block is an embedding.  Reconstructing it and scoring it with the",
        "  same metric code every t-SNE trial goes through puts the fit at a single point of the",
        "  ten-objective space, so what the detector infers is stated in the same units as what",
        "  the sweep achieved in @tab:geometry-match-r2.",
        "  $rho$ is that arm's normalised signature residual, gauged by $n dot d_"
        + '"max"'
        + "^2$;",
        "  it is also exactly the eigenvalue mass the reconstruction discarded, so it is the",
        "  reconstruction's own error and not a separate misfit score.",
        "",
        "  $kappa = |K| dot R_"
        + '"rms"'
        + "^2$ of @eq:kappa, measured on the configuration the",
        "  fit implies.  It shares that gauge with $tilde(kappa)$ of @tab:geometry-match-r2, so",
        "  the two may be set side by side --- what the detector infers about the data against",
        "  what the optimiser actually chose.  It is $0$ for the Euclidean arm, where $K = 0$",
        "  exactly.",
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
    if len(wilson_n) == 1 and wilson_n[0] != n:
        caption += [
            "",
            f"  The Wilson fits are run at $N = {n_desc}$ --- the radius search is $O(n^3)$ per",
            "  candidate radius --- and reused here: $rho$ and $kappa$ are dimensionless by",
            "  construction, so they characterise the generator rather than the sample.",
        ]

    return render(
        by_dataset,
        order,
        ["[$rho$]", "[$kappa$]"],
        ["right", "right"],
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
    a = ap.parse_args()

    rows = load_rows(a.input)

    ensure_parent(a.output)
    with open(a.output, "w") as f:
        f.write(build(rows, a.n))
    print(f"wrote {a.output} (n={a.n})")


if __name__ == "__main__":
    main()
