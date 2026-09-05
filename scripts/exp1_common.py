"""
Shared machinery for the two Experiment 1 Typst tables.

`exp1_wilson_typst.py` and `exp1_r2_typst.py` render two halves of one JSONL
(`results/exp1_geometry_match.jsonl`, from `crates/analysis/src/bin/exp1.rs`):
what the curvature detector inferred, and what the sweep achieved. They are
separate figures placed separately in the thesis, but a reader sets a row of one
beside the same row of the other -- the two kappa columns share the gauge of
@eq:kappa -- so the row set, the dataset order, the arm order and the bolding of
the matched arm must be identical in both. That is what lives here: computed
once, in [`prepare`], and laid out once, in [`render`].

What deliberately does *not* live here is anything a table decides for itself:
its columns, its caption, its label. Those are the arguments to `render`.

The sibling generators (`r2_delta_typst.py`, `three_arm_typst.py`) duplicate
small pieces of this -- `ensure_parent`, a `cell`, a `sci` -- but their label
maps cover different dataset sets, so this module is scoped to Experiment 1
rather than being a general Typst toolkit.

Stdlib only.
"""

import json
import math
import os

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


def ensure_parent(path):
    """Create the output's directory, so the default `tables/` works on a fresh
    clone (the directory holds only generated fragments, so it may not exist)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_rows(path):
    """The exp1 JSONL, one dict per (dataset, N, geometry) row."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


# ─── Number formatting ────────────────────────────────────────────────────────


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
    """A signed fixed-point number, for the gain column.

    Deliberately not `sci`: that column exists to be compared *down*, and it
    spans barely two decades (roughly 4e-4 to 4e-2), so a common decimal scale
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
    nonsense. The matched row is already marked by its bold geometry label,
    which is enough.
    """
    if pinned:
        text += PINNED_MARK
    return f"[{'*' + text + '*' if bold else text}]"


# ─── Table assembly ───────────────────────────────────────────────────────────


def prepare(rows, n):
    """The row set both tables are built from: filtered to one `N`, indexed by
    dataset, in report order.

    Returns `(rows, by_dataset, order, setting)`. Both tables call this so their
    rows line up one-for-one; see the module docstring."""
    rows = [r for r in rows if r["n"] == n]
    if not rows:
        raise SystemExit(f"no rows with n = {n}")

    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], {})[r["geometry"]] = r

    order = [d for d in ROW_ORDER if d in by_dataset]
    order += [d for d in by_dataset if d not in ROW_ORDER]

    setting = sorted({r["setting"] for r in rows})[0]
    return rows, by_dataset, order, setting


def render(by_dataset, order, header, align, data_cells, caption, label, generator):
    """Assemble one `#figure(table(...))`.

    `header` and `align` cover the data columns only -- the two identity columns
    (dataset, geometry) are the same in both tables and are prepended here.
    `data_cells(row)` returns that row's data cells, which is the only thing
    that differs between the two tables. `generator` is the script to name in
    the do-not-edit banner, so a fragment says which of the two wrote it."""
    n_cols = 2 + len(header)
    align = ["left", "left"] + align

    lines = [
        f"// Generated by scripts/{generator} -- do not edit by hand.",
        "#figure(",
        "  table(",
        f"    columns: {n_cols},",
        f"    align: ({', '.join(align)}),",
        "    stroke: none,",
        # Same inset as the hand-written table in `5results.typ`: the default 5pt
        # spends enough of the text width on padding to wrap the signed columns
        # onto a second line.
        "    inset: (x: 2.5pt, y: 4pt),",
        "    table.hline(stroke: 0.8pt),",
        "    table.header(",
        "      " + ", ".join(["[*Dataset*]", "[*Geom.*]"] + header) + ",",
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

            cells = []
            if gi == 0:
                name = DATASET_LABEL.get(dataset, dataset)
                truth = TRUTH_LABEL.get(r["truth"], r["truth"])
                cells.append(
                    f"table.cell(rowspan: {len(present)})"
                    f"[{name}\\ #text(size: 0.8em)[({truth})]]"
                )

            # The matched arm is marked by bolding its geometry label, which
            # costs no width; a marker column would spend a whole column on one
            # bit.
            cells.append(
                cell(GEOMETRY_LABEL.get(geometry, geometry), bold=r["matched"])
            )
            cells += data_cells(r)
            lines.append("    " + ", ".join(cells) + ",")

    lines.append("    table.hline(stroke: 0.8pt),")
    lines.append("  ),")
    lines += caption
    lines.append("  ],")
    lines.append(f") <{label}>")
    return "\n".join(lines) + "\n"
