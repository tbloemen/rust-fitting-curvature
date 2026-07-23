#!/usr/bin/env python3
"""Thesis results figures (Experiments 2–5) from the qParEGO sweeps.

Produces the figures the results chapter marks with ``// TODO: figure:`` in
``docs/thesis/sections/5results.typ``:

* **Exp 2** (``ablation-results``) — stacked Pareto fronts, one panel per
  (dataset, geometry), one curve per loss-weight setting.
* **Exp 3** (``curvature-magnitude-results``) — median Pareto-front dimensionless
  curvature κ = |K|·R_rms² against the data-intrinsic κ_data (from
  ``results/kappa_data.jsonl``), synthetic vs real markers; plus the
  unanchored-vs-``rms_anchored`` κ overlay (skipped with a notice if no
  rms_anchored runs exist).
* **Exp 4** (``manifold-projection-gap``) — ρ_man-proj(κ): per cell, the Spearman
  correlation between each metric's manifold and 2D-projected variants over the
  cell's trials, plotted against the cell's median κ, one panel per metric,
  hyperbolic vs spherical, both N overlaid.
* **Exp 5** (``comparison-real-results``) — the 4×3 trustworthiness-vs-stress
  Pareto-front grid with convex envelope, and the hyperparameter marginal
  histograms of the ``all_off`` fronts.

κ uses **R_rms** (``r_rms``), not R_max — the thesis definition (the earlier
``analyze_hyperparams.py`` used r_max, which this supersedes).

This is the figure-producing stage: unlike the numpy-only ``pareto_utils`` /
``hv_stats`` it uses matplotlib + scipy and runs locally only.

Usage::

    uv run python analyze_experiments.py            # all figures, N=1000 and 5000
    uv run python analyze_experiments.py --n 1000   # just N=1000
    uv run python analyze_experiments.py --exp 4    # just experiment 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import pareto_utils as pu

# ─── Palette (Okabe-Ito, colourblind-safe, fixed order — never cycled) ─────────

OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

SETTING_COLOR = {
    "all_off": OKABE["black"],
    "centering_only": OKABE["orange"],
    "global_only": OKABE["skyblue"],
    "norm_only": OKABE["green"],
    "all_free": OKABE["vermillion"],
}
SETTING_ORDER = ["all_off", "centering_only", "global_only", "norm_only", "all_free"]

GEOMETRY_COLOR = {
    "euclidean": OKABE["grey"],
    "hyperbolic": OKABE["blue"],
    "spherical": OKABE["vermillion"],
}

REAL_DATASETS = ["mnist", "fashion_mnist", "pbmc", "wordnet_mammals"]
SYNTH_DATASETS = ["sphere", "antipodal_clusters", "tree", "hyperbolic_shells"]
ALL_DATASETS = REAL_DATASETS + SYNTH_DATASETS
GEOMETRIES = ["euclidean", "hyperbolic", "spherical"]
CURVED = ["hyperbolic", "spherical"]

# The five paired metrics (2D vs manifold) used in Exp 4.
PAIRED_METRICS = [
    "trustworthiness",
    "continuity",
    "normalized_stress",
    "shepard_goodness",
    "neighborhood_hit",
]

PLOTS_DIR = Path("plots")


# ─── Loading & shared helpers ─────────────────────────────────────────────────


def load_all_cells(results_dir: Path) -> dict[tuple, list[dict]]:
    """Map every (setting, dataset, n, geometry) to its list of trial records."""
    cells: dict[tuple, list[dict]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        cell = pu.parse_cell_stem(path.stem)
        if cell is None:
            continue
        cells[cell.key()] = pu.trial_records(path)
    return cells


def load_kappa_data(n: int) -> dict[str, dict]:
    """κ_data records keyed by dataset for sample size *n*.

    Prefers ``results/kappa_data_n{n}.jsonl`` and falls back to the unsuffixed
    ``results/kappa_data.jsonl`` (which the local n=1000 run writes).
    """
    for name in (f"results/kappa_data_n{n}.jsonl", "results/kappa_data.jsonl"):
        p = Path(name)
        if p.exists():
            rows = [json.loads(line) for line in open(p) if line.strip()]
            # Only trust the unsuffixed file for the N it was actually run at.
            rows = [r for r in rows if r.get("n_samples") == n]
            if rows:
                return {r["dataset"]: r for r in rows}
    return {}


def kappa(record: dict) -> float | None:
    """Dimensionless embedding curvature κ = |K|·R_rms² for one trial.

    ``curvature_magnitude`` is |K|; ``r_rms`` is the RMS geodesic radius. Returns
    None when either is missing or non-finite (diverged / Euclidean trials).
    """
    k = record.get("curvature_magnitude")
    r = record.get("r_rms")
    if k is None or r is None:
        return None
    k = float(k)
    r = float(r)
    if not (np.isfinite(k) and np.isfinite(r)):
        return None
    return k * r * r


def median_front_kappa(records: list[dict]) -> float | None:
    """Median κ over the 10-objective Pareto front of *records*."""
    front = pu.pareto_front_records(records)
    ks = [k for r in front if (k := kappa(r)) is not None]
    return float(np.median(ks)) if ks else None


def _slice_front(x: np.ndarray, y: np.ndarray, x_up: bool, y_up: bool) -> np.ndarray:
    """Indices of the 2D Pareto front, sorted by x ascending.

    *x_up*/*y_up* say whether larger is better on each axis.
    """
    n = len(x)
    xs = x if x_up else -x
    ys = y if y_up else -y
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (xs >= xs[i]) & (ys >= ys[i]) & ((xs > xs[i]) | (ys > ys[i]))
        keep[dominated] = False
        keep[i] = True
    idx = np.where(keep)[0]
    return idx[np.argsort(x[idx])]


def _finite_xy(records: list[dict], xm: str, ym: str) -> tuple[np.ndarray, np.ndarray]:
    """Finite (x, y) arrays of two raw metric columns over *records*."""
    xs, ys = [], []
    for r in records:
        xv, yv = r.get(xm), r.get(ym)
        if xv is None or yv is None:
            continue
        xv, yv = float(xv), float(yv)
        if np.isfinite(xv) and np.isfinite(yv):
            xs.append(xv)
            ys.append(yv)
    return np.asarray(xs), np.asarray(ys)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(PLOTS_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote plots/{name}.svg (+.png)")


# ─── Experiment 2: stacked Pareto fronts ──────────────────────────────────────

# Trustworthiness (local, ↑) vs normalised stress (global, ↓): the local/global
# cross-section the thesis uses for the front cross-sections.
X_METRIC, Y_METRIC = "trustworthiness", "normalized_stress"
X_LABEL, Y_LABEL = "trustworthiness →", "← normalised stress"


def exp2_stacked_fronts(cells: dict, n: int) -> None:
    datasets, geoms = REAL_DATASETS, GEOMETRIES
    fig, axes = plt.subplots(
        len(datasets), len(geoms), figsize=(11, 13), squeeze=False, sharex=False, sharey=False
    )
    any_data = False
    for i, ds in enumerate(datasets):
        for j, geom in enumerate(geoms):
            ax = axes[i][j]
            _style_axes(ax)
            for setting in SETTING_ORDER:
                recs = cells.get((setting, ds, n, geom))
                if not recs:
                    continue
                x, y = _finite_xy(recs, X_METRIC, Y_METRIC)
                if len(x) < 3:
                    continue
                idx = _slice_front(x, y, x_up=True, y_up=False)
                any_data = True
                ax.plot(
                    x[idx], y[idx], "-o", color=SETTING_COLOR[setting], markersize=3,
                    linewidth=1.3, label=setting, alpha=0.9,
                )
            if i == 0:
                ax.set_title(geom, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{ds}\n{Y_LABEL}", fontsize=9)
            if i == len(datasets) - 1:
                ax.set_xlabel(X_LABEL, fontsize=9)
    if not any_data:
        plt.close(fig)
        print(f"  exp2 N={n}: no data, skipped")
        return
    handles = [
        plt.Line2D([], [], color=SETTING_COLOR[s], marker="o", markersize=4, label=s)
        for s in SETTING_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"Experiment 2 — Pareto fronts by loss setting (N={n})\n"
        "trustworthiness vs normalised stress cross-section",
        fontsize=12, y=1.035,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, f"exp2_stacked_fronts_N{n}")


# ─── Experiment 3: median-front κ vs κ_data ───────────────────────────────────


def exp3_kappa_scatter(cells: dict, n: int) -> None:
    kd = load_kappa_data(n)
    if not kd:
        print(f"  exp3 N={n}: no kappa_data file, skipped scatter")
        return
    # One panel per curved geometry keeps the dataset labels legible.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), squeeze=False)
    hyp_key = {"hyperbolic": "hyp_kappa", "spherical": "sph_kappa"}
    for col, geom in enumerate(CURVED):
        ax = axes[0][col]
        _style_axes(ax)
        pairs: list[tuple[float, float]] = []
        for ds in ALL_DATASETS:
            recs = cells.get(("all_off", ds, n, geom))
            if not recs or ds not in kd:
                continue
            mk = median_front_kappa(recs)
            kdata = kd[ds].get(hyp_key[geom])
            if mk is None or kdata is None or not np.isfinite(kdata) or kdata <= 0:
                continue
            pairs.append((float(kdata), mk))
            marker = "o" if ds in REAL_DATASETS else "s"
            ax.scatter(kdata, mk, marker=marker, s=110, facecolor=GEOMETRY_COLOR[geom],
                       edgecolor="black", linewidth=0.7, alpha=0.85, zorder=3)
            ax.annotate(ds, (kdata, mk), fontsize=8, xytext=(5, 5),
                        textcoords="offset points", color="#333333")
        # Spearman ρ across datasets for this geometry class.
        title = f"{geom} embeddings"
        if len(pairs) >= 3:
            xs, ys = zip(*pairs)
            rho, p = spearmanr(xs, ys)
            title += f"   (Spearman ρ={rho:+.2f}, p={p:.2f}, n={len(pairs)})"
        elif pairs:
            title += f"   (n={len(pairs)}, too few for ρ)"
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"data-intrinsic $\kappa_{\mathrm{data}} = |K|\,d_{\mathrm{rms}}^2$", fontsize=11)
        if col == 0:
            ax.set_ylabel(r"median Pareto-front $\kappa = |K|\,R_{\mathrm{rms}}^2$", fontsize=11)
    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#888", markeredgecolor="k",
                   markersize=10, label="real dataset"),
        plt.Line2D([], [], marker="s", color="w", markerfacecolor="#888", markeredgecolor="k",
                   markersize=10, label="synthetic dataset"),
    ]
    fig.legend(handles=handles, fontsize=9, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Experiment 3 — Pareto-favoured vs intrinsic curvature (N={n})", fontsize=12, y=1.06)
    fig.tight_layout()
    _save(fig, f"exp3_kappa_vs_kappadata_N{n}")


def exp3_rms_anchored(cells: dict, n: int) -> None:
    have_anchored = any(k[0] == "rms_anchored" for k in cells)
    if not have_anchored:
        print(f"  exp3 N={n}: no rms_anchored runs present — overlay figure skipped "
              "(flagged in the discrepancy report)")
        return
    # Present only if the data exists; overlay unanchored (all settings pooled)
    # vs rms_anchored κ distributions per hyperbolic dataset.
    datasets = [ds for ds in ALL_DATASETS
                if any(k[1] == ds and k[3] == "hyperbolic" for k in cells)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 4),
                             squeeze=False, sharey=True)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        _style_axes(ax)
        unanchored, anchored = [], []
        for (setting, d, nn, geom), recs in cells.items():
            if d != ds or nn != n or geom != "hyperbolic":
                continue
            ks = [k for r in pu.pareto_front_records(recs) if (k := kappa(r)) is not None]
            (anchored if setting == "rms_anchored" else unanchored).extend(ks)
        if unanchored:
            ax.hist(unanchored, bins=20, alpha=0.5, color=OKABE["blue"], label="unanchored", density=True)
        if anchored:
            ax.hist(anchored, bins=20, alpha=0.5, color=OKABE["orange"], label="rms_anchored", density=True)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel(r"$\kappa$", fontsize=10)
        if j == 0:
            ax.set_ylabel("density", fontsize=10)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Experiment 3 — unanchored vs rms_anchored κ (hyperbolic, N={n})", fontsize=12)
    fig.tight_layout()
    _save(fig, f"exp3_rms_anchored_kappa_N{n}")


# ─── Experiment 4: ρ_man-proj(κ) ──────────────────────────────────────────────


def _binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = 6):
    """Median of y in log-spaced x bins. Returns (bin_centres, medians) for bins
    with ≥2 points — the trend curve through a cloud of independent cells."""
    if len(x) < 2:
        return np.array([]), np.array([])
    lo, hi = np.log10(x.min()), np.log10(x.max())
    if hi <= lo:
        return np.array([x.mean()]), np.array([np.median(y)])
    edges = np.logspace(lo, hi, n_bins + 1)
    centres, meds = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() >= 2:
            centres.append(np.sqrt(a * b))
            meds.append(np.median(y[m]))
    return np.asarray(centres), np.asarray(meds)


def exp4_rho_man_proj(cells: dict, ns: list[int]) -> None:
    # One point per (dataset, geometry, setting, N) cell: x = median κ over the
    # cell's trials, y = Spearman(manifold variant, 2D variant) over the trials.
    # Cells are independent, so we scatter them and overlay a binned-median trend
    # (never connect raw cells — that would imply a within-series ordering).
    fig, axes = plt.subplots(1, len(PAIRED_METRICS), figsize=(4 * len(PAIRED_METRICS), 4.4),
                             squeeze=False, sharey=True)
    any_data = False
    for m_idx, metric in enumerate(PAIRED_METRICS):
        ax = axes[0][m_idx]
        _style_axes(ax)
        manifold = f"{metric}_manifold"
        for geom in CURVED:
            for n in ns:
                pts = []
                for (_setting, _ds, nn, g), recs in cells.items():
                    if g != geom or nn != n:
                        continue
                    proj, man, ks = [], [], []
                    for r in recs:
                        pv, mv, kv = r.get(metric), r.get(manifold), kappa(r)
                        if pv is None or mv is None or kv is None:
                            continue
                        pv, mv = float(pv), float(mv)
                        if np.isfinite(pv) and np.isfinite(mv):
                            proj.append(pv)
                            man.append(mv)
                            ks.append(kv)
                    if len(proj) < 10:
                        continue
                    rho, _ = spearmanr(man, proj)
                    if np.isfinite(rho):
                        pts.append((float(np.median(ks)), rho))
                if not pts:
                    continue
                any_data = True
                xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                mk = "o" if n == ns[0] else "^"
                ls = "-" if n == ns[0] else "--"
                # Faint cell cloud ...
                ax.scatter(xs, ys, s=16, color=GEOMETRY_COLOR[geom], alpha=0.25, marker=mk, zorder=2)
                # ... plus the binned-median trend.
                bx, by = _binned_median(xs, ys)
                if len(bx):
                    ax.plot(bx, by, ls, marker=mk, color=GEOMETRY_COLOR[geom], markersize=6,
                            linewidth=1.8, alpha=0.95, zorder=3, label=f"{geom} N={n}")
        ax.set_title(metric, fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel(r"median $\kappa$ (cell)", fontsize=10)
        ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle=":")
        ax.set_ylim(-0.8, 1.1)
        if m_idx == 0:
            ax.set_ylabel(r"$\rho_{\mathrm{man\text{-}proj}}$", fontsize=11)
    if not any_data:
        plt.close(fig)
        print("  exp4: no data, skipped")
        return
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=len(seen), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Experiment 4 — manifold vs projected rank agreement vs curvature "
                 "(points = cells, lines = binned median)", fontsize=12, y=1.09)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _save(fig, "exp4_rho_man_proj")


# ─── Experiment 5: front grid + hyperparameter marginals ──────────────────────


def _convex_upper_left(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the upper-left convex hull chain (max x-good, min y) sorted by x.

    We want the envelope of the (trustworthiness ↑, stress ↓) trade-off: the
    lower boundary in stress as trustworthiness grows. Compute the lower convex
    hull of points sorted by x.
    """
    order = np.argsort(x)
    hull: list[int] = []
    for i in order:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # cross product of (b-a)×(i-b); keep lower hull (turn right → pop)
            cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.array(hull)


def exp5_front_grid(cells: dict, n: int) -> None:
    datasets, geoms = REAL_DATASETS, GEOMETRIES
    fig, axes = plt.subplots(len(datasets), len(geoms), figsize=(11, 13), squeeze=False)
    any_data = False
    for i, ds in enumerate(datasets):
        for j, geom in enumerate(geoms):
            ax = axes[i][j]
            _style_axes(ax)
            recs = cells.get(("all_off", ds, n, geom))
            if recs:
                front = pu.pareto_front_records(recs)
                x, y = _finite_xy(front, X_METRIC, Y_METRIC)
                if len(x) >= 1:
                    any_data = True
                    ax.scatter(x, y, s=14, color=GEOMETRY_COLOR[geom], alpha=0.6, zorder=2)
                    if len(x) >= 3:
                        hull = _convex_upper_left(x, y)
                        ax.plot(x[hull], y[hull], "-", color="black", linewidth=1.4, zorder=3)
            if i == 0:
                ax.set_title(geom, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{ds}\n{Y_LABEL}", fontsize=9)
            if i == len(datasets) - 1:
                ax.set_xlabel(X_LABEL, fontsize=9)
    if not any_data:
        plt.close(fig)
        print(f"  exp5 grid N={n}: no data, skipped")
        return
    fig.suptitle(
        f"Experiment 5 — all_off Pareto fronts, trustworthiness vs stress (N={n})\n"
        "black line = convex envelope",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, f"exp5_front_grid_N{n}")


# Hyperparameters that vary under all_off (auxiliary loss weights are fixed at 0).
HYPERPARAMS = [
    ("learning_rate", True),
    ("perplexity_ratio", False),
    ("early_exaggeration_factor", False),
    ("curvature_magnitude", True),
]


def exp5_marginals(cells: dict, n: int) -> None:
    fig, axes = plt.subplots(len(REAL_DATASETS), len(HYPERPARAMS),
                             figsize=(4 * len(HYPERPARAMS), 3 * len(REAL_DATASETS)),
                             squeeze=False)
    any_data = False
    for i, ds in enumerate(REAL_DATASETS):
        for j, (param, logx) in enumerate(HYPERPARAMS):
            ax = axes[i][j]
            _style_axes(ax)
            for geom in GEOMETRIES:
                recs = cells.get(("all_off", ds, n, geom))
                if not recs:
                    continue
                front = pu.pareto_front_records(recs)
                vals = [float(r[param]) for r in front
                        if r.get(param) is not None and np.isfinite(float(r[param]))
                        and float(r[param]) > 0]
                # curvature_magnitude is meaningless for euclidean (fixed): skip
                if param == "curvature_magnitude" and geom == "euclidean":
                    continue
                if len(vals) < 2:
                    continue
                any_data = True
                bins = (np.logspace(np.log10(min(vals)), np.log10(max(vals)), 18)
                        if logx else 18)
                ax.hist(vals, bins=bins, alpha=0.45, color=GEOMETRY_COLOR[geom],
                        label=geom, density=True)
            if logx:
                ax.set_xscale("log")
            if i == 0:
                ax.set_title(param, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{ds}\ndensity", fontsize=9)
    if not any_data:
        plt.close(fig)
        print(f"  exp5 marginals N={n}: no data, skipped")
        return
    handles = [plt.Line2D([], [], color=GEOMETRY_COLOR[g], linewidth=6, alpha=0.5, label=g)
               for g in GEOMETRIES]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Experiment 5 — all_off Pareto-front hyperparameter marginals (N={n})",
                 fontsize=12, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    _save(fig, f"exp5_marginals_N{n}")


# ─── Driver ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--n", type=int, nargs="*", default=[1000, 5000],
                    help="Sample sizes to plot (default 1000 5000).")
    ap.add_argument("--exp", type=int, nargs="*", default=[2, 3, 4, 5],
                    help="Which experiments to render (default all).")
    args = ap.parse_args(argv)

    print("Loading result cells...")
    cells = load_all_cells(args.results_dir)
    print(f"  {len(cells)} cells loaded")

    ns = args.n
    if 2 in args.exp:
        print("Experiment 2 — stacked Pareto fronts")
        for n in ns:
            exp2_stacked_fronts(cells, n)
    if 3 in args.exp:
        print("Experiment 3 — κ vs κ_data")
        for n in ns:
            exp3_kappa_scatter(cells, n)
            exp3_rms_anchored(cells, n)
    if 4 in args.exp:
        print("Experiment 4 — ρ_man-proj(κ)")
        exp4_rho_man_proj(cells, ns)
    if 5 in args.exp:
        print("Experiment 5 — front grid + marginals")
        for n in ns:
            exp5_front_grid(cells, n)
            exp5_marginals(cells, n)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
