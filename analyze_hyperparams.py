"""
Hyperparameter sensitivity analysis for fitting-curvature optimizer results.

--mode optimize (default)
  1. spearman_heatmap.svg         — Spearman ρ per (parameter × curvature)
  2. param_importance.svg         — Mean |ρ| across curvatures (bar chart)
  3. param_<name>_vs_metric.svg   — Per-parameter scatter vs metric, one series per curvature
  6. metric_correlation.svg       — Metric–metric Spearman rank correlation (do metrics agree?)
  7. importance_heatmap.svg       — |ρ| for every (parameter × metric) pair, pooled across curvatures
  9. marginal_effects.svg         — Binned mean ± 95% CI of metric per parameter (response curves)
  10. good_regions.svg            — p10–p90 intervals of top-k% runs, normalised to [0, 1]

--mode scan
  1. scan_effects.svg             — Clean effect curve per parameter (metric vs swept value)
  2. scan_sensitivity.svg         — Heatmap of metric range (max−min) per (parameter × curvature)
  3. scan_optimal.svg             — Where in the sweep range the optimum sits per (param × curvature)

--mode gp
  Fits the same GP surrogate used by the Bayesian optimizer (Frazier 2018) to the
  observed trial data and visualises the resulting loss landscape.  One GP is fitted
  per curvature using MLE for the RBF length-scale (§3.2), zero-mean prior, and
  Cholesky-based posterior inference (Eq. 3).

  1. gp_slices.svg    — 1D marginal predictions: posterior mean ± 2σ per parameter,
                        with raw observations overlaid.  Other parameters are held
                        at their sample median in GP input space.
  2. gp_landscape.svg — 2D posterior mean heatmap for learning_rate × perplexity_ratio,
                        one panel per curvature.  White dashed contours show posterior σ.
  3. gp_ei.svg        — 1D Expected Improvement curves (Frazier Eq. 7):
                        EI = Δ·Φ(Δ/σ) + σ·φ(Δ/σ), where Δ = µ − f*.
                        Peaks indicate where the optimizer would sample next.

  Requires --metric.

Use --metric to select which metric column to analyze (e.g. davies_bouldin_ratio).
If omitted, the first metric column present in the data is used automatically.

Usage:
    uv run analyze_hyperparams.py --metric davies_bouldin_ratio
    uv run analyze_hyperparams.py --mode scan --input results/results --output plots/
    uv run analyze_hyperparams.py --mode gp --metric knn_overlap --input results/results_mnist
    uv run analyze_hyperparams.py --input results/results_mnist --output plots/ --top-pairs 5
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ─── Configuration ────────────────────────────────────────────────────────────

CURVATURES = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]

# All possible parameter field names (union of old and new format).
# Ordered by conceptual grouping.
ALL_PARAMS = [
    "learning_rate",
    "perplexity_ratio",  # new format
    "perplexity",  # old format (absolute)
    "momentum_main",
    "centering_weight",
    "global_loss_weight",
    "norm_loss_weight",
    "n_iterations",  # old format (now fixed)
    "early_exaggeration_iterations",  # old format (now fixed)
]

LOG_SCALE_PARAMS = {"learning_rate", "perplexity", "perplexity_ratio"}
CATEGORICAL_PARAMS: set[str] = set()
# Metrics where lower is better (will be handled accordingly in plots).
MINIMIZE_METRICS = {"geodesic_distortion_gu2019", "geodesic_distortion_mse"}

ALL_METRICS = [
    "trustworthiness",
    "continuity",
    "knn_overlap",
    "geodesic_distortion_gu2019",
    "geodesic_distortion_mse",
    "davies_bouldin_ratio",
    "dunn_index",
    "class_density_measure",
    "cluster_density_measure",
]


def get_metric_value(record: dict, metric: str | None) -> float | None:
    """Extract the named metric value from a record, or None if absent."""
    if metric and metric in record:
        return record[metric]
    return None


def get_param_value(record: dict, param: str) -> float | None:
    """Extract a parameter value as float."""
    val = record.get(param)
    if val is None:
        return None
    return float(val)


# Curvature color map (tab10 via curvature index)
_CMAP_K = plt.cm.tab10  # type: ignore


def k_color(k: float, all_ks: list[float]) -> tuple:
    idx = sorted(all_ks).index(k)
    return _CMAP_K(idx / max(len(all_ks) - 1, 1))


# ─── Data loading ─────────────────────────────────────────────────────────────


def _load_jsonl(path: str) -> list[dict]:
    """Load all records from a single JSONL file."""
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return records


def load_results(path: str) -> list[dict]:
    """Load non-scan trial results from a single JSONL file."""
    return [r for r in _load_jsonl(path) if not r.get("scan_param")]


def present_params(records: list[dict]) -> list[str]:
    """Return only params that actually appear in the data, preserving order."""
    if not records:
        return []
    keys = set().union(*(r.keys() for r in records))
    return [p for p in ALL_PARAMS if p in keys]


# ─── Spearman helpers ─────────────────────────────────────────────────────────


def compute_correlations(
    records: list[dict],
    params: list[str],
    metric: str | None = None,
) -> dict[str, dict[float, tuple[float, float]]]:
    """
    Returns {param: {curvature: (rho, pval)}}.
    Only included where enough data exists (≥5 trials).
    """
    by_k: dict[float, list[dict]] = {}
    for r in records:
        by_k.setdefault(r["curvature"], []).append(r)

    result: dict[str, dict[float, tuple[float, float]]] = {}
    for param in params:
        result[param] = {}
        for k, group in by_k.items():
            pairs = [
                (get_param_value(r, param), get_metric_value(r, metric))
                for r in group
                if param in r
            ]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 5:
                continue
            xs, ys = zip(*pairs)
            rho, pval = stats.spearmanr(xs, ys)
            result[param][k] = (float(rho), float(pval))  # type: ignore
    return result


def mean_abs_rho(correlations: dict, params: list[str]) -> dict[str, float]:
    return {
        p: float(np.mean([abs(rho) for rho, _ in correlations.get(p, {}).values()]))
        for p in params
        if correlations.get(p)
    }


# ─── Plot 1: Spearman heatmap ─────────────────────────────────────────────────


def plot_spearman_heatmap(correlations: dict, params: list[str], out_path: str) -> None:
    curvatures = sorted({k for p in correlations.values() for k in p})
    n_p, n_k = len(params), len(curvatures)

    matrix = np.full((n_p, n_k), np.nan)
    sig = np.zeros((n_p, n_k), dtype=bool)

    for i, param in enumerate(params):
        for j, k in enumerate(curvatures):
            entry = correlations.get(param, {}).get(k)
            if entry is not None:
                rho, pval = entry
                matrix[i, j] = rho
                sig[i, j] = pval < 0.05

    fig_w = max(6, n_k * 1.0 + 2.5)
    fig_h = max(3, n_p * 0.55 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"k={k:+.1f}" for k in curvatures], rotation=45, ha="right")
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(params)

    for i in range(n_p):
        for j in range(n_k):
            if np.isnan(matrix[i, j]):
                continue
            rho = matrix[i, j]
            star = "*" if sig[i, j] else ""
            color = "white" if abs(rho) > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{rho:+.2f}{star}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )

    plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.03, pad=0.02)
    ax.set_title(
        "Spearman correlation: hyperparameter → metric\n(* = p < 0.05)", pad=10
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Plot 2: Parameter importance bar chart ───────────────────────────────────


def plot_param_importance(correlations: dict, params: list[str], out_path: str) -> None:
    importance = mean_abs_rho(correlations, params)
    if not importance:
        print(f"  {out_path} (skipped — not enough data for correlations)")
        return
    stderr = {
        p: float(
            np.std([abs(rho) for rho, _ in correlations[p].values()])
            / math.sqrt(max(1, len(correlations[p])))
        )
        for p in importance
    }
    # Sort descending
    order = sorted(importance, key=lambda p: importance[p], reverse=True)

    fig, ax = plt.subplots(figsize=(7, max(3, len(order) * 0.45 + 1.5)))

    ys = list(range(len(order)))
    xs = [importance[p] for p in order]
    errs = [stderr[p] for p in order]

    colors = [
        "#d7191c" if x >= 0.4 else "#fdae61" if x >= 0.25 else "#abd9e9" for x in xs
    ]
    ax.barh(ys, xs, xerr=errs, color=colors, alpha=0.85, capsize=3, height=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Spearman ρ| across curvatures  (± 1 SE)")
    ax.set_title("Parameter importance")
    ax.axvline(
        0.25,
        color="#fdae61",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="|ρ| = 0.25",
    )
    ax.axvline(
        0.40,
        color="#d7191c",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="|ρ| = 0.40",
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(1.05, max(xs) * 1.3 + 0.05))

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Plot 3: Per-parameter scatter vs metric ──────────────────────────────────


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(y, np.ones(window) / window, mode="valid")


def plot_param_vs_metric(
    records: list[dict],
    params: list[str],
    out_dir: str,
    metric: str | None = None,
) -> None:
    all_ks = sorted({r["curvature"] for r in records})
    metric_label = metric or "metric"

    for param in params:
        is_log = param in LOG_SCALE_PARAMS

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k in all_ks:
            group = [
                r
                for r in records
                if r["curvature"] == k
                and param in r
                and get_metric_value(r, metric) is not None
            ]
            if not group:
                continue
            col = k_color(k, all_ks)
            xs = np.array([get_param_value(r, param) for r in group])
            ys = np.array([get_metric_value(r, metric) for r in group])
            ax.scatter(xs, ys, color=col, alpha=0.35, s=12, linewidths=0)
            # Trend: sort by x, then moving average
            order = np.argsort(xs)
            xs_s, ys_s = xs[order], ys[order]
            win = max(3, len(xs_s) // 8)
            if len(xs_s) >= win * 2:
                trend_y = _moving_average(ys_s, win)
                trend_x = xs_s[win // 2 : win // 2 + len(trend_y)]
                ax.plot(
                    trend_x,
                    trend_y,
                    color=col,
                    linewidth=1.8,
                    label=f"k={k:+.1f}",
                    alpha=0.9,
                )

        if is_log:
            ax.set_xscale("log")
        ax.set_xlabel(param + (" (log scale)" if is_log else ""))
        ax.legend(fontsize=7, ncol=3, loc="best", markerscale=1.2)

        ax.set_ylabel(metric_label)
        ax.set_title(f"Effect of {param} on {metric_label}")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"param_{param}_vs_metric.svg")
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path}")


# ─── Plot 6: Metric–metric rank correlation ───────────────────────────────────


def plot_metric_correlation(records: list[dict], out_path: str) -> None:
    """
    Spearman rank correlation between every pair of metrics.
    Shown as four panels: all data pooled, hyperbolic (k<0), Euclidean (k=0), spherical (k>0).
    High ρ means optimising one metric ≈ optimising the other.
    """
    present = [m for m in ALL_METRICS if any(r.get(m) is not None for r in records)]
    if len(present) < 2:
        print(f"  {out_path} (skipped — fewer than 2 metrics present)")
        return

    groups = [
        (records, "All curvatures"),
        ([r for r in records if r["curvature"] < 0], "Hyperbolic (k < 0)"),
        ([r for r in records if r["curvature"] == 0.0], "Euclidean (k = 0)"),
        ([r for r in records if r["curvature"] > 0], "Spherical (k > 0)"),
    ]
    n_m = len(present)
    panel = max(3.5, n_m * 0.55 + 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(panel * 4 + 1.0, panel + 0.5))

    for ax, (subset, title) in zip(axes, groups):
        matrix = np.full((n_m, n_m), np.nan)
        for i, m1 in enumerate(present):
            for j, m2 in enumerate(present):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                vals = [
                    (r[m1], r[m2])
                    for r in subset
                    if r.get(m1) is not None and r.get(m2) is not None
                ]
                if len(vals) < 5:
                    continue
                v1, v2 = zip(*vals)
                # Negate minimisation metrics so that positive ρ always means
                # 'both metrics agree in direction'.
                s1 = -1 if m1 in MINIMIZE_METRICS else 1
                s2 = -1 if m2 in MINIMIZE_METRICS else 1
                v1 = tuple(s1 * x for x in v1)
                v2 = tuple(s2 * x for x in v2)
                rho, _ = stats.spearmanr(v1, v2)
                matrix[i, j] = float(rho)  # type: ignore

        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_m))
        ax.set_xticklabels(present, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(n_m))
        ax.set_yticklabels(present, fontsize=6)
        for i in range(n_m):
            for j in range(n_m):
                if not np.isnan(matrix[i, j]):
                    v = matrix[i, j]
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if abs(v) > 0.6 else "black",
                    )
        ax.set_title(title, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Metric–metric Spearman rank correlation\n"
        "high ρ = optimising one metric ≈ optimising the other  "
        "(minimisation metrics negated so sign is consistent)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Plot 7: Variable importance heatmap (param × metric) ─────────────────────


def plot_variable_importance_heatmap(
    records: list[dict], params: list[str], out_path: str
) -> None:
    """
    |Spearman ρ| for every (parameter × metric) pair, pooled across all curvatures.
    Rows that are uniformly low mean that parameter barely affects any metric.
    Columns that are uniformly low mean that metric is insensitive to all parameters.
    """
    present = [m for m in ALL_METRICS if any(r.get(m) is not None for r in records)]
    continuous_params = [p for p in params if p not in CATEGORICAL_PARAMS]
    if not present or not continuous_params:
        print(f"  {out_path} (skipped — no data)")
        return

    n_p, n_m = len(continuous_params), len(present)
    matrix = np.full((n_p, n_m), np.nan)

    for i, param in enumerate(continuous_params):
        for j, metric in enumerate(present):
            pairs = [
                (get_param_value(r, param), r.get(metric))
                for r in records
                if param in r and r.get(metric) is not None
            ]
            pairs = [(x, y) for x, y in pairs if x is not None]
            if len(pairs) < 5:
                continue
            xs, ys = zip(*pairs)
            rho, _ = stats.spearmanr(xs, ys)
            matrix[i, j] = abs(float(rho))  # type: ignore

    vmax = min(0.8, float(np.nanmax(matrix))) if not np.all(np.isnan(matrix)) else 0.8
    fig, ax = plt.subplots(figsize=(n_m * 0.95 + 2.5, n_p * 0.55 + 1.8))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_m))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(continuous_params, fontsize=8)
    for i in range(n_p):
        for j in range(n_m):
            if not np.isnan(matrix[i, j]):
                v = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v > vmax * 0.65 else "black",
                )
    plt.colorbar(im, ax=ax, label="|Spearman ρ|", fraction=0.03, pad=0.02)
    ax.set_title(
        "Variable importance: |Spearman ρ| per (parameter × metric)\n"
        "(pooled across all curvatures)",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Plot 9: Marginal effects (binned response curves) ───────────────────────


def plot_marginal_effects(
    records: list[dict],
    params: list[str],
    out_path: str,
    metric: str | None = None,
    n_bins: int = 10,
) -> None:
    """
    Binned response curves: split each continuous parameter into quantile bins,
    compute mean ± 95% CI of the metric per bin, and plot as line + shaded band
    with one series per curvature.

    Unlike raw scatter + trend, this directly shows the functional effect shape
    and identifies non-monotonic relationships (U-curves, thresholds, plateaus).
    """
    continuous_params = [p for p in params if p not in CATEGORICAL_PARAMS]
    if not continuous_params:
        print(f"  {out_path} (skipped — no continuous params)")
        return
    all_ks = sorted({r["curvature"] for r in records})
    metric_label = metric or "metric"

    n_p = len(continuous_params)
    ncols = min(3, n_p)
    nrows = math.ceil(n_p / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.4))
    axes_flat = np.array(axes).flatten() if n_p > 1 else [axes]

    for ax, param in zip(axes_flat, continuous_params):
        is_log = param in LOG_SCALE_PARAMS
        for k in all_ks:
            group = [
                r
                for r in records
                if r["curvature"] == k
                and param in r
                and get_metric_value(r, metric) is not None
            ]
            if len(group) < n_bins:
                continue
            col = k_color(k, all_ks)
            xs = np.array([get_param_value(r, param) for r in group], dtype=float)
            ys = np.array([get_metric_value(r, metric) for r in group], dtype=float)

            # Quantile-based bin edges, deduplicated
            bin_edges = np.unique(np.quantile(xs, np.linspace(0, 1, n_bins + 1)))
            if len(bin_edges) < 3:
                continue

            bin_centers, means, cis = [], [], []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (xs >= lo) & (xs <= hi)
                if mask.sum() < 2:
                    continue
                vals = ys[mask]
                m = float(np.mean(vals))
                se = float(stats.sem(vals))
                ci = float(stats.t.ppf(0.975, df=max(len(vals) - 1, 1)) * se)
                if is_log:
                    center = float(
                        np.exp((np.log(max(lo, 1e-12)) + np.log(max(hi, 1e-12))) / 2)
                    )
                else:
                    center = float((lo + hi) / 2)
                bin_centers.append(center)
                means.append(m)
                cis.append(ci)

            if not bin_centers:
                continue
            bx = np.array(bin_centers)
            my = np.array(means)
            ci_arr = np.array(cis)
            ax.plot(
                bx,
                my,
                color=col,
                linewidth=1.8,
                label=f"k={k:+.1f}",
                marker="o",
                markersize=3.5,
            )
            ax.fill_between(bx, my - ci_arr, my + ci_arr, color=col, alpha=0.18)

        if is_log:
            ax.set_xscale("log")
        ax.set_title(param, fontsize=9)
        ax.set_xlabel(param + (" (log)" if is_log else ""), fontsize=7)
        ylabel = metric_label + (
            " (lower=better)" if metric in MINIMIZE_METRICS else ""
        )
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n_p:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=7,
            ncol=min(5, len(all_ks)),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.suptitle(
        f"Marginal effects: binned mean ± 95% CI of {metric_label} per parameter\n"
        "(quantile bins; non-monotonic shapes reveal U-curves or threshold effects)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Plot 10: Good regions (p10–p90 of top-k% runs) ──────────────────────────


def _good_region_data(
    records: list[dict],
    params: list[str],
    metric: str | None,
    top_pct: float,
    subset: list[dict],
) -> dict[str, tuple[float, float, float]]:
    """
    Compute p10/p50/p90 of each continuous parameter in the top `top_pct` fraction
    of `subset` (by metric direction), normalised to the full-data [min, max] range.
    Returns {param: (p10_norm, p50_norm, p90_norm)}.
    """
    continuous_params = [p for p in params if p not in CATEGORICAL_PARAMS]
    if not subset:
        return {}
    valid = [r for r in subset if get_metric_value(r, metric) is not None]
    if not valid:
        return {}
    n_top = max(1, int(len(valid) * top_pct))
    ordered = sorted(valid, key=lambda r: get_metric_value(r, metric) or 0.0)
    top_records = ordered[:n_top] if metric in MINIMIZE_METRICS else ordered[-n_top:]

    result: dict[str, tuple[float, float, float]] = {}
    for param in continuous_params:
        all_vals = [
            get_param_value(r, param)
            for r in records
            if param in r and get_param_value(r, param) is not None
        ]
        top_vals = [
            get_param_value(r, param)
            for r in top_records
            if param in r and get_param_value(r, param) is not None
        ]
        if len(all_vals) < 2 or not top_vals:
            continue
        lo, hi = float(min(all_vals)), float(max(all_vals))  # type: ignore
        if hi <= lo:
            continue

        def norm(v: float, _lo: float = lo, _hi: float = hi) -> float:
            return (float(v) - _lo) / (_hi - _lo)

        p10, p50, p90 = np.percentile(top_vals, [10, 50, 90])  # type: ignore
        result[param] = (norm(p10), norm(p50), norm(p90))
    return result


def print_good_regions(
    records: list[dict],
    params: list[str],
    metric: str | None,
    top_pct: float,
) -> None:
    """
    Print a text table of p10/p50/p90 (raw parameter values) for the top `top_pct`%
    runs, broken out by geometry group (All / Hyperbolic / Euclidean / Spherical).
    """
    continuous_params = [p for p in params if p not in CATEGORICAL_PARAMS]
    groups = [
        ("All", records),
        ("Hyperbolic (k<0)", [r for r in records if r["curvature"] < 0]),
        ("Euclidean  (k=0)", [r for r in records if r["curvature"] == 0.0]),
        ("Spherical  (k>0)", [r for r in records if r["curvature"] > 0]),
    ]
    direction = "lower=better" if metric in MINIMIZE_METRICS else "higher=better"
    print(
        f"\nGood regions — top {int(top_pct * 100)}% runs by "
        f"{metric or 'auto metric'}  ({direction})"
    )
    for group_name, subset in groups:
        valid = [r for r in subset if get_metric_value(r, metric) is not None]
        if not valid:
            continue
        n_top = max(1, int(len(valid) * top_pct))
        ordered = sorted(valid, key=lambda r: get_metric_value(r, metric) or 0.0)
        top_records = (
            ordered[:n_top] if metric in MINIMIZE_METRICS else ordered[-n_top:]
        )
        print(f"\n  {group_name}  (n={len(valid)}, top {n_top}):")
        print(f"  {'param':<35} {'p10':>10} {'p50':>10} {'p90':>10}")
        print("  " + "-" * 70)
        for param in continuous_params:
            top_vals = [
                get_param_value(r, param)
                for r in top_records
                if param in r and get_param_value(r, param) is not None
            ]
            if not top_vals:
                continue
            p10, p50, p90 = np.percentile(top_vals, [10, 50, 90])  # type: ignore
            print(f"  {param:<35} {p10:>10.4g} {p50:>10.4g} {p90:>10.4g}")


def plot_good_regions(
    records: list[dict],
    params: list[str],
    out_path: str,
    metric: str | None = None,
    top_pct: float = 0.1,
) -> None:
    """
    Horizontal bars showing the p10–p90 interval of each continuous parameter
    in the top `top_pct`% runs, normalised to [0, 1] using the full data range.
    Four panels: All / Hyperbolic (k<0) / Euclidean (k=0) / Spherical (k>0).

    A narrow bar near 0 → best runs use low values.
    A bar spanning [0.2, 0.8] → many values work but extremes hurt.
    The tick mark (|) shows the median (p50).
    """
    continuous_params = [p for p in params if p not in CATEGORICAL_PARAMS]
    if not continuous_params:
        print(f"  {out_path} (skipped — no continuous params)")
        return

    groups = [
        ("All", records),
        ("Hyperbolic\n(k < 0)", [r for r in records if r["curvature"] < 0]),
        ("Euclidean\n(k = 0)", [r for r in records if r["curvature"] == 0.0]),
        ("Spherical\n(k > 0)", [r for r in records if r["curvature"] > 0]),
    ]

    n_p = len(continuous_params)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, n_p * 0.55 + 2.5), sharey=True)

    for ax, (group_name, subset) in zip(axes, groups):
        regions = _good_region_data(records, params, metric, top_pct, subset)
        for yi, param in enumerate(continuous_params):
            if param not in regions:
                continue
            p10, p50, p90 = regions[param]
            ax.barh(yi, p90 - p10, left=p10, height=0.5, color="#4393c3", alpha=0.7)
            ax.plot(p50, yi, marker="|", color="#053061", markersize=10, linewidth=2)

        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(group_name, fontsize=8)
        ax.set_xlabel("Normalised parameter value\n(full [min, max] range)", fontsize=7)
        ax.tick_params(labelsize=7)

    axes[0].set_yticks(range(n_p))
    axes[0].set_yticklabels(continuous_params, fontsize=7)
    axes[0].invert_yaxis()

    metric_dir = " (lower=better)" if metric in MINIMIZE_METRICS else " (higher=better)"
    fig.suptitle(
        f"Good regions: p10–p90 of top {int(top_pct * 100)}% runs  ·  "
        f"metric: {metric or 'auto'}{metric_dir}\n"
        "bar = p10–p90 interval  |  tick = p50  |  values normalised to full [min, max]",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── GP surrogate (--mode gp) ────────────────────────────────────────────────
#
# Python only does *forward prediction* using the GP state exported by the Rust
# Bayesian optimizer (--mode bayes writes <output>_gp_state.json).
# All fitting logic lives exclusively in gp.rs — MLE length-scale search,
# Cholesky decomposition, and I/O standardisation are done in Rust, then
# serialised to JSON.  Python reads the JSON and reconstructs predictions with
# ~15 lines of numpy, avoiding any duplication of the GP implementation.
#
# Forward prediction (Frazier Eq. 3):
#   k_star[i] = exp(−‖xs_norm[i] − x_norm‖² / (2·l²))
#   µₙ(x)     = k_star ᵀ α           (α = K⁻¹ y_norm stored in state)
#   v          = L⁻¹ k_star           (L = Cholesky factor stored in state)
#   σₙ(x)     = √max(0, 1 − ‖v‖²)
#   mu_orig    = µₙ · y_std + y_mean  (undo standardisation; undo sign-flip if minimize)


def load_gp_states(input_prefix: str) -> "dict[float, dict]":
    """
    Load all GP state JSON files matching <input_prefix>*_gp_state.json.

    The Rust optimizer writes one file per (dataset, curvature) pair; the
    curvature is encoded in the filename as e.g. ``k-0_1`` (dots replaced by
    underscores).  Returns a dict mapping float curvature → state dict.
    """
    states: dict[float, dict] = {}
    prefix_path = Path(input_prefix)
    search_dir = prefix_path.parent
    # GP state files are named <output_stem>_gp_<dataset>_<curvature>.json
    name_pattern = f"{prefix_path.stem}_gp_*.json"
    for path in sorted(search_dir.glob(name_pattern)):
        m = re.search(r"_([+-]\d+_\d+)$", path.stem)
        if not m:
            continue
        k = float(m.group(1).replace("_", "."))
        try:
            with open(path) as f:
                states[k] = json.load(f)
        except Exception as e:
            print(f"  Warning: could not load {path}: {e}")
    return states


def _prepare_gp_state(state: dict) -> dict:
    """
    Recompute L (Cholesky) and alpha (K⁻¹y) from the stored observations.

    These are not persisted in the JSON (they grow as O(n²) / O(n)) but are
    cheap to rebuild: O(n³) Cholesky, negligible for typical n < 1000.
    The result is cached back into the state dict so repeated calls are free.
    """
    if "L" in state:
        return state  # already prepared
    obs = state["observations"]
    xs_norm = np.array([o["x_norm"] for o in obs])
    n = len(xs_norm)
    l = state["length_scale"]

    metrics = np.array([o["metric"] for o in obs])
    if state["direction"] == "minimize":
        metrics = -metrics
    y_norm = (metrics - state["y_mean"]) / state["y_std"]

    diffs = xs_norm[:, None, :] - xs_norm[None, :, :]
    K = np.exp(-np.sum(diffs**2, axis=-1) / (2 * l**2)) + 1e-4 * np.eye(n)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))

    state["L"] = L
    state["alpha"] = alpha
    return state


def _best_x_enc(state: dict) -> "np.ndarray":
    """Return the encoded input vector of the best observed trial."""
    obs = state["observations"]
    if state["direction"] == "maximize":
        best = max(obs, key=lambda o: o["metric"])
    else:
        best = min(obs, key=lambda o: o["metric"])
    return np.array(best["x_encoded"])


def _gp_predict_batch(
    state: dict, X_enc: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Vectorized GP posterior mean and std from an exported Rust GP state.

    Parameters
    ----------
    state : dict
        GP state loaded from a ``*_gp_state.json`` file.
    X_enc : (n_test, d) ndarray
        Log-transformed (but not yet standardised) GP inputs, in the same
        order as ``state["param_names"]``.

    Returns
    -------
    mu, sigma : (n_test,) arrays in original (non-sign-flipped) metric units.
    """
    _prepare_gp_state(state)
    x_means = np.array(state["x_means"])
    x_stds = np.array(state["x_stds"])
    X_norm = (X_enc - x_means) / x_stds

    xs_norm = np.array([obs["x_norm"] for obs in state["observations"]])
    l = state["length_scale"]
    L, alpha = state["L"], state["alpha"]

    # K_star[i, j] = k(X_norm[i], xs_norm[j])
    diffs = X_norm[:, None, :] - xs_norm[None, :, :]
    K_star = np.exp(-np.sum(diffs**2, axis=-1) / (2 * l**2))

    mu_norm = K_star @ alpha  # (n_test,)
    v = np.linalg.solve(L, K_star.T)  # (n_train, n_test)
    sigma_norm = np.sqrt(np.maximum(1.0 - np.sum(v**2, axis=0), 0.0))

    mu = mu_norm * state["y_std"] + state["y_mean"]
    sigma = sigma_norm * state["y_std"]
    if state["direction"] == "minimize":
        mu = -mu  # undo the sign-flip applied during fitting
    return mu, sigma


def _gp_ei_batch(state: dict, X_enc: "np.ndarray") -> "np.ndarray":
    """
    Vectorized Expected Improvement from an exported Rust GP state (Frazier Eq. 7):
      EI = max(Δ·Φ(Δ/σ) + σ·φ(Δ/σ), 0)  where Δ = µₙ(x) − f*ₙ

    Operates in normalised space (matching gp.rs), then scales to metric units.
    """
    _prepare_gp_state(state)
    x_means = np.array(state["x_means"])
    x_stds = np.array(state["x_stds"])
    X_norm = (X_enc - x_means) / x_stds

    xs_norm = np.array([obs["x_norm"] for obs in state["observations"]])
    l = state["length_scale"]
    L, alpha = state["L"], state["alpha"]

    diffs = X_norm[:, None, :] - xs_norm[None, :, :]
    K_star = np.exp(-np.sum(diffs**2, axis=-1) / (2 * l**2))

    mu_norm = K_star @ alpha
    v = np.linalg.solve(L, K_star.T)
    sigma_norm = np.sqrt(np.maximum(1.0 - np.sum(v**2, axis=0), 0.0))

    delta = mu_norm - state["f_best_norm"]
    z = np.where(sigma_norm > 1e-10, delta / sigma_norm, 0.0)
    ei = delta * stats.norm.cdf(z) + sigma_norm * stats.norm.pdf(z)
    return np.maximum(ei, 0.0) * state["y_std"]


# ─── GP plot 1: 1D marginal prediction slices ─────────────────────────────────


def plot_gp_slices(
    states: dict[float, dict],
    metric: str,
    out_path: str,
    n_grid: int = 120,
) -> None:
    """
    For each hyperparameter, draw the GP's 1D marginal prediction curve:
    posterior mean (solid) ± 2σ (shaded band), with raw observations as dots.

    All parameters except the swept one are held at their sample median in
    GP encoded space.  One curve per curvature, loaded from the Rust GP state.
    """
    if not states:
        print(f"  {out_path} (skipped — no GP states found)")
        return

    first = next(iter(states.values()))
    gp_params = first["param_names"]
    log_params = set(first["log_scale_params"])
    minimize = first["direction"] == "minimize"
    all_ks = sorted(states.keys())
    n_p = len(gp_params)
    ncols = min(3, n_p)
    nrows = math.ceil(n_p / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5), squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, sweep_param in zip(axes_flat, gp_params):
        is_log = sweep_param in log_params
        sweep_idx = gp_params.index(sweep_param)

        for k in all_ks:
            state = states.get(k)
            if state is None:
                continue
            col = k_color(k, all_ks)
            obs = state["observations"]

            # Scatter: back-transform encoded value to original scale for the x-axis
            obs_x = [
                (
                    np.exp(o["x_encoded"][sweep_idx])
                    if is_log
                    else o["x_encoded"][sweep_idx]
                )
                for o in obs
            ]
            obs_y = [o["metric"] for o in obs]
            ax.scatter(obs_x, obs_y, color=col, alpha=0.15, s=8, linewidths=0, zorder=1)

            # Build 1D test grid in encoded space across the observed range
            X_enc_obs = np.array([o["x_encoded"] for o in obs])
            enc_lo = X_enc_obs[:, sweep_idx].min()
            enc_hi = X_enc_obs[:, sweep_idx].max()
            enc_grid = np.linspace(enc_lo, enc_hi, n_grid)

            # Hold all other params at the values of the best observed trial
            X_test_enc = np.tile(_best_x_enc(state), (n_grid, 1))
            X_test_enc[:, sweep_idx] = enc_grid

            mu, sigma = _gp_predict_batch(state, X_test_enc)
            x_plot = np.exp(enc_grid) if is_log else enc_grid

            ax.plot(
                x_plot,
                mu,
                color=col,
                linewidth=1.8,
                label=f"k={k:+.1f}",
                zorder=3,
                alpha=0.9,
            )
            ax.fill_between(
                x_plot, mu - 2 * sigma, mu + 2 * sigma, color=col, alpha=0.12, zorder=2
            )

        if is_log:
            ax.set_xscale("log")
        ax.set_title(sweep_param, fontsize=9)
        ax.set_xlabel(sweep_param + (" (log scale)" if is_log else ""), fontsize=7)
        direction_note = " (lower=better)" if minimize else ""
        ax.set_ylabel(metric + direction_note, fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n_p:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=7,
            ncol=min(5, len(all_ks)),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.suptitle(
        f"GP surrogate — 1D marginal predictions  ·  metric: {metric}\n"
        "solid line = posterior mean  |  band = ±2σ  |  dots = observations\n"
        "(all other parameters held at the values of the best observed trial)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── GP plot 2: 2D posterior mean landscape ───────────────────────────────────


def plot_gp_landscape(
    states: dict[float, dict],
    metric: str,
    out_path: str,
    n_grid: int = 50,
) -> None:
    """
    2D GP posterior mean heatmap for learning_rate × perplexity_ratio,
    one subplot per curvature.

    Colour encodes the GP posterior mean (viridis, shared scale across
    curvatures).  White dashed contours show posterior σ (uncertainty).
    White dots are observed data points.  All other parameters are held at
    their sample median in GP encoded space.
    """
    if not states:
        print(f"  {out_path} (skipped — no GP states found)")
        return

    first = next(iter(states.values()))
    gp_params = first["param_names"]
    log_params = set(first["log_scale_params"])
    minimize = first["direction"] == "minimize"
    all_ks = sorted(states.keys())

    px, py = "learning_rate", "perplexity_ratio"
    if px not in gp_params:
        px = gp_params[0]
    if py not in gp_params or py == px:
        remaining = [p for p in gp_params if p != px]
        py = remaining[0] if remaining else None
    if py is None:
        print(f"  {out_path} (skipped — need at least 2 GP params)")
        return

    xi = gp_params.index(px)
    yi = gp_params.index(py)

    # Pass 1: compute grids (needed for shared colour scale)
    fitted: dict[float, tuple] = {}
    for k in all_ks:
        state = states.get(k)
        if state is None:
            continue
        obs = state["observations"]
        X_enc_obs = np.array([o["x_encoded"] for o in obs])

        x_enc_grid = np.linspace(X_enc_obs[:, xi].min(), X_enc_obs[:, xi].max(), n_grid)
        y_enc_grid = np.linspace(X_enc_obs[:, yi].min(), X_enc_obs[:, yi].max(), n_grid)
        xx, yy = np.meshgrid(x_enc_grid, y_enc_grid)

        X_test_enc = np.tile(_best_x_enc(state), (n_grid * n_grid, 1))
        X_test_enc[:, xi] = xx.ravel()
        X_test_enc[:, yi] = yy.ravel()

        mu, sigma = _gp_predict_batch(state, X_test_enc)
        mu_grid = mu.reshape(n_grid, n_grid)
        sigma_grid = sigma.reshape(n_grid, n_grid)

        x_plot = np.exp(x_enc_grid) if px in log_params else x_enc_grid
        y_plot = np.exp(y_enc_grid) if py in log_params else y_enc_grid

        fitted[k] = (state, X_enc_obs, obs, mu_grid, sigma_grid, x_plot, y_plot)

    if not fitted:
        print(f"  {out_path} (skipped — no data)")
        return

    all_mus = np.concatenate([v[3].ravel() for v in fitted.values()])
    vmin, vmax = float(all_mus.min()), float(all_mus.max())
    cmap = "viridis_r" if minimize else "viridis"

    # Pass 2: draw
    n_k = len(fitted)
    ncols = min(3, n_k)
    nrows = math.ceil(n_k / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.2, nrows * 3.8), squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, k in zip(axes_flat, all_ks):
        if k not in fitted:
            ax.set_visible(False)
            continue

        state, X_enc_obs, obs, mu_grid, sigma_grid, x_plot, y_plot = fitted[k]
        mesh_x, mesh_y = np.meshgrid(x_plot, y_plot)

        pcm = ax.pcolormesh(
            mesh_x, mesh_y, mu_grid, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
        )
        try:
            ax.contour(
                mesh_x,
                mesh_y,
                sigma_grid,
                levels=4,
                colors="white",
                linewidths=0.6,
                alpha=0.55,
                linestyles="--",
            )
        except Exception:
            pass

        # Observed points in original scale
        obs_x = [
            np.exp(o["x_encoded"][xi]) if px in log_params else o["x_encoded"][xi]
            for o in obs
        ]
        obs_y = [
            np.exp(o["x_encoded"][yi]) if py in log_params else o["x_encoded"][yi]
            for o in obs
        ]
        ax.scatter(obs_x, obs_y, c="white", s=6, alpha=0.35, linewidths=0, zorder=5)

        if px in log_params:
            ax.set_xscale("log")
        if py in log_params:
            ax.set_yscale("log")

        ax.set_xlabel(px, fontsize=7)
        ax.set_ylabel(py, fontsize=7)
        ax.set_title(f"k={k:+.1f}  (l={state['length_scale']:.2f})", fontsize=8)
        ax.tick_params(labelsize=6)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)

    for ax in axes_flat[n_k:]:
        ax.set_visible(False)

    direction = " (lower=better)" if minimize else " (higher=better)"
    fig.suptitle(
        f"GP surrogate 2D landscape  ·  {px} × {py}  ·  metric: {metric}{direction}\n"
        "colour = posterior mean  |  white dashed = posterior σ  |  dots = observations\n"
        "(remaining parameters held at the values of the best observed trial; shared colour scale across curvatures)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── GP plot 3: Expected Improvement curves ───────────────────────────────────


def plot_gp_ei(
    states: "dict[float, dict]",
    metric: str,
    out_path: str,
    n_grid: int = 120,
) -> None:
    """
    1D Expected Improvement curves for each hyperparameter (Frazier Eq. 7):
      EI(x) = Δ·Φ(Δ/σ) + σ·φ(Δ/σ),  Δ = µₙ(x) − f*ₙ

    Peaks indicate where the Bayesian optimizer would most want to sample next.
    Y-axis is in original metric units.
    """
    if not states:
        print(f"  {out_path} (skipped — no GP states found)")
        return

    first = next(iter(states.values()))
    gp_params = first["param_names"]
    log_params = set(first["log_scale_params"])
    all_ks = sorted(states.keys())
    n_p = len(gp_params)
    ncols = min(3, n_p)
    nrows = math.ceil(n_p / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4.5, nrows * 3.0), squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, sweep_param in zip(axes_flat, gp_params):
        is_log = sweep_param in log_params
        sweep_idx = gp_params.index(sweep_param)

        for k in all_ks:
            state = states.get(k)
            if state is None:
                continue
            col = k_color(k, all_ks)
            obs = state["observations"]
            X_enc_obs = np.array([o["x_encoded"] for o in obs])

            enc_lo = X_enc_obs[:, sweep_idx].min()
            enc_hi = X_enc_obs[:, sweep_idx].max()
            enc_grid = np.linspace(enc_lo, enc_hi, n_grid)

            X_test_enc = np.tile(_best_x_enc(state), (n_grid, 1))
            X_test_enc[:, sweep_idx] = enc_grid

            ei = _gp_ei_batch(state, X_test_enc)
            x_plot = np.exp(enc_grid) if is_log else enc_grid

            ax.plot(
                x_plot, ei, color=col, linewidth=1.8, label=f"k={k:+.1f}", alpha=0.9
            )

        if is_log:
            ax.set_xscale("log")
        ax.set_title(sweep_param, fontsize=9)
        ax.set_xlabel(sweep_param + (" (log scale)" if is_log else ""), fontsize=7)
        ax.set_ylabel("Expected Improvement", fontsize=7)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n_p:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=7,
            ncol=min(5, len(all_ks)),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.suptitle(
        f"GP surrogate — Expected Improvement (EI)  ·  metric: {metric}\n"
        "EI = Δ·Φ(Δ/σ) + σ·φ(Δ/σ)  where  Δ = µₙ(x) − f*ₙ  (Frazier Eq. 7)\n"
        "peaks = where the optimizer would sample next  |  (others held at best observed trial)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Scan data loading ────────────────────────────────────────────────────────


def load_scan_results(path: str) -> list[dict]:
    """Load scan trial records from a single JSONL file (filtered by scan_param presence)."""
    return [r for r in _load_jsonl(path) if r.get("scan_param")]


def _scan_params_ordered(records: list[dict]) -> list[str]:
    """Return scan parameters in the order they first appear in the data."""
    seen: list[str] = []
    for r in records:
        p = r.get("scan_param")
        if p and p not in seen:
            seen.append(p)
    return seen


def _base_config(records: list[dict], sweep_param: str, curvature: float) -> dict:
    """
    Recover the base (fixed) config for a curvature by reading the non-swept
    parameter values from any record where a *different* parameter is being swept.
    """
    for r in records:
        if r["curvature"] == curvature and r.get("scan_param") != sweep_param:
            return r
    return {}


# ─── Scan plot 1: effect curves ───────────────────────────────────────────────


def plot_scan_effects(
    records: list[dict], out_path: str, metric: str | None = None
) -> None:
    """
    One subplot per swept parameter. Each subplot shows metric vs parameter
    value, one line per curvature. A vertical dashed line marks the base-config
    value for that parameter.
    """
    all_ks = sorted({r["curvature"] for r in records})
    sweep_params = _scan_params_ordered(records)
    metric_label = metric or "metric"
    n_params = len(sweep_params)
    if n_params == 0:
        print(f"  {out_path} (skipped — no scan_param field found)")
        return

    ncols = min(3, n_params)
    nrows = math.ceil(n_params / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.4))
    axes_flat = np.array(axes).flatten() if n_params > 1 else [axes]

    for ax, param in zip(axes_flat, sweep_params):
        is_log = param in LOG_SCALE_PARAMS

        for k in all_ks:
            group = sorted(
                [
                    r
                    for r in records
                    if r["curvature"] == k and r.get("scan_param") == param
                ],
                key=lambda r: get_param_value(r, param) or 0,
            )
            if not group:
                continue

            xs = np.array([get_param_value(r, param) for r in group])
            ys = np.array([get_metric_value(r, metric) or 0.0 for r in group])
            col = k_color(k, all_ks)

            ax.plot(
                xs,
                ys,
                color=col,
                linewidth=1.8,
                label=f"k={k:+.1f}",
                marker="o",
                markersize=3.5,
                zorder=3,
            )

        # Mark base-config value with a vertical dashed line.
        base = _base_config(records, param, all_ks[0])
        if param in base:
            ax.axvline(
                get_param_value(base, param),
                color="black",
                linestyle=":",
                linewidth=1.2,
                alpha=0.6,
                label="base config",
            )

        if is_log:
            ax.set_xscale("log")

        ax.set_title(param, fontsize=9)
        ax.set_xlabel(param + (" (log)" if is_log else ""), fontsize=7)
        ax.set_ylabel(metric_label, fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    # Shared legend (curvature lines)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        ncol=min(5, len(all_ks) + 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Scan: effect of each parameter on {metric_label} (others fixed at base config)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Scan plot 2: sensitivity heatmap ─────────────────────────────────────────


def plot_scan_sensitivity(
    records: list[dict], out_path: str, metric: str | None = None
) -> None:
    """
    Heatmap of metric range (max − min) across the sweep, for each
    (parameter × curvature). Larger range = more sensitive to that parameter.
    """
    all_ks = sorted({r["curvature"] for r in records})
    sweep_params = _scan_params_ordered(records)
    metric_label = metric or "metric"
    if not sweep_params:
        return

    matrix = np.full((len(sweep_params), len(all_ks)), np.nan)
    for i, param in enumerate(sweep_params):
        for j, k in enumerate(all_ks):
            group = [
                r
                for r in records
                if r["curvature"] == k and r.get("scan_param") == param
            ]
            if len(group) < 2:
                continue
            vals = [get_metric_value(r, metric) for r in group]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                continue
            matrix[i, j] = max(vals) - min(vals)

    fig, ax = plt.subplots(
        figsize=(len(all_ks) * 0.95 + 2.0, len(sweep_params) * 0.55 + 1.8)
    )

    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(all_ks)))
    ax.set_xticklabels([f"k={k:+.1f}" for k in all_ks], rotation=45, ha="right")
    ax.set_yticks(range(len(sweep_params)))
    ax.set_yticklabels(sweep_params)

    for i in range(len(sweep_params)):
        for j in range(len(all_ks)):
            if not np.isnan(matrix[i, j]):
                color = "white" if matrix[i, j] > vmax * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    plt.colorbar(im, ax=ax, label="metric range (max − min)", fraction=0.03, pad=0.02)
    ax.set_title(
        f"Scan sensitivity: {metric_label} range per (parameter × curvature)\n"
        "larger = more impact when this parameter is varied",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Scan plot 3: optimal value location ──────────────────────────────────────


def plot_scan_optimal(
    records: list[dict], out_path: str, metric: str | None = None
) -> None:
    """
    For each (parameter × curvature), show where in the sweep range the optimum
    lies, normalised to [0, 1] (0 = low end of range, 1 = high end).
    Discrete parameters show the optimal category index.
    Helps answer: "is the optimum always near the boundary, or in the interior?"
    """
    all_ks = sorted({r["curvature"] for r in records})
    sweep_params = _scan_params_ordered(records)
    metric_label = metric or "metric"
    if not sweep_params:
        return

    # Two matrices: normalised optimal position, and actual optimal value
    pos_matrix = np.full((len(sweep_params), len(all_ks)), np.nan)
    val_matrix = np.full((len(sweep_params), len(all_ks)), np.nan)

    for i, param in enumerate(sweep_params):
        for j, k in enumerate(all_ks):
            group = sorted(
                [
                    r
                    for r in records
                    if r["curvature"] == k
                    and r.get("scan_param") == param
                    and get_metric_value(r, metric) is not None
                ],
                key=lambda r: get_param_value(r, param) or 0,
            )
            if not group:
                continue
            xs = [get_param_value(r, param) or 0 for r in group]
            ys = [get_metric_value(r, metric) for r in group]
            best_idx = int(
                np.argmin(ys) if metric in MINIMIZE_METRICS else np.argmax(ys)  # type: ignore
            )
            val_matrix[i, j] = ys[best_idx]
            # Normalise position: 0 = lowest x, 1 = highest x
            x_min, x_max = min(xs), max(xs)
            if x_max > x_min:
                pos_matrix[i, j] = (xs[best_idx] - x_min) / (x_max - x_min)
            else:
                pos_matrix[i, j] = 0.5

    fig, axes = plt.subplots(
        1, 2, figsize=(len(all_ks) * 1.6 + 2.5, len(sweep_params) * 0.6 + 2.2)
    )

    # Left: optimal position (0=low end, 1=high end)
    # Diverging colormap centred at 0.5 highlights boundary-sitting optima
    im0 = axes[0].imshow(pos_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(all_ks)))
    axes[0].set_xticklabels([f"k={k:+.1f}" for k in all_ks], rotation=45, ha="right")
    axes[0].set_yticks(range(len(sweep_params)))
    axes[0].set_yticklabels(sweep_params)
    for i in range(len(sweep_params)):
        for j in range(len(all_ks)):
            if not np.isnan(pos_matrix[i, j]):
                v = pos_matrix[i, j]
                color = "black" if 0.25 < v < 0.75 else "white"
                axes[0].text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color
                )
    plt.colorbar(
        im0,
        ax=axes[0],
        label="normalised position of optimum\n(0=low end, 1=high end)",
        fraction=0.04,
        pad=0.02,
    )
    axes[0].set_title(
        "Where is the optimum?\n(red=low end, green=high end, yellow=interior)"
    )

    # Right: best metric value achieved
    vmin2 = np.nanmin(val_matrix)
    vmax2 = np.nanmax(val_matrix)
    im1 = axes[1].imshow(
        val_matrix, cmap="Blues", vmin=vmin2, vmax=vmax2, aspect="auto"
    )
    axes[1].set_xticks(range(len(all_ks)))
    axes[1].set_xticklabels([f"k={k:+.1f}" for k in all_ks], rotation=45, ha="right")
    axes[1].set_yticks(range(len(sweep_params)))
    axes[1].set_yticklabels([""] * len(sweep_params))
    for i in range(len(sweep_params)):
        for j in range(len(all_ks)):
            if not np.isnan(val_matrix[i, j]):
                v = val_matrix[i, j]
                color = "white" if v > vmin2 + (vmax2 - vmin2) * 0.6 else "black"
                axes[1].text(
                    j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=color
                )
    plt.colorbar(
        im1, ax=axes[1], label="best metric value in sweep", fraction=0.04, pad=0.02
    )
    axes[1].set_title(f"Best {metric_label} achieved\nat the optimal sweep point")

    fig.suptitle(
        f"Scan: optimal sweep point per (parameter × curvature) — {metric_label}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity analysis for fitting-curvature"
    )
    parser.add_argument(
        "--mode",
        default="optimize",
        choices=["optimize", "scan", "gp"],
        help="Analysis mode: 'optimize' (default), 'scan', or 'gp'",
    )
    parser.add_argument(
        "--input",
        default="results/results.jsonl",
        help="JSONL results file (default: results/results.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="plots",
        help="Output directory for SVG files (default: plots/)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric column to analyze, e.g. 'davies_bouldin_ratio'. Auto-detected from present columns if not set.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=5,
        help="Number of top-importance parameters to include in pairwise plot (default: 5)",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=0.1,
        help="Fraction of top runs used in the consensus plot, e.g. 0.1 = top 10% (default: 0.1)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    metric = args.metric

    if args.mode == "gp":
        if not metric:
            print("Error: --metric is required for --mode gp")
            return
        stem = Path(args.input).stem
        print(f"Loading GP states matching '{stem}_gp_*.json' ...")
        states = load_gp_states(args.input)
        if not states:
            print(
                "No GP state files found. Run the optimizer with --mode bayes first.\n"
                f"Expected files matching: {stem}_gp_*.json"
            )
            return
        print(f"Loaded GP states for {len(states)} curvature(s): {sorted(states)}")
        print("Generating GP plots:")
        plot_gp_slices(states, metric, os.path.join(args.output, "gp_slices.svg"))
        plot_gp_landscape(states, metric, os.path.join(args.output, "gp_landscape.svg"))
        plot_gp_ei(states, metric, os.path.join(args.output, "gp_ei.svg"))
        print(f"\nDone. All plots saved to '{args.output}/'.")
        return

    if args.mode == "scan":
        print(f"Loading scan results from '{args.input}' ...")
        records = load_scan_results(args.input)
        if not records:
            print("No scan results found. Run the optimizer with --mode scan first.")
            return
        n_curvatures = len({r["curvature"] for r in records})
        sweep_params = _scan_params_ordered(records)
        print(f"Loaded {len(records)} scan records across {n_curvatures} curvatures.")
        print(f"Parameters swept: {sweep_params}")
        print(f"Metric: {metric or 'not specified — scan plots will be skipped'}\n")

        if metric is not None:
            print("Generating scan plots:")
            plot_scan_effects(
                records, os.path.join(args.output, "scan_effects.svg"), metric
            )
            plot_scan_sensitivity(
                records, os.path.join(args.output, "scan_sensitivity.svg"), metric
            )
            plot_scan_optimal(
                records, os.path.join(args.output, "scan_optimal.svg"), metric
            )
        else:
            print("No plots generated (pass --metric to enable scan plots).")

        print(f"\nDone. All plots saved to '{args.output}/'.")
        return

    # ── optimize mode ──────────────────────────────────────────────────────────
    print(f"Loading results from '{args.input}' ...")
    records = load_results(args.input)
    if not records:
        print("No results found. Check --input path.")
        return
    print(
        f"Loaded {len(records)} trials across {len({r['curvature'] for r in records})} curvatures."
    )

    print(
        f"Metric: {metric or 'not specified — metric-specific plots will be skipped'}\n"
    )

    params = present_params(records)
    print(f"Parameters: {params}\n")

    print("Generating plots:")
    # Metric-independent plots (always generated)
    plot_metric_correlation(
        records, os.path.join(args.output, "metric_correlation.svg")
    )
    plot_variable_importance_heatmap(
        records, params, os.path.join(args.output, "importance_heatmap.svg")
    )

    if metric is not None:
        # Metric-specific plots (skipped when --metric is not given)
        correlations = compute_correlations(records, params, metric)

        plot_spearman_heatmap(
            correlations, params, os.path.join(args.output, "spearman_heatmap.svg")
        )
        plot_param_importance(
            correlations, params, os.path.join(args.output, "param_importance.svg")
        )
        plot_param_vs_metric(records, params, args.output, metric)

        plot_marginal_effects(
            records,
            params,
            os.path.join(args.output, "marginal_effects.svg"),
            metric,
        )
        plot_good_regions(
            records,
            params,
            os.path.join(args.output, "good_regions.svg"),
            metric,
            top_pct=args.top_pct,
        )
        print_good_regions(records, params, metric, args.top_pct)

    print(f"\nDone. All plots saved to '{args.output}/'.")


if __name__ == "__main__":
    main()
