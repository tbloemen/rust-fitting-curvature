"""
Hyperparameter sensitivity analysis for fitting-curvature optimizer results.

--mode optimize (default)
  1. spearman_heatmap.svg         — Spearman ρ per (parameter × curvature)
  2. param_importance.svg         — Mean |ρ| across curvatures (bar chart)
  3. param_<name>_vs_metric.svg   — Per-parameter scatter vs metric, one series per curvature
  4. convergence.svg              — Best metric found vs trial number per curvature
  5. pairwise_top<N>.svg          — Pairwise scatter for the top-N most important parameters

--mode scan
  1. scan_effects.svg             — Clean effect curve per parameter (metric vs swept value)
  2. scan_sensitivity.svg         — Heatmap of metric range (max−min) per (parameter × curvature)
  3. scan_optimal.svg             — Where in the sweep range the optimum sits per (param × curvature)

Supports both old format (metric_mean/metric_std/trial fields) and new format
(individual metric columns like trustworthiness, davies_bouldin_ratio, etc.).
Use --metric to select which metric column to analyze when using the new format.

Usage:
    uv run analyze_hyperparams.py --metric davies_bouldin_ratio
    uv run analyze_hyperparams.py --mode scan --input results/results --output plots/
    uv run analyze_hyperparams.py --input results/results --output plots/ --top-pairs 5
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
    "scaling_loss",
    "centering_weight",
    "global_loss_weight",
    "norm_loss_weight",
    "n_iterations",  # old format (now fixed)
    "early_exaggeration_iterations",  # old format (now fixed)
]

LOG_SCALE_PARAMS = {"learning_rate", "perplexity", "perplexity_ratio"}

# Scaling loss: string ↔ numeric mapping for backward compatibility.
SCALING_LOSS_NAMES = ["none", "hard_barrier", "softplus_barrier", "rms", "mean_distance"]
SCALING_LOSS_TO_NUM = {name: i for i, name in enumerate(SCALING_LOSS_NAMES)}

ALL_METRICS = [
    "trustworthiness", "continuity", "knn_overlap",
    "geodesic_distortion_gu2019", "geodesic_distortion_mse",
    "davies_bouldin_ratio", "dunn_index",
    "class_density_measure", "cluster_density_measure",
]


def scaling_loss_to_numeric(val) -> float:
    """Convert a scaling_loss value (string or int) to a numeric value."""
    if isinstance(val, str):
        return float(SCALING_LOSS_TO_NUM.get(val, 0))
    return float(val)


def scaling_loss_label(val) -> str:
    """Convert a scaling_loss value (string or int) to a human-readable label."""
    if isinstance(val, str):
        return val.replace("_", " ").title()
    labels = {0: "None", 1: "Hard Barrier", 2: "Softplus Barrier", 3: "Rms", 4: "Mean Distance"}
    return labels.get(int(val), str(val))


def get_metric_value(record: dict, metric: str | None) -> float | None:
    """Extract the metric value from a record.

    Tries the explicit metric column first (new format), then falls back
    to metric_mean (old format).
    """
    if metric and metric in record and record[metric] is not None:
        return record[metric]
    if "metric_mean" in record and record["metric_mean"] is not None:
        return record["metric_mean"]
    return None


def get_param_value(record: dict, param: str) -> float | None:
    """Extract a parameter value, converting scaling_loss strings to numeric."""
    val = record.get(param)
    if val is None:
        return None
    if param == "scaling_loss":
        return scaling_loss_to_numeric(val)
    return float(val)

# Curvature color map (tab10 via curvature index)
_CMAP_K = plt.cm.tab10


def k_color(k: float, all_ks: list[float]) -> tuple:
    idx = sorted(all_ks).index(k)
    return _CMAP_K(idx / max(len(all_ks) - 1, 1))


# ─── Data loading ─────────────────────────────────────────────────────────────


def _rust_output_path(prefix: str, curvature: float, suffix: str = "") -> str:
    """Replicate the Rust output_path() convention: dots replaced with underscores."""
    return f"{prefix}_k{curvature:.1f}{suffix}.jsonl".replace(".", "_")


def load_results(prefix: str) -> list[dict]:
    """Load all JSONL result files matching the Rust output convention."""
    records = []
    for k in CURVATURES:
        path = _rust_output_path(prefix, k)
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def present_params(records: list[dict]) -> list[str]:
    """Return only params that actually appear in the data, preserving order."""
    if not records:
        return []
    keys = set().union(*(r.keys() for r in records))
    return [p for p in ALL_PARAMS if p in keys]


# ─── Spearman helpers ─────────────────────────────────────────────────────────


def compute_correlations(
    records: list[dict], params: list[str], metric: str | None = None,
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
                for r in group if param in r
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
    records: list[dict], params: list[str], out_dir: str, metric: str | None = None,
) -> None:
    all_ks = sorted({r["curvature"] for r in records})
    metric_label = metric or "metric_mean"

    for param in params:
        is_log = param in LOG_SCALE_PARAMS
        is_cat = param == "scaling_loss"

        if is_cat:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # Box plot per category value, grouped by curvature
            cat_vals = sorted({scaling_loss_to_numeric(r[param]) for r in records if param in r})
            width = 0.8 / len(all_ks)
            for ki, k in enumerate(all_ks):
                group = [r for r in records if r["curvature"] == k and param in r]
                positions = [
                    ci + ki * width - 0.4 + width / 2 for ci in range(len(cat_vals))
                ]
                data = [
                    [get_metric_value(r, metric) for r in group
                     if scaling_loss_to_numeric(r[param]) == c and get_metric_value(r, metric) is not None]
                    for c in cat_vals
                ]
                data = [d for d in data if d]  # drop empty
                if not data:
                    continue
                ax.boxplot(
                    data,
                    positions=[p for p, d in zip(positions, data) if d],
                    widths=width * 0.8,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    boxprops=dict(facecolor=(*k_color(k, all_ks)[:3], 0.6)),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4),
                )
            ax.set_xticks(range(len(cat_vals)))
            ax.set_xticklabels([scaling_loss_label(SCALING_LOSS_NAMES[int(c)]) for c in cat_vals])
            ax.set_xlabel("scaling_loss variant")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for k in all_ks:
                group = [
                    r for r in records
                    if r["curvature"] == k and param in r and get_metric_value(r, metric) is not None
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


# ─── Plot 4: Optimizer convergence ────────────────────────────────────────────


def plot_convergence(records: list[dict], out_path: str, metric: str | None = None) -> None:
    all_ks = sorted({r["curvature"] for r in records})
    metric_label = metric or "metric_mean"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for k in all_ks:
        group = [r for r in records if r["curvature"] == k and get_metric_value(r, metric) is not None]
        if not group:
            continue
        col = k_color(k, all_ks)
        trials = list(range(1, len(group) + 1))
        metrics = [get_metric_value(r, metric) for r in group]
        best_so_far = [max(metrics[: i + 1]) for i in range(len(metrics))]

        axes[0].plot(trials, metrics, color=col, alpha=0.4, linewidth=0.8)
        axes[0].scatter(trials, metrics, color=col, s=5, alpha=0.5)
        axes[1].plot(trials, best_so_far, color=col, linewidth=1.8, label=f"k={k:+.1f}")

    axes[0].set_title(f"{metric_label} per trial")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel(metric_label)

    axes[1].set_title(f"Best {metric_label} found so far")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel(f"Best {metric_label}")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")

    fig.suptitle("Convergence by curvature", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Plot 5: Pairwise interaction grid ────────────────────────────────────────


def plot_pairwise_interactions(
    records: list[dict],
    top_params: list[str],
    out_path: str,
    metric: str | None = None,
) -> None:
    """
    Grid of pairwise scatter plots for top_params.
    Upper triangle: scatter colored by metric value.
    Diagonal:       KDE / histogram of each parameter.
    Lower triangle: scatter colored by curvature.
    """
    all_ks = sorted({r["curvature"] for r in records})
    n = len(top_params)
    metric_label = metric or "metric_mean"

    fig, axes = plt.subplots(n, n, figsize=(n * 2.3, n * 2.1))
    if n == 1:
        axes = np.array([[axes]])

    metrics = np.array([get_metric_value(r, metric) or 0.0 for r in records])
    norm_metric = mcolors.Normalize(
        vmin=np.percentile(metrics, 5), vmax=np.percentile(metrics, 95)
    )
    cmap_metric = cm.plasma

    def get_col(k):
        return k_color(k, all_ks)

    for i, pi in enumerate(top_params):
        for j, pj in enumerate(top_params):
            ax = axes[i, j]

            xi = np.array([get_param_value(r, pi) if pi in r else np.nan for r in records], dtype=float)
            xj = np.array([get_param_value(r, pj) if pj in r else np.nan for r in records], dtype=float)
            m = np.array([get_metric_value(r, metric) or np.nan for r in records], dtype=float)
            k_vals = np.array([r["curvature"] for r in records])

            if i == j:
                # Diagonal: overlapping histograms per curvature
                valid = xi[~np.isnan(xi)]
                bins = 20
                log = pi in LOG_SCALE_PARAMS
                if log and (valid > 0).all():
                    edges = np.exp(
                        np.linspace(np.log(valid.min()), np.log(valid.max()), bins + 1)
                    )
                else:
                    edges = np.linspace(np.nanmin(valid), np.nanmax(valid), bins + 1)
                for k in all_ks:
                    mask = (k_vals == k) & ~np.isnan(xi)
                    if mask.sum() < 2:
                        continue
                    ax.hist(
                        xi[mask],
                        bins=edges,
                        alpha=0.4,
                        color=get_col(k),
                        density=True,
                        histtype="stepfilled",
                    )
                ax.set_xlabel(pi, fontsize=7)
                if pi in LOG_SCALE_PARAMS:
                    ax.set_xscale("log")

            elif i < j:
                # Upper triangle: colored by metric
                mask = ~(np.isnan(xi) | np.isnan(xj))
                ax.scatter(
                    xj[mask],
                    xi[mask],
                    c=m[mask],
                    cmap=cmap_metric,
                    norm=norm_metric,
                    s=6,
                    alpha=0.6,
                    linewidths=0,
                )
            else:
                # Lower triangle: colored by curvature
                for k in all_ks:
                    mask = (k_vals == k) & ~np.isnan(xi) & ~np.isnan(xj)
                    if mask.sum() < 2:
                        continue
                    ax.scatter(
                        xj[mask],
                        xi[mask],
                        color=get_col(k),
                        s=6,
                        alpha=0.45,
                        linewidths=0,
                    )

            # Axis labels only on edges
            if j == 0 and i != j:
                ax.set_ylabel(pi, fontsize=7)
            else:
                ax.set_ylabel("")
            if i == n - 1 and i != j:
                ax.set_xlabel(pj, fontsize=7)
            else:
                ax.set_xlabel("")
            ax.tick_params(labelsize=6)

            if pi in LOG_SCALE_PARAMS and i != j:
                ax.set_yscale("log")
            if pj in LOG_SCALE_PARAMS and i != j:
                ax.set_xscale("log")

    # Shared colorbar for metric (upper triangle)
    sm = cm.ScalarMappable(cmap=cmap_metric, norm=norm_metric)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes, label=f"{metric_label} (upper triangle)", shrink=0.6, pad=0.02
    )
    cbar.ax.tick_params(labelsize=7)

    # Legend for curvature (lower triangle)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=get_col(k),
            markersize=5,
            label=f"k={k:+.1f}",
        )
        for k in all_ks
    ]
    fig.legend(
        handles=handles,
        fontsize=6,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        ncol=3,
        title="curvature\n(lower triangle)",
        title_fontsize=6,
    )

    fig.suptitle(
        f"Pairwise interactions — top {n} parameters\n"
        "upper: colored by metric | lower: colored by curvature | diagonal: distribution per k",
        y=1.01,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Scan data loading ────────────────────────────────────────────────────────

def load_scan_results(prefix: str) -> list[dict]:
    """Load all scan JSONL files matching the Rust scan output convention."""
    records = []
    for k in CURVATURES:
        path = _rust_output_path(prefix, k, suffix="_scan")
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


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

def plot_scan_effects(records: list[dict], out_path: str, metric: str | None = None) -> None:
    """
    One subplot per swept parameter. Each subplot shows metric vs parameter
    value, one line per curvature. A vertical dashed line marks the base-config
    value for that parameter.
    """
    all_ks = sorted({r["curvature"] for r in records})
    sweep_params = _scan_params_ordered(records)
    metric_label = metric or "metric_mean"
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
        is_cat = param == "scaling_loss"

        for k in all_ks:
            group = sorted(
                [r for r in records if r["curvature"] == k and r.get("scan_param") == param],
                key=lambda r: get_param_value(r, param) or 0,
            )
            if not group:
                continue

            xs = np.array([get_param_value(r, param) for r in group])
            ys = np.array([get_metric_value(r, metric) or 0.0 for r in group])
            col = k_color(k, all_ks)

            ax.plot(xs, ys, color=col, linewidth=1.8, label=f"k={k:+.1f}", marker="o",
                    markersize=3.5, zorder=3)

        # Mark base-config value with a vertical dashed line.
        base = _base_config(records, param, all_ks[0])
        if param in base:
            ax.axvline(get_param_value(base, param), color="black", linestyle=":", linewidth=1.2,
                       alpha=0.6, label="base config")

        if is_log:
            ax.set_xscale("log")
        if is_cat:
            ticks = list(range(len(SCALING_LOSS_NAMES)))
            ax.set_xticks(ticks)
            ax.set_xticklabels([scaling_loss_label(n) for n in SCALING_LOSS_NAMES], fontsize=7, rotation=20)

        ax.set_title(param, fontsize=9)
        ax.set_xlabel(param + (" (log)" if is_log else ""), fontsize=7)
        ax.set_ylabel(metric_label, fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    # Shared legend (curvature lines)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=min(5, len(all_ks) + 1),
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Scan: effect of each parameter on {metric_label} (others fixed at base config)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Scan plot 2: sensitivity heatmap ─────────────────────────────────────────

def plot_scan_sensitivity(records: list[dict], out_path: str, metric: str | None = None) -> None:
    """
    Heatmap of metric range (max − min) across the sweep, for each
    (parameter × curvature). Larger range = more sensitive to that parameter.
    """
    all_ks = sorted({r["curvature"] for r in records})
    sweep_params = _scan_params_ordered(records)
    metric_label = metric or "metric_mean"
    if not sweep_params:
        return

    matrix = np.full((len(sweep_params), len(all_ks)), np.nan)
    for i, param in enumerate(sweep_params):
        for j, k in enumerate(all_ks):
            group = [r for r in records if r["curvature"] == k and r.get("scan_param") == param]
            if len(group) < 2:
                continue
            vals = [get_metric_value(r, metric) for r in group]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                continue
            matrix[i, j] = max(vals) - min(vals)

    fig, ax = plt.subplots(figsize=(len(all_ks) * 0.95 + 2.0, len(sweep_params) * 0.55 + 1.8))

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
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="metric range (max − min)", fraction=0.03, pad=0.02)
    ax.set_title(f"Scan sensitivity: {metric_label} range per (parameter × curvature)\n"
                 "larger = more impact when this parameter is varied", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Scan plot 3: optimal value location ──────────────────────────────────────

def plot_scan_optimal(records: list[dict], out_path: str, metric: str | None = None) -> None:
    """
    For each (parameter × curvature), show where in the sweep range the optimum
    lies, normalised to [0, 1] (0 = low end of range, 1 = high end).
    Discrete parameters show the optimal category index.
    Helps answer: "is the optimum always near the boundary, or in the interior?"
    """
    all_ks = sorted({r["curvature"] for r in records})
    sweep_params = _scan_params_ordered(records)
    metric_label = metric or "metric_mean"
    if not sweep_params:
        return

    # Two matrices: normalised optimal position, and actual optimal value
    pos_matrix = np.full((len(sweep_params), len(all_ks)), np.nan)
    val_matrix = np.full((len(sweep_params), len(all_ks)), np.nan)

    for i, param in enumerate(sweep_params):
        for j, k in enumerate(all_ks):
            group = sorted(
                [r for r in records if r["curvature"] == k and r.get("scan_param") == param
                 and get_metric_value(r, metric) is not None],
                key=lambda r: get_param_value(r, param) or 0,
            )
            if not group:
                continue
            xs = [get_param_value(r, param) or 0 for r in group]
            ys = [get_metric_value(r, metric) for r in group]
            best_idx = int(np.argmax(ys))
            val_matrix[i, j] = ys[best_idx]
            # Normalise position: 0 = lowest x, 1 = highest x
            x_min, x_max = min(xs), max(xs)
            if x_max > x_min:
                pos_matrix[i, j] = (xs[best_idx] - x_min) / (x_max - x_min)
            else:
                pos_matrix[i, j] = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(len(all_ks) * 1.6 + 2.5, len(sweep_params) * 0.6 + 2.2))

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
                axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=8, color=color)
    plt.colorbar(im0, ax=axes[0], label="normalised position of optimum\n(0=low end, 1=high end)",
                 fraction=0.04, pad=0.02)
    axes[0].set_title("Where is the optimum?\n(red=low end, green=high end, yellow=interior)")

    # Right: best metric value achieved
    vmin2 = np.nanmin(val_matrix)
    vmax2 = np.nanmax(val_matrix)
    im1 = axes[1].imshow(val_matrix, cmap="Blues", vmin=vmin2, vmax=vmax2, aspect="auto")
    axes[1].set_xticks(range(len(all_ks)))
    axes[1].set_xticklabels([f"k={k:+.1f}" for k in all_ks], rotation=45, ha="right")
    axes[1].set_yticks(range(len(sweep_params)))
    axes[1].set_yticklabels([""] * len(sweep_params))
    for i in range(len(sweep_params)):
        for j in range(len(all_ks)):
            if not np.isnan(val_matrix[i, j]):
                v = val_matrix[i, j]
                color = "white" if v > vmin2 + (vmax2 - vmin2) * 0.6 else "black"
                axes[1].text(j, i, f"{v:.3f}", ha="center", va="center",
                             fontsize=8, color=color)
    plt.colorbar(im1, ax=axes[1], label="best metric value in sweep",
                 fraction=0.04, pad=0.02)
    axes[1].set_title(f"Best {metric_label} achieved\nat the optimal sweep point")

    fig.suptitle(f"Scan: optimal sweep point per (parameter × curvature) — {metric_label}", fontsize=10)
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
        choices=["optimize", "scan"],
        help="Analysis mode: 'optimize' (default) or 'scan'",
    )
    parser.add_argument(
        "--input",
        default="results/results",
        help="JSONL file prefix (default: results/results)",
    )
    parser.add_argument(
        "--output",
        default="plots",
        help="Output directory for SVG files (default: plots/)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric column to analyze, e.g. 'davies_bouldin_ratio'. "
             "Falls back to 'metric_mean' field if not set (old format).",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=5,
        help="Number of top-importance parameters to include in pairwise plot (default: 5)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    metric = args.metric

    if args.mode == "scan":
        print(f"Loading scan results from '{args.input}_k*_scan.jsonl' ...")
        records = load_scan_results(args.input)
        if not records:
            print("No scan results found. Run the optimizer with --mode scan first.")
            return
        n_curvatures = len({r["curvature"] for r in records})
        sweep_params = _scan_params_ordered(records)
        print(f"Loaded {len(records)} scan records across {n_curvatures} curvatures.")
        print(f"Parameters swept: {sweep_params}")
        print(f"Metric: {metric or 'metric_mean (auto)'}\n")

        print("Generating scan plots:")
        plot_scan_effects(records, os.path.join(args.output, "scan_effects.svg"), metric)
        plot_scan_sensitivity(records, os.path.join(args.output, "scan_sensitivity.svg"), metric)
        plot_scan_optimal(records, os.path.join(args.output, "scan_optimal.svg"), metric)

        print(f"\nDone. All plots saved to '{args.output}/'.")
        return

    # ── optimize mode ──────────────────────────────────────────────────────────
    print(f"Loading results from '{args.input}_k*.jsonl' ...")
    records = load_results(args.input)
    if not records:
        print("No results found. Check --input prefix.")
        return
    print(
        f"Loaded {len(records)} trials across {len({r['curvature'] for r in records})} curvatures."
    )
    print(f"Metric: {metric or 'metric_mean (auto)'}\n")

    params = present_params(records)
    print(f"Parameters: {params}\n")

    correlations = compute_correlations(records, params, metric)
    importance = mean_abs_rho(correlations, params)
    top_params = sorted(importance, key=lambda p: importance[p], reverse=True)[
        : args.top_pairs
    ]

    print("Generating plots:")
    plot_spearman_heatmap(
        correlations, params, os.path.join(args.output, "spearman_heatmap.svg")
    )
    plot_param_importance(
        correlations, params, os.path.join(args.output, "param_importance.svg")
    )
    plot_convergence(records, os.path.join(args.output, "convergence.svg"), metric)
    plot_param_vs_metric(records, params, args.output, metric)

    print(f"\nTop {args.top_pairs} for pairwise plot: {top_params}")
    if not top_params:
        print("  (skipped pairwise — not enough data for correlations)")
        print(f"\nDone. All plots saved to '{args.output}'.")
        return
    plot_pairwise_interactions(
        records,
        top_params,
        os.path.join(args.output, f"pairwise_top{args.top_pairs}.svg"),
        metric,
    )

    print(f"\nDone. All plots saved to '{args.output}/'.")


if __name__ == "__main__":
    main()
