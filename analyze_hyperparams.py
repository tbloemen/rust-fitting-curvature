"""
Hyperparameter sensitivity analysis for fitting-curvature optimizer results.

Produces SVG plots:
  1. spearman_heatmap.svg         — Spearman ρ per (parameter × curvature)
  2. param_importance.svg         — Mean |ρ| across curvatures (bar chart)
  3. param_<name>_vs_metric.svg   — Per-parameter scatter vs metric, one series per curvature
  4. convergence.svg              — Best metric found vs trial number per curvature
  5. pairwise_top<N>.svg          — Pairwise scatter for the top-N most important parameters

Usage:
    uv run analyze_hyperparams.py
    uv run analyze_hyperparams.py --input results --output plots/ --top-pairs 5
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

# Curvature color map (tab10 via curvature index)
_CMAP_K = plt.cm.tab10


def k_color(k: float, all_ks: list[float]) -> tuple:
    idx = sorted(all_ks).index(k)
    return _CMAP_K(idx / max(len(all_ks) - 1, 1))


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_results(prefix: str) -> list[dict]:
    """Load all JSONL result files matching <prefix>_k<curvature>.jsonl."""
    records = []
    for k in CURVATURES:
        path = f"{prefix}_k{k:.1f}.jsonl"
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
    records: list[dict], params: list[str]
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
            pairs = [(r[param], r["metric_mean"]) for r in group if param in r]
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


def plot_param_vs_metric(records: list[dict], params: list[str], out_dir: str) -> None:
    all_ks = sorted({r["curvature"] for r in records})

    for param in params:
        is_log = param in LOG_SCALE_PARAMS
        is_cat = param == "scaling_loss"

        if is_cat:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # Box plot per category value, grouped by curvature
            categories = sorted({int(r[param]) for r in records if param in r})
            width = 0.8 / len(all_ks)
            for ki, k in enumerate(all_ks):
                group = [r for r in records if r["curvature"] == k and param in r]
                positions = [
                    c + ki * width - 0.4 + width / 2 for c in range(len(categories))
                ]
                data = [
                    [r["metric_mean"] for r in group if int(r[param]) == c]
                    for c in categories
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
            cat_labels = {
                0: "None",
                1: "HardBarrier",
                2: "Softplus",
                3: "Rms",
                4: "MeanDist",
            }
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([cat_labels.get(c, str(c)) for c in categories])
            ax.set_xlabel("scaling_loss variant")
        else:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for k in all_ks:
                group = [r for r in records if r["curvature"] == k and param in r]
                if not group:
                    continue
                col = k_color(k, all_ks)
                xs = np.array([r[param] for r in group])
                ys = np.array([r["metric_mean"] for r in group])
                # Scatter (size ∝ 1/std for certainty)
                stds = np.array([r.get("metric_std", 0.1) for r in group])
                sizes = np.clip(20 / (stds + 0.01), 3, 60)
                ax.scatter(xs, ys, color=col, alpha=0.35, s=sizes, linewidths=0)
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

        ax.set_ylabel("metric_mean")
        ax.set_title(f"Effect of {param} on metric")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"param_{param}_vs_metric.svg")
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path}")


# ─── Plot 4: Optimizer convergence ────────────────────────────────────────────


def plot_convergence(records: list[dict], out_path: str) -> None:
    all_ks = sorted({r["curvature"] for r in records})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for k in all_ks:
        group = sorted(
            [r for r in records if r["curvature"] == k], key=lambda r: r["trial"]
        )
        if not group:
            continue
        col = k_color(k, all_ks)
        trials = [r["trial"] for r in group]
        metrics = [r["metric_mean"] for r in group]
        best_so_far = [max(metrics[: i + 1]) for i in range(len(metrics))]

        axes[0].plot(trials, metrics, color=col, alpha=0.4, linewidth=0.8)
        axes[0].scatter(trials, metrics, color=col, s=5, alpha=0.5)
        axes[1].plot(trials, best_so_far, color=col, linewidth=1.8, label=f"k={k:+.1f}")

    axes[0].set_title("Metric per trial")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("metric_mean")

    axes[1].set_title("Best metric found so far")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Best metric_mean")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")

    fig.suptitle("Optimizer convergence by curvature", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")


# ─── Plot 5: Pairwise interaction grid ────────────────────────────────────────


def plot_pairwise_interactions(
    records: list[dict],
    top_params: list[str],
    out_path: str,
) -> None:
    """
    Grid of pairwise scatter plots for top_params.
    Upper triangle: scatter colored by metric_mean.
    Diagonal:       KDE / histogram of each parameter.
    Lower triangle: scatter colored by curvature.
    """
    all_ks = sorted({r["curvature"] for r in records})
    n = len(top_params)

    fig, axes = plt.subplots(n, n, figsize=(n * 2.3, n * 2.1))
    if n == 1:
        axes = np.array([[axes]])

    metrics = np.array([r["metric_mean"] for r in records])
    norm_metric = mcolors.Normalize(
        vmin=np.percentile(metrics, 5), vmax=np.percentile(metrics, 95)
    )
    cmap_metric = cm.plasma

    def get_col(k):
        return k_color(k, all_ks)

    for i, pi in enumerate(top_params):
        for j, pj in enumerate(top_params):
            ax = axes[i, j]

            xi = np.array([r.get(pi, np.nan) for r in records], dtype=float)
            xj = np.array([r.get(pj, np.nan) for r in records], dtype=float)
            m = np.array([r["metric_mean"] for r in records], dtype=float)
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
        sm, ax=axes, label="metric_mean (upper triangle)", shrink=0.6, pad=0.02
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


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity analysis for fitting-curvature"
    )
    parser.add_argument(
        "--input",
        default="results",
        help="JSONL file prefix, e.g. 'results' loads 'results_k0.0.jsonl' etc. (default: results)",
    )
    parser.add_argument(
        "--output",
        default="plots",
        help="Output directory for SVG files (default: plots/)",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=5,
        help="Number of top-importance parameters to include in pairwise plot (default: 5)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading results from '{args.input}_k*.jsonl' ...")
    records = load_results(args.input)
    if not records:
        print("No results found. Check --input prefix.")
        return
    print(
        f"Loaded {len(records)} trials across {len({r['curvature'] for r in records})} curvatures."
    )

    params = present_params(records)
    print(f"Parameters: {params}\n")

    correlations = compute_correlations(records, params)
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
    plot_convergence(records, os.path.join(args.output, "convergence.svg"))
    plot_param_vs_metric(records, params, args.output)

    print(f"\nTop {args.top_pairs} for pairwise plot: {top_params}")
    if not top_params:
        print("  (skipped pairwise — not enough data for correlations)")
        print(f"\nDone. All plots saved to '{args.output}'.")
        return
    plot_pairwise_interactions(
        records,
        top_params,
        os.path.join(args.output, f"pairwise_top{args.top_pairs}.svg"),
    )

    print(f"\nDone. All plots saved to '{args.output}/'.")


if __name__ == "__main__":
    main()
