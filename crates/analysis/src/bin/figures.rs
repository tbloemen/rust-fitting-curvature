//! Thesis results figures (Experiments 1–4) from the qParEGO sweeps.
//!
//! Rust port of `analyze_experiments.py`. Local-only: this is the one part of
//! the analysis that needs plotters (and therefore a system font stack), which
//! is why it sits behind the crate's `plots` feature and never reaches the
//! cluster.
//!
//! ```bash
//! cargo run --release -p fitting-analysis --features plots --bin figures
//! cargo run --release -p fitting-analysis --features plots --bin figures -- --n 1000 5000
//! cargo run --release -p fitting-analysis --features plots --bin figures -- --exp 4
//! cargo run --release -p fitting-analysis --features plots --bin figures -- \
//!     --exp 1 --exp1-region all structure
//! ```

use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Parser;

use fitting_analysis::figures::{
    self, exp1, exp2, exp2_dependence, exp2_dumbbell, exp2_kappa_hist, exp2_proj_gap,
    exp2_region_gain, exp3, exp4, exp4_epsilon_dots, exp4_gain_dots, exp4_tradeoff, save,
};
use fitting_analysis::indicators::EpsilonRow;
use fitting_analysis::objectives::ObjectiveSpace;
use fitting_analysis::{Error, Result};

#[derive(Parser, Debug)]
#[command(
    name = "figures",
    about = "Thesis result figures from the qParEGO sweeps"
)]
struct Args {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Where the SVGs are written.
    #[arg(long, default_value = "plots")]
    out_dir: PathBuf,

    /// Sample sizes to plot. N=5000 is what the thesis reports; N=1000 is
    /// still available on request.
    #[arg(long, num_args = 1.., default_values_t = [5000usize])]
    n: Vec<usize>,

    /// Which experiments to render, numbered as the results chapter numbers
    /// its research questions. Out of range is an error rather than a silent
    /// no-op, which is what an unrecognised number used to be.
    #[arg(
        long,
        num_args = 1..,
        value_parser = clap::value_parser!(u8).range(1..=4),
        default_values_t = [1u8, 2, 3, 4],
    )]
    exp: Vec<u8>,

    /// The Experiment 1 table written by the `exp1` binary, plotted as two
    /// charts: the matched-minus-mismatched R2 gain, and the ε-indicator
    /// between the same fronts. Absent is not an error — it is a separate
    /// `exp1` run — and the charts are then simply not written. A table
    /// predating the `epsilon` block still draws the gain chart.
    #[arg(long)]
    exp1: Option<PathBuf>,

    /// Preference regions to draw Experiment 1's gain chart for, one figure
    /// each. The ε chart takes no region — carrying no
    /// preference model is the point of it — so it is drawn once per N.
    #[arg(long, num_args = 1.., default_values_t = ["all".to_string()])]
    exp1_region: Vec<String>,

    /// Lowest κ in Exp 2's zoomed projection-gap figure, which is written
    /// alongside the full one. Set to 0 to skip the zoom.
    #[arg(long, default_value_t = 0.01)]
    gap_zoom_kappa: f64,

    /// Force the objective space instead of checking it on the sweeps.
    #[arg(long)]
    objectives: Option<ObjectiveSpace>,

    /// The R2 table written by `r2 aggregate --deltas`, plotted as one bar
    /// chart per (dataset, geometry) under `<out-dir>/experiment_4`. Absent is
    /// not an error — it is a separate `r2` run — and the charts are then
    /// simply not written.
    #[arg(long)]
    r2_delta: Option<PathBuf>,

    /// The ε table written by `r2 compare`, plotted as the per-setting
    /// ε-indicator dot plot under `<out-dir>/experiment_4`. Defaults to
    /// `results/r2_epsilon_<space>.jsonl`; note `r2 compare --out` itself
    /// defaults to the untagged `results/r2_epsilon.jsonl`, so pass the path
    /// it was written to. Absent is not an error; the figure is then simply
    /// not written.
    #[arg(long)]
    r2_epsilon: Option<PathBuf>,

    /// The stage-1 table written by `r2 stats`, plotted as Experiment 2's
    /// per-region gain heatmaps under `<out-dir>/experiment_2/region_gain`.
    /// Absent is not
    /// an error — it is a separate `r2` run — and the heatmaps are then simply
    /// not written.
    #[arg(long)]
    r2_local: Option<PathBuf>,
}

/// A figure with no data behind it is skipped, not an error: the sweep grid is
/// not rectangular (no spherical `norm_only`, hyperbolic-only `rms_anchored`),
/// and a skeleton figure reports no data at all. What is missing is visible in
/// `out_dir` — the figure simply isn't there.
fn main() -> Result<()> {
    let args = Args::parse();

    let (cells, space) = figures::load_all_cells(&args.results_dir, args.objectives)?;

    // Exp 1 is built from the `exp1` table rather than from `cells`. A region
    // the table does not carry, like a missing table, leaves the figure
    // unwritten rather than failing.
    if args.exp.contains(&1) {
        let path = args.exp1.clone().unwrap_or_else(|| {
            PathBuf::from(format!("results/exp1_geometry_match_{}.jsonl", space.tag()))
        });
        let rows = exp1::load_rows(&path)?;
        for n in &args.n {
            for region in &args.exp1_region {
                let fig = exp1::MatchedGain::new(&rows, *n, region);
                if !fig.has_data() {
                    continue;
                }
                // The filename is the only place the space is recorded — the
                // figure itself draws nothing identifying — so a table from the
                // other space would be silently mislabelled.
                if fig.space() != space.tag() {
                    return Err(mixed_spaces(fig.space(), &path, &args.results_dir, space));
                }
                save(&fig, &args.out_dir, space)?;
            }

            // The ε companion, once per N rather than once per region: the
            // indicator carries no preference model, which is the whole reason
            // it is reported beside the R2 gain.
            let fig = exp1::MatchedEpsilon::new(&rows, *n);
            if fig.has_data() {
                if fig.space() != space.tag() {
                    return Err(mixed_spaces(fig.space(), &path, &args.results_dir, space));
                }
                save(&fig, &args.out_dir, space)?;
            }
        }
    }

    if args.exp.contains(&2) {
        render_exp2(&args, &cells, space)?;
    }

    // Exp 3: κ against |K| over the pooled Pareto fronts, one panel per curved
    // geometry; `panels` returns only the ones with data. Under its own
    // directory, as Exp 2's panels are.
    if args.exp.contains(&3) {
        let exp3_dir = args.out_dir.join("experiment_3");
        for n in &args.n {
            for fig in exp3::KappaLanding::panels(&cells, *n, space) {
                save(&fig, &exp3_dir, space)?;
            }
        }
    }

    if args.exp.contains(&4) {
        // The stage-2 ΔR2 table: every (dataset, geometry, setting) gain on one
        // axis per N, and the local-versus-global trade of the same table.
        let r2_delta = args
            .r2_delta
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("results/r2_delta_{}.jsonl", space.tag())));
        let deltas = exp4::load_deltas(&r2_delta)?;
        let exp4_dir = args.out_dir.join("experiment_4");
        for n in &args.n {
            let fig = exp4_gain_dots::GainDots::new(&deltas, *n);
            if fig.has_data() {
                save(&fig, &exp4_dir, space)?;
            }
            let fig = exp4_tradeoff::TradeoffScatter::new(&deltas, *n);
            if fig.has_data() {
                save(&fig, &exp4_dir, space)?;
            }
        }

        // The ε-indicator in the same layout, from `r2 compare`'s table.
        let r2_epsilon = args
            .r2_epsilon
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("results/r2_epsilon_{}.jsonl", space.tag())));
        let epsilons: Vec<EpsilonRow> = exp4::load_table(&r2_epsilon)?;
        for n in &args.n {
            let fig = exp4_epsilon_dots::EpsilonDots::new(&epsilons, *n);
            if fig.has_data() {
                save(&fig, &exp4_dir, space)?;
            }
        }
    }

    Ok(())
}

/// Exp 2 draws one metric-trend panel per curved geometry and per x axis —
/// κ, the embedding curvature, and |K|, the searched hyperparameter — with
/// their shared legend as a separate file, one panel per unbounded metric,
/// the metric-dependence dumbbell, the κ histograms and the projection gap.
/// All read the **Pareto front** of each cell, reduced here once in the
/// scoring space. The per-region gain heatmap is built from the stage-1 R2
/// table instead, one panel per curved geometry and a colourbar per N.
/// Everything lands under `<out-dir>/experiment_2`, in one subdirectory per
/// figure family: `metric_trend`, `unbounded`, `dependence`, `region_gain`.
fn render_exp2(args: &Args, cells: &figures::CellMap, space: ObjectiveSpace) -> Result<()> {
    // One subdirectory per figure family; the family a panel belongs to is
    // decided here, not by parsing its filename.
    let exp2_dir = args.out_dir.join("experiment_2");
    let trend_dir = exp2_dir.join("metric_trend");
    let unbounded_dir = exp2_dir.join("unbounded");
    let dependence_dir = exp2_dir.join("dependence");
    let region_gain_dir = exp2_dir.join("region_gain");

    // Where the `all_off` corpora land in κ, one panel per curved geometry.
    // κ needs `r_gyration`, so from `results/` this writes nothing.
    let fronts = figures::front_cells(cells, space);
    for n in &args.n {
        for fig in exp2_kappa_hist::KappaHistogram::panels(&fronts, *n) {
            save(&fig, &exp2_dir, space)?;
        }
    }

    // The manifold-vs-projection gap, once per N: its unit is a front point,
    // not a cell, so each N stands on its own. Drawn above `--gap-zoom-kappa`:
    // over the full range the x axis is dominated by the collapsed-embedding
    // spike at κ ≈ 2e-7, three decades left of anything else. A floor of 0
    // draws the full range.
    for n in &args.n {
        let fig = exp2_proj_gap::ProjGap::new(cells, *n, space);
        let fig = if args.gap_zoom_kappa > 0.0 {
            fig.zoomed(args.gap_zoom_kappa)
        } else {
            fig
        };
        if fig.has_data() {
            save(&fig, &exp2_dir, space)?;
        }
    }

    let r2_local = args
        .r2_local
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("results/r2_local_{}.jsonl", space.tag())));
    let local_rows = exp4::load_table(&r2_local)?;
    for n in &args.n {
        let mut panels = Vec::new();
        for x in [exp2::XAxis::Kappa, exp2::XAxis::Curvature] {
            panels.extend(exp2::MetricTrend::panels(&fronts, *n, x));
            // The unbounded metrics: one panel each, no legend.
            for fig in exp2::UnboundedTrend::panels(&fronts, *n, x) {
                save(&fig, &unbounded_dir, space)?;
            }
        }
        // The overlays and their legend go together: they are set as one row.
        for fig in &panels {
            save(fig, &trend_dir, space)?;
        }
        let legend = exp2::MetricLegend::from_panels(&panels, *n);
        if legend.has_data() {
            save(&legend, &trend_dir, space)?;
        }

        // The metric dependence: one row per metric pair, with the geometries
        // side by side.
        let dependence = exp2_dependence::MetricDependence::panels(&fronts, *n);
        if let Some(fig) = exp2_dumbbell::DependenceDumbbell::from_panels(&dependence, *n) {
            save(&fig, &dependence_dir, space)?;
        }

        let gains = exp2_region_gain::RegionGain::panels(&local_rows, *n, space);
        for fig in &gains {
            save(fig, &region_gain_dir, space)?;
        }
        if let Some(bar) = exp2_region_gain::RegionGainColorbar::from_panels(&gains, *n) {
            save(&bar, &region_gain_dir, space)?;
        }
    }
    Ok(())
}

/// The Exp 1 table was scored in a different objective space than the sweeps.
///
/// Both Exp 1 figures draw nothing identifying, so the filename is the only
/// record of the space and a table from the other one would be silently
/// mislabelled — the two are not comparable. *`table`* is the `--exp1` path,
/// *`results_dir`* the sweeps whose space was resolved.
fn mixed_spaces(
    table_space: &str,
    table: &Path,
    results_dir: &Path,
    space: ObjectiveSpace,
) -> Error {
    Error::MixedObjectiveSpaces {
        first: table.display().to_string(),
        first_space: ObjectiveSpace::from_str(table_space).map_or("unknown", ObjectiveSpace::tag),
        second: results_dir.display().to_string(),
        second_space: space.tag(),
    }
}
