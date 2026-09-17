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
    self, exp1, exp2, exp2_dependence, exp2_dumbbell, exp2_region_gain, exp3, exp4,
    exp4_epsilon_dots, exp4_gain_dots, exp4_tradeoff, save,
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
    /// each. The same choice `scripts/exp1_r2_typst.py --region` makes for the
    /// table, and the same default. The ε chart takes no region — carrying no
    /// preference model is the point of it — so it is drawn once per N.
    #[arg(long, num_args = 1.., default_values_t = ["all".to_string()])]
    exp1_region: Vec<String>,

    /// Force the objective space instead of reading it off the sweeps. The
    /// sweeps under `results/` were searched in the 10-objective space
    /// (projected + manifold) and the re-runs in the current 6-objective one;
    /// every figure's filename carries the tag, so the two sets never collide.
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
    /// defaults to the untagged `results/r2_epsilon.jsonl`, so an `obj10`
    /// table has to be named here. The rows carry no `space` field, so a
    /// table from the other space cannot be detected. Absent is not an
    /// error; the figure is then simply not written.
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

    // Exp 1 is the one figure built from the stage-2 table rather than from
    // `cells`: it plots the numbers `@tab:geometry-match-r2` carries, so the
    // two cannot disagree. A region the table does not carry, like a missing
    // table, leaves the figure unwritten rather than failing.
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
        for n in &args.n {
            let fig = exp4::StackedFronts::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir, space)?;
            }
        }

        // The R2 table as bar charts, one per (dataset, geometry) per N. These
        // come from the stage-2 JSONL rather than from `cells`, so they are the
        // same numbers the thesis table carries, and they go in their own
        // subdirectory: 9 datasets x 3 geometries per N is a lot of files to
        // leave loose among the other figures.
        let r2_delta = args
            .r2_delta
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("results/r2_delta_{}.jsonl", space.tag())));
        let deltas = exp4::load_deltas(&r2_delta)?;
        let bars_dir = args.out_dir.join("experiment_4");
        for n in &args.n {
            for fig in exp4::R2Bars::panels(&deltas, *n, space) {
                save(&fig, &bars_dir, space)?;
            }
            // The same table as one figure per N: every (dataset, geometry,
            // setting) gain on one axis, in the same directory as the bars —
            // drawn twice, log and `_linear`, as the Exp 2 panels are.
            for scale in exp4_gain_dots::Scale::ALL {
                let fig = exp4_gain_dots::GainDots::new(&deltas, *n, scale);
                if fig.has_data() {
                    save(&fig, &bars_dir, space)?;
                }
                // The local-versus-global trade of the same table: the two
                // family regions on orthogonal axes, same scales, same dir.
                let fig = exp4_tradeoff::TradeoffScatter::new(&deltas, *n, scale);
                if fig.has_data() {
                    save(&fig, &bars_dir, space)?;
                }
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
                save(&fig, &bars_dir, space)?;
            }
        }
    }

    Ok(())
}

/// Exp 2 draws one metric-trend panel per curved geometry and per x axis —
/// κ, the embedding curvature, and |K|, the searched hyperparameter;
/// `panels` returns only the ones with data, so there is no guard on the
/// loop. Their shared legend is a separate file, built from all of the
/// panels so it names exactly the curves they draw. The metric-dependence
/// heatmap is one panel per geometry from the same cells, with its ρ
/// colourbar written once per run. Both read the **Pareto front** of each
/// cell, reduced here once in the scoring space: the thesis compares
/// corpora, and a corpus is the front, so a trial the search discarded is
/// not part of what either figure describes. The per-region gain heatmap is the one
/// Exp 2 figure built from a table rather than from `cells` — the stage-1
/// R2 table, so it carries the numbers the thesis tables carry — one panel
/// per curved geometry and a colourbar per N, since its scale is the
/// data's. Everything lands under `<out-dir>/experiment_2`, in one
/// subdirectory per figure family — `metric_trend` (the overlays and their
/// legend, set together as one row), `metric_spread`, `unbounded`,
/// `dependence`, `region_gain` — because the two per-metric figures alone
/// write over a hundred files per space.
fn render_exp2(args: &Args, cells: &figures::CellMap, space: ObjectiveSpace) -> Result<()> {
    // One subdirectory per figure family. Exp 2 writes over two hundred files
    // per space — every (metric, geometry, axis, scale) of two per-metric
    // figures on top of the overlays and heatmaps — and a flat directory
    // stops being a directory a person can read. The families are the ones
    // the module docs name, and the family a panel belongs to is decided
    // here, not by parsing its filename.
    let exp2_dir = args.out_dir.join("experiment_2");
    let trend_dir = exp2_dir.join("metric_trend");
    let spread_dir = exp2_dir.join("metric_spread");
    let unbounded_dir = exp2_dir.join("unbounded");
    let dependence_dir = exp2_dir.join("dependence");
    let region_gain_dir = exp2_dir.join("region_gain");

    let r2_local = args
        .r2_local
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("results/r2_local_{}.jsonl", space.tag())));
    let local_rows = exp4::load_table(&r2_local)?;
    let fronts = figures::front_cells(cells, space);
    let mut any_dependence = false;
    for n in &args.n {
        let mut panels = Vec::new();
        for x in [exp2::XAxis::Kappa, exp2::XAxis::Curvature] {
            panels.extend(exp2::MetricTrend::panels(&fronts, *n, x));
            // One panel per metric, the same curve with its spread around
            // it: the thesis shows one and the appendix carries the rest,
            // so every metric's is written.
            for fig in exp2::MetricSpread::panels(&fronts, *n, x) {
                save(&fig, &spread_dir, space)?;
            }
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

        let dependence = exp2_dependence::MetricDependence::panels(&fronts, *n);
        for fig in &dependence {
            save(fig, &dependence_dir, space)?;
            any_dependence = true;
        }
        // The same ρ once more, one row per pair with the geometries side by
        // side — the rendering the thesis compares across curvature on.
        if let Some(fig) = exp2_dumbbell::DependenceDumbbell::from_panels(&dependence, *n) {
            save(&fig, &dependence_dir, space)?;
        }

        // Twice: over every region, and over the projected surface's
        // regions alone (legacy space only — `panels` is empty otherwise).
        for columns in [
            exp2_region_gain::Columns::Full,
            exp2_region_gain::Columns::Projected,
        ] {
            let gains = exp2_region_gain::RegionGain::panels(&local_rows, *n, space, columns);
            for fig in &gains {
                save(fig, &region_gain_dir, space)?;
            }
            if let Some(bar) = exp2_region_gain::RegionGainColorbar::from_panels(&gains, *n) {
                save(&bar, &region_gain_dir, space)?;
            }
        }
    }
    if any_dependence {
        save(&exp2_dependence::DependenceColorbar, &dependence_dir, space)?;
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
