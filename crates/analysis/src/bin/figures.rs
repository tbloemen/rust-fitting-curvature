//! Thesis results figures (Experiments 1–5) from the qParEGO sweeps.
//!
//! Rust port of `analyze_experiments.py`. Local-only: this is the one part of
//! the analysis that needs plotters (and therefore a system font stack), which
//! is why it sits behind the crate's `plots` feature and never reaches the
//! cluster.
//!
//! ```bash
//! cargo run --release -p fitting-analysis --features plots --bin figures
//! cargo run --release -p fitting-analysis --features plots --bin figures -- --n 1000
//! cargo run --release -p fitting-analysis --features plots --bin figures -- --exp 4
//! cargo run --release -p fitting-analysis --features plots --bin figures -- \
//!     --exp 1 --exp1-region all structure
//! ```

use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Parser;

use fitting_analysis::figures::{self, exp1, exp2, exp3, exp4, exp5, r2_bars, save};
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

    /// Where the SVG + PNG pairs are written.
    #[arg(long, default_value = "plots")]
    out_dir: PathBuf,

    /// Sample sizes to plot.
    #[arg(long, num_args = 1.., default_values_t = [1000usize, 5000])]
    n: Vec<usize>,

    /// Which experiments to render.
    #[arg(long, num_args = 1.., default_values_t = [1usize, 2, 3, 4, 5])]
    exp: Vec<usize>,

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

    /// Lowest κ in Exp 4's zoomed gap figure, which is written alongside the
    /// full one. Set to 0 to skip the zoom.
    #[arg(long, default_value_t = 0.01)]
    gap_zoom_kappa: f64,

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
}

/// A figure with no data behind it is skipped, not an error: the sweep grid is
/// not rectangular (no spherical `norm_only`, hyperbolic-only `rms_anchored`)
/// and Exp 3's scatter needs a `κ_data` export that is a separate run. What is
/// missing is visible in `out_dir` — the figure simply isn't there.
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
        for n in &args.n {
            let fig = exp2::StackedFronts::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir, space)?;
            }
        }
    }

    if args.exp.contains(&3) {
        for n in &args.n {
            let fig = exp3::KappaScatter::new(&cells, &args.results_dir, *n, space)?;
            if fig.has_kappa_data() {
                save(&fig, &args.out_dir, space)?;
            }

            // One small SVG per dataset rather than one wide strip, so the
            // panels can be arranged freely in the report.
            for fig in exp3::RmsAnchored::panels(&cells, *n, space) {
                if fig.has_anchored() {
                    save(&fig, &args.out_dir, space)?;
                }
            }
        }
    }

    // Exp 4 is the one figure that is *not* drawn once per N: ρ is a within-cell
    // rank correlation, so two sample sizes are two populations rather than a
    // trend, and overlaying them only doubled the marks. The largest N asked for
    // is the one plotted — `--n 1000` alone still gets a figure.
    if args.exp.contains(&4) {
        if let Some(n) = args.n.iter().max() {
            // ρ is a within-cell rank correlation over every trial, not over a
            // front, so it does not depend on the objective space.
            let fig = exp4::RhoManProj::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir, space)?;
            }
        }

        // The gap figure *is* drawn once per N: its unit is a front point, not
        // a cell, so the two sample sizes are two independent estimates of the
        // same κ trend and each one stands on its own.
        //
        // Each also gets a zoom above `--gap-zoom-kappa`, as its own file. The
        // full figure's x axis is dominated by the collapsed-embedding spike at
        // κ ≈ 2e-7, three decades left of anything else; the zoom is where the
        // curved half of the sweep actually lives. Both are kept — the zoom
        // excludes real front points, and the reader should be able to see what.
        for n in &args.n {
            let fig = exp4::ProjGap::new(&cells, *n, space);
            if fig.has_data() {
                save(&fig, &args.out_dir, space)?;
            }

            // A floor of 0 is "no zoom", not "zoom at zero": with it the zoom
            // would carry the full figure's name and overwrite it.
            if args.gap_zoom_kappa > 0.0 {
                let zoom = exp4::ProjGap::new(&cells, *n, space).zoomed(args.gap_zoom_kappa);
                if zoom.has_data() {
                    save(&zoom, &args.out_dir, space)?;
                }
            }
        }

        // The R2 table as bar charts, one per (dataset, geometry) per N. These
        // come from the stage-2 JSONL rather than from `cells`, so they are the
        // same numbers the thesis table carries, and they go in their own
        // subdirectory: 9 datasets x 3 geometries x 2 N is a lot of files to
        // leave loose among the other figures.
        let r2_delta = args
            .r2_delta
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("results/r2_delta_{}.jsonl", space.tag())));
        let deltas = r2_bars::load_deltas(&r2_delta)?;
        let bars_dir = args.out_dir.join("experiment_4");
        for n in &args.n {
            for fig in r2_bars::R2Bars::panels(&deltas, *n, space) {
                save(&fig, &bars_dir, space)?;
            }
        }
    }

    if args.exp.contains(&5) {
        for n in &args.n {
            let fig = exp5::FrontGrid::new(&cells, *n, space);
            if fig.has_data() {
                save(&fig, &args.out_dir, space)?;
            }

            let fig = exp5::Marginals::new(&cells, *n, space);
            if fig.has_data() {
                save(&fig, &args.out_dir, space)?;
            }
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
