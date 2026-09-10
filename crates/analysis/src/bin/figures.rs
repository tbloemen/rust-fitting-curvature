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

use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;

use fitting_analysis::figures::{self, exp1, exp2, r2_bars, save};
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

    /// The Experiment 1 table written by the `exp1` binary, plotted as the
    /// matched-minus-mismatched gain chart. Absent is not an error — it is a
    /// separate `exp1` run — and the chart is then simply not written.
    #[arg(long)]
    exp1: Option<PathBuf>,

    /// Preference regions to draw Experiment 1's gain chart for, one figure
    /// each. The same choice `scripts/exp1_r2_typst.py --region` makes for the
    /// table, and the same default.
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
                    return Err(Error::MixedObjectiveSpaces {
                        first: path.display().to_string(),
                        first_space: ObjectiveSpace::from_str(fig.space())
                            .map_or("unknown", ObjectiveSpace::tag),
                        second: args.results_dir.display().to_string(),
                        second_space: space.tag(),
                    });
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

    // Exp 4 is the one figure that is *not* drawn once per N: ρ is a within-cell
    // rank correlation, so two sample sizes are two populations rather than a
    // trend, and overlaying them only doubled the marks. The largest N asked for
    // is the one plotted — `--n 1000` alone still gets a figure.
    if args.exp.contains(&4) {
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

    Ok(())
}
