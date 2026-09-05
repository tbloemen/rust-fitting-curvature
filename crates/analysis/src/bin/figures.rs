//! Thesis results figures (Experiments 2–5) from the qParEGO sweeps.
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
//! ```

use std::path::PathBuf;

use clap::Parser;

use fitting_analysis::figures::{self, exp2, exp3, exp4, exp5, r2_bars, save};
use fitting_analysis::Result;

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
    #[arg(long, num_args = 1.., default_values_t = [2usize, 3, 4, 5])]
    exp: Vec<usize>,

    /// Lowest κ in Exp 4's zoomed gap figure, which is written alongside the
    /// full one. Set to 0 to skip the zoom.
    #[arg(long, default_value_t = 0.01)]
    gap_zoom_kappa: f64,

    /// The R2 table written by `r2 aggregate --deltas`, plotted as one bar
    /// chart per (dataset, geometry) under `<out-dir>/experiment_4`. Absent is
    /// not an error — it is a separate `r2` run — and the charts are then
    /// simply not written.
    #[arg(long, default_value = "results/r2_delta.jsonl")]
    r2_delta: PathBuf,
}

/// A figure with no data behind it is skipped, not an error: the sweep grid is
/// not rectangular (no spherical `norm_only`, hyperbolic-only `rms_anchored`)
/// and Exp 3's scatter needs a κ_data export that is a separate run. What is
/// missing is visible in `out_dir` — the figure simply isn't there.
fn main() -> Result<()> {
    let args = Args::parse();

    let cells = figures::load_all_cells(&args.results_dir)?;

    if args.exp.contains(&2) {
        for n in &args.n {
            let fig = exp2::StackedFronts::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir)?;
            }
        }
    }

    if args.exp.contains(&3) {
        for n in &args.n {
            let fig = exp3::KappaScatter::new(&cells, &args.results_dir, *n)?;
            if fig.has_kappa_data() {
                save(&fig, &args.out_dir)?;
            }

            // One small SVG per dataset rather than one wide strip, so the
            // panels can be arranged freely in the report.
            for fig in exp3::RmsAnchored::panels(&cells, *n) {
                if fig.has_anchored() {
                    save(&fig, &args.out_dir)?;
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
            let fig = exp4::RhoManProj::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir)?;
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
            let fig = exp4::ProjGap::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir)?;
            }

            // A floor of 0 is "no zoom", not "zoom at zero": with it the zoom
            // would carry the full figure's name and overwrite it.
            if args.gap_zoom_kappa > 0.0 {
                let zoom = exp4::ProjGap::new(&cells, *n).zoomed(args.gap_zoom_kappa);
                if zoom.has_data() {
                    save(&zoom, &args.out_dir)?;
                }
            }
        }

        // The R2 table as bar charts, one per (dataset, geometry) per N. These
        // come from the stage-2 JSONL rather than from `cells`, so they are the
        // same numbers the thesis table carries, and they go in their own
        // subdirectory: 9 datasets x 3 geometries x 2 N is a lot of files to
        // leave loose among the other figures.
        let deltas = r2_bars::load_deltas(&args.r2_delta)?;
        let bars_dir = args.out_dir.join("experiment_4");
        for n in &args.n {
            for fig in r2_bars::R2Bars::panels(&deltas, *n) {
                save(&fig, &bars_dir)?;
            }
        }
    }

    if args.exp.contains(&5) {
        for n in &args.n {
            let fig = exp5::FrontGrid::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir)?;
            }

            let fig = exp5::Marginals::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir)?;
            }
        }
    }

    Ok(())
}
