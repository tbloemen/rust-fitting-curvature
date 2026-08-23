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

use fitting_analysis::figures::{self, exp2, exp3, exp4, exp5, save};
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

            let fig = exp3::RmsAnchored::new(&cells, *n);
            if fig.has_anchored() {
                save(&fig, &args.out_dir)?;
            }
        }
    }

    if args.exp.contains(&4) {
        let fig = exp4::RhoManProj::new(&cells, &args.n);
        if fig.has_data() {
            save(&fig, &args.out_dir)?;
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
