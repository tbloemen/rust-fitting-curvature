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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading result cells...");
    let cells = figures::load_all_cells(&args.results_dir)?;
    println!("  {} cells loaded", cells.len());

    if args.exp.contains(&2) {
        println!("Experiment 2 — stacked Pareto fronts");
        for n in &args.n {
            let fig = exp2::StackedFronts::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir)?;
            } else {
                println!("  exp2 N={n}: no data, skipped");
            }
        }
    }

    if args.exp.contains(&3) {
        println!("Experiment 3 — κ vs κ_data");
        for n in &args.n {
            let fig = exp3::KappaScatter::new(&cells, &args.results_dir, *n);
            if !fig.has_kappa_data() {
                println!("  exp3 N={n}: no kappa_data file, skipped scatter");
            } else {
                save(&fig, &args.out_dir)?;
            }

            let fig = exp3::RmsAnchored::new(&cells, *n);
            if fig.has_anchored() {
                save(&fig, &args.out_dir)?;
            } else {
                println!(
                    "  exp3 N={n}: no rms_anchored runs present — overlay figure skipped \
                     (flagged in the discrepancy report)"
                );
            }
        }
    }

    if args.exp.contains(&4) {
        println!("Experiment 4 — ρ_man-proj(κ)");
        let fig = exp4::RhoManProj::new(&cells, &args.n);
        if fig.has_data() {
            save(&fig, &args.out_dir)?;
        } else {
            println!("  exp4: no data, skipped");
        }
    }

    if args.exp.contains(&5) {
        println!("Experiment 5 — front grid + marginals");
        for n in &args.n {
            let fig = exp5::FrontGrid::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir)?;
            } else {
                println!("  exp5 grid N={n}: no data, skipped");
            }

            let fig = exp5::Marginals::new(&cells, *n);
            if fig.has_data() {
                save(&fig, &args.out_dir)?;
            } else {
                println!("  exp5 marginals N={n}: no data, skipped");
            }
        }
    }

    println!("Done.");
    Ok(())
}
