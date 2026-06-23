//! Plot the Gromov δ-hyperbolicity against ball size `k` for the **real**
//! datasets, the same growing-ball procedure used by
//! `gromov_delta_curve.rs` for the synthetic fixtures (see that example
//! for the method description and references).
//!
//! Datasets:
//!   - `mnist` / `fashion_mnist` — 28×28 pixel vectors (Euclidean distances
//!      in pixel space).
//!   - `pbmc` — single-cell RNA-seq PCA features (Euclidean distances).
//!   - `wordnet_mammals` — the mammal subtree; distances are the graph
//!      shortest-path (BFS) matrix, already supplied by the loader.
//!
//! WordNet (a tree) should saturate at a small δ (hyperbolic); the
//! feature-space datasets are expected to keep growing (not δ-hyperbolic).
//!
//! Run with (data path defaults to `www/public/data`):
//!
//! ```bash
//! cargo run --release -p fitting-core --example gromov_delta_curve_real
//! cargo run --release -p fitting-core --example gromov_delta_curve_real -- path/to/data
//! ```
//!
//! Writes `plots/gromov_delta_curve_real.png` (normalised δ) and
//! `plots/gromov_delta_curve_real_raw.png` (raw δ), and prints the per-k
//! table plus the saturated δ / curvature estimate for each dataset.

use std::error::Error;

use fitting_core::curvature_detection::{
    gromov_delta_curve, GromovBallCurve, SATURATION_SLOPE_THRESHOLD,
};
use fitting_core::data::{load_fashion_mnist, load_mnist, load_pbmc, load_wordnet_mammals};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::synthetic_data::DataPoints;
use plotters::prelude::*;

const N: usize = 500;
const CURVE_SEED: u64 = 7;
const N_CENTERS: usize = 40;
const MAX_QUADS_PER_BALL: usize = 20_000;

/// Ball sizes that grow geometrically up to `n` (so the tail slope is read
/// over the largest balls the dataset can supply).
fn ball_sizes(n: usize) -> Vec<usize> {
    let mut sizes = vec![4, 6, 8, 10, 15, 20, 30, 40, 60, 80, 120, 160, 200, 280, 400];
    sizes.retain(|&k| k < n);
    sizes.push(n);
    sizes
}

struct Case {
    name: &'static str,
    color: RGBColor,
    curve: GromovBallCurve,
}

/// Ensure a loaded dataset has a pairwise distance matrix: graph datasets
/// (WordNet) arrive with one precomputed; feature datasets get Euclidean
/// distances in feature space.
fn distances_for(data: &DataPoints) -> Vec<f64> {
    if !data.distances.is_empty() {
        data.distances.clone()
    } else {
        compute_euclidean_distance_matrix(&data.x, data.n_points, data.ambient_dim)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Data directory: first CLI arg, else the repo's www/public/data.
    let data_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "www/public/data".to_string());

    let fixtures: Vec<(&'static str, RGBColor, DataPoints)> = vec![
        ("mnist", RED, load_mnist(&format!("{data_root}/mnist"), N)?),
        (
            "fashion_mnist",
            RGBColor(255, 140, 0), // orange
            load_fashion_mnist(&format!("{data_root}/fashion-mnist"), N)?,
        ),
        ("pbmc", GREEN, load_pbmc(&format!("{data_root}/pbmc"), N)?),
        (
            "wordnet_mammals",
            BLUE,
            load_wordnet_mammals(&format!("{data_root}/wordnet"), N)?,
        ),
    ];

    let mut cases: Vec<Case> = Vec::new();
    for (name, color, data) in &fixtures {
        let n = data.n_points;
        let sizes = ball_sizes(n);
        let distances = distances_for(data);
        let curve = gromov_delta_curve(
            &distances,
            n,
            &sizes,
            N_CENTERS,
            MAX_QUADS_PER_BALL,
            CURVE_SEED,
        );
        cases.push(Case {
            name,
            color: *color,
            curve,
        });
    }

    // ── Console report ──────────────────────────────────────────────
    for case in &cases {
        println!(
            "\n── {} (median d = {:.3}) ──",
            case.name, case.curve.median_distance
        );
        println!("     k     δ(k) raw     δ(k) norm");
        for p in &case.curve.points {
            println!(
                "  {:5}   {:10.4}   {:10.4}",
                p.k, p.delta_mean, p.delta_mean_normalised
            );
        }
        let slope = case.curve.tail_loglog_slope();
        let is_hyperbolic = slope < SATURATION_SLOPE_THRESHOLD;
        println!(
            "  saturated δ = {:.4} (norm {:.4}),  tail log-log slope = {:.4}",
            case.curve.saturated_delta(),
            case.curve.saturated_delta_normalised(),
            slope,
        );
        println!(
            "  verdict: {} (slope {} {:.2})  →  K ≈ {:.4}",
            if is_hyperbolic {
                "HYPERBOLIC (δ saturates)"
            } else {
                "not hyperbolic (δ keeps growing)"
            },
            if is_hyperbolic { "<" } else { "≥" },
            SATURATION_SLOPE_THRESHOLD,
            if is_hyperbolic {
                case.curve.estimated_hyperbolic_curvature()
            } else {
                0.0
            },
        );
    }

    // ── Plots ───────────────────────────────────────────────────────
    plot(
        "plots/gromov_delta_curve_real.png",
        "Gromov δ vs ball size k — real datasets (normalised by median distance)",
        "normalised δ(k)",
        &cases,
        |p| p.delta_mean_normalised,
    )?;
    plot(
        "plots/gromov_delta_curve_real_raw.png",
        "Gromov δ vs ball size k — real datasets (raw)",
        "δ(k)",
        &cases,
        |p| p.delta_mean,
    )?;

    println!("\nWrote plots/gromov_delta_curve_real.png and plots/gromov_delta_curve_real_raw.png");
    Ok(())
}

fn plot(
    path: &str,
    caption: &str,
    y_desc: &str,
    cases: &[Case],
    value: impl Fn(&fitting_core::curvature_detection::GromovBallPoint) -> f64 + Copy,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (960, 640)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = cases
        .iter()
        .flat_map(|c| c.curve.points.iter())
        .map(|p| p.k)
        .max()
        .unwrap_or(1) as f64;
    let y_max = cases
        .iter()
        .flat_map(|c| c.curve.points.iter())
        .map(value)
        .fold(0.0_f64, f64::max)
        * 1.1
        + 1e-6;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..x_max, 0f64..y_max)?;

    chart
        .configure_mesh()
        .x_desc("ball size k")
        .y_desc(y_desc)
        .axis_desc_style(("sans-serif", 18))
        .draw()?;

    for case in cases {
        let series: Vec<(f64, f64)> = case
            .curve
            .points
            .iter()
            .map(|p| (p.k as f64, value(p)))
            .collect();
        chart
            .draw_series(LineSeries::new(series.clone(), case.color.stroke_width(2)))?
            .label(case.name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], case.color));
        chart.draw_series(
            series
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 3, case.color.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 18))
        .draw()?;

    root.present()?;
    Ok(())
}
