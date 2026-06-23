//! Plot the Gromov δ-hyperbolicity against ball size `k`, following the
//! growing-ball procedure of the NeTS proposal (Krioukov, Boguñá,
//! Claffy; "δ-hyperbolic spaces", Task 2):
//!
//!   For each ball size `k`, sample random centres, take each centre's
//!   `k` nearest neighbours as a ball, measure the ball's δ as the
//!   supremum of the 4-point condition `S_max − S_mid` over its
//!   quadruples, and average over centres to get δ(k).  As `k` grows,
//!   δ(k) saturates at the δ of the underlying space.
//!
//! Hyperbolic data saturates at a small δ; Euclidean data keeps growing
//! (ℝⁿ is not δ-hyperbolic); spherical data sits in between.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p fitting-core --example gromov_delta_curve
//! ```
//!
//! Writes `gromov_delta_curve.png` (normalised δ) and
//! `gromov_delta_curve_raw.png` (raw δ) to the current directory, and
//! prints the per-k table plus the saturated δ / curvature estimate.

use std::error::Error;

use fitting_core::curvature_detection::{
    gromov_delta_curve, GromovBallCurve, SATURATION_SLOPE_THRESHOLD,
};
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere, DataPoints,
};
use plotters::prelude::*;

const N: usize = 400;
const SEED: u64 = 42;
const CURVE_SEED: u64 = 7;
const N_CENTERS: usize = 40;
const MAX_QUADS_PER_BALL: usize = 20_000;

fn ball_sizes() -> Vec<usize> {
    vec![4, 6, 8, 10, 15, 20, 30, 40, 60, 80, 120, 160, 200, 280, 400]
}

struct Case {
    name: &'static str,
    color: RGBColor,
    curve: GromovBallCurve,
}

fn main() -> Result<(), Box<dyn Error>> {
    let sizes = ball_sizes();

    let fixtures: Vec<(&'static str, RGBColor, DataPoints)> = vec![
        ("hyperbolic", RED, generate_uniform_hyperbolic(N, SEED, 5.0)),
        ("euclidean", BLUE, generate_uniform_ball_2d(N, SEED, 5.0)),
        ("spherical", GREEN, generate_uniform_sphere(N, SEED)),
    ];

    let mut cases: Vec<Case> = Vec::new();
    for (name, color, data) in fixtures {
        let curve = gromov_delta_curve(
            &data.distances,
            N,
            &sizes,
            N_CENTERS,
            MAX_QUADS_PER_BALL,
            CURVE_SEED,
        );
        let _ = data;
        cases.push(Case { name, color, curve });
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
        "plots/gromov_delta_curve.png",
        "Gromov δ vs ball size k (normalised by median distance)",
        "normalised δ(k)",
        &cases,
        |p| p.delta_mean_normalised,
    )?;
    plot(
        "plots/gromov_delta_curve_raw.png",
        "Gromov δ vs ball size k (raw)",
        "δ(k)",
        &cases,
        |p| p.delta_mean,
    )?;

    println!("\nWrote gromov_delta_curve.png and gromov_delta_curve_raw.png");
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
        .caption(caption, ("sans-serif", 26))
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
