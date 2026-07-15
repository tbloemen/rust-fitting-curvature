//! Plot the Wilson signature residual against the radius of curvature
//! `r` for the **real** datasets — the same normalised/unnormalised
//! residual sweep used by `wilson_residual_curve.rs` for the synthetic
//! fixtures (see that example for the method description and the theory).
//!
//! Datasets:
//!   - `mnist` / `fashion_mnist` — 28×28 pixel vectors (Euclidean distances in pixel space).
//!   - `pbmc` — single-cell RNA-seq PCA features (Euclidean distances).
//!   - `wordnet_mammals` — the mammal subtree; distances are the graph shortest-path (BFS) matrix, already supplied by the loader.
//!
//! For each dataset we sweep `r` and record, for both the spherical
//! (`r² cos(d/r)`) and hyperbolic (`−r² cosh(d/r)`) Gram models:
//!   - the normalised residual `ρ(r) = E(r)/‖Z‖_F² ∈ [0,1]` the fitter minimises, and
//!   - the raw energy `E(r) = ρ·‖Z‖_F²` (scaled by `d_max⁴` so datasets of different diameter overlay).
//!
//! Vertical black lines mark the search windows each fit actually uses (spherical `[d_max/π, d_max/SPHERICAL_ANGULAR_MIN]`, hyperbolic `[d_max/20, d_max]`);
//! a bold cross marks the fitted `r*`. The x-axis is `log₁₀(r/d_max)` so all datasets share a scale.
//!
//! Run with (data path defaults to `www/public/data`):
//!
//! ```bash
//! cargo run --release -p fitting-core --features plot-examples \
//!     --example wilson_residual_curve_real
//! cargo run --release -p fitting-core --features plot-examples \
//!     --example wilson_residual_curve_real -- path/to/data
//! ```
//!
//! Writes four PNGs to `plots/` (normalised + raw, spherical + hyperbolic).

use std::error::Error;

use fitting_core::curvature_detection::{
    fit_hyperbolic, fit_spherical, hyperbolic_residual_at, spherical_residual_at, WilsonFit,
    SPHERICAL_ANGULAR_MIN,
};
use fitting_core::data::{load_fashion_mnist, load_mnist, load_pbmc, load_wordnet_mammals};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::synthetic_data::DataPoints;
use plotters::prelude::*;

const N: usize = 500;
const DIM: usize = 2;
const N_GRID: usize = 60;
/// Radius sweep, as a multiple of the dataset's own `d_max`.  Lower bound
/// `d_max/20` matches `fit_hyperbolic`'s overflow-safe floor
/// (`cosh(20) ≈ 2.4·10⁸`); upper bound reaches well into the flat limit.
const R_LO_FRAC: f64 = 1.0 / 20.0;
const R_HI_FRAC: f64 = 10.0;
/// Floor for the raw-energy log axis, so exact-fit dips (ρ≈0) stay finite.
const RAW_LOG_FLOOR: f64 = -6.0;

/// Search bounds as fractions of `d_max`, in `log₁₀(r/d_max)` units.
fn spherical_bounds_x() -> (f64, f64) {
    (
        (1.0 / std::f64::consts::PI).log10(),
        (1.0 / SPHERICAL_ANGULAR_MIN).log10(),
    )
}
fn hyperbolic_bounds_x() -> (f64, f64) {
    ((1.0 / 20.0_f64).log10(), 1.0_f64.log10())
}

/// `‖Z(r)‖_F² = Σ_ij (kernel(d_ij, r))²` summed directly from the flat
/// distance matrix (includes the diagonal, `cos(0)=cosh(0)=1`).
fn frob_sq_z(distances: &[f64], r: f64, hyperbolic: bool) -> f64 {
    let r2 = r * r;
    let inv_r = 1.0 / r;
    distances
        .iter()
        .map(|&d| {
            let z = if hyperbolic {
                r2 * (d * inv_r).cosh()
            } else {
                r2 * (d * inv_r).cos()
            };
            z * z
        })
        .sum()
}

/// log₁₀ of raw misfit energy `E = ρ·‖Z‖_F²`, scaled by the constant
/// `d_max⁴` so datasets of different diameter overlay.
fn raw_log(rho: f64, distances: &[f64], r: f64, d_max: f64, hyperbolic: bool) -> f64 {
    let energy = rho * frob_sq_z(distances, r, hyperbolic) / d_max.powi(4);
    energy.max(10f64.powf(RAW_LOG_FLOOR)).log10()
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

struct Case {
    name: &'static str,
    color: RGBColor,
    n: usize,
    d_max: f64,
    /// `log₁₀(r / d_max)` grid, shared shape across datasets.
    xs: Vec<f64>,
    s_norm: Vec<f64>,
    h_norm: Vec<f64>,
    s_raw: Vec<f64>,
    h_raw: Vec<f64>,
    fit_s: WilsonFit,
    fit_h: WilsonFit,
    fit_s_raw: f64,
    fit_h_raw: f64,
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
        let d = distances_for(data);
        let d_max = d.iter().cloned().fold(0.0_f64, f64::max);

        let log_lo = R_LO_FRAC.ln();
        let log_hi = R_HI_FRAC.ln();
        let step = (log_hi - log_lo) / (N_GRID - 1) as f64;

        let mut xs = Vec::with_capacity(N_GRID);
        let (mut s_norm, mut h_norm) = (Vec::new(), Vec::new());
        let (mut s_raw, mut h_raw) = (Vec::new(), Vec::new());
        for i in 0..N_GRID {
            let frac = (log_lo + i as f64 * step).exp(); // r / d_max
            let r = frac * d_max;
            xs.push(frac.log10());

            let rs = spherical_residual_at(&d, n, DIM, r);
            let rh = hyperbolic_residual_at(&d, n, DIM, r);
            s_norm.push(rs);
            h_norm.push(rh);
            s_raw.push(raw_log(rs, &d, r, d_max, false));
            h_raw.push(raw_log(rh, &d, r, d_max, true));
        }

        let fit_s = fit_spherical(&d, n, DIM);
        let fit_h = fit_hyperbolic(&d, n, DIM);
        let fit_s_raw = raw_log(fit_s.residual, &d, fit_s.radius, d_max, false);
        let fit_h_raw = raw_log(fit_h.residual, &d, fit_h.radius, d_max, true);

        cases.push(Case {
            name,
            color: *color,
            n,
            d_max,
            xs,
            s_norm,
            h_norm,
            s_raw,
            h_raw,
            fit_s,
            fit_h,
            fit_s_raw,
            fit_h_raw,
        });
    }

    // ── Console report ──────────────────────────────────────────────
    for case in &cases {
        println!(
            "\n── {} (n = {}, d_max = {:.3}) ──",
            case.name, case.n, case.d_max
        );
        let s = &case.fit_s;
        let h = &case.fit_h;
        println!(
            "  spherical fit:  r* = {:.4}  (r*/d_max = {:.3}, angular d_max/r* = {:.3}),  ρ = {:.4}{}",
            s.radius,
            s.radius / case.d_max,
            case.d_max / s.radius,
            s.residual,
            if s.at_upper_bound { "  [at upper bound → not spherical]" } else { "  [interior → spherical]" },
        );
        println!(
            "  hyperbolic fit: r* = {:.4}  (r*/d_max = {:.3}),  ρ = {:.4}{}",
            h.radius,
            h.radius / case.d_max,
            h.residual,
            if h.at_upper_bound {
                "  [at upper bound → flat]"
            } else {
                ""
            },
        );
    }

    // ── Plots ───────────────────────────────────────────────────────
    let sph_b = spherical_bounds_x();
    let hyp_b = hyperbolic_bounds_x();

    plot(
        "plots/wilson_residual_real_spherical.png",
        "Wilson spherical residual ρ(r) — real datasets (normalised)",
        "normalised residual ρ(r)",
        (0.0, 1.05),
        &cases,
        |c| &c.s_norm,
        |c| ((c.fit_s.radius / c.d_max).log10(), c.fit_s.residual),
        sph_b,
    )?;
    plot(
        "plots/wilson_residual_real_hyperbolic.png",
        "Wilson hyperbolic residual ρ(r) — real datasets (normalised)",
        "normalised residual ρ(r)",
        (0.0, 1.05),
        &cases,
        |c| &c.h_norm,
        |c| ((c.fit_h.radius / c.d_max).log10(), c.fit_h.residual),
        hyp_b,
    )?;
    plot(
        "plots/wilson_residual_real_spherical_raw.png",
        "Wilson spherical raw energy E(r) — real datasets (unnormalised)",
        "log10( raw energy E(r) / d_max^4 )",
        y_range_raw(&cases, |c| &c.s_raw),
        &cases,
        |c| &c.s_raw,
        |c| ((c.fit_s.radius / c.d_max).log10(), c.fit_s_raw),
        sph_b,
    )?;
    plot(
        "plots/wilson_residual_real_hyperbolic_raw.png",
        "Wilson hyperbolic raw energy E(r) — real datasets (unnormalised)",
        "log10( raw energy E(r) / d_max^4 )",
        y_range_raw(&cases, |c| &c.h_raw),
        &cases,
        |c| &c.h_raw,
        |c| ((c.fit_h.radius / c.d_max).log10(), c.fit_h_raw),
        hyp_b,
    )?;

    println!("\nWrote 4 plots to plots/ (real datasets: normalised + raw, spherical + hyperbolic)");
    Ok(())
}

/// Padded [min, max] over a raw-energy series across all datasets.
fn y_range_raw(cases: &[Case], series: impl Fn(&Case) -> &Vec<f64>) -> (f64, f64) {
    let lo = cases
        .iter()
        .flat_map(|c| series(c).iter().cloned())
        .fold(f64::INFINITY, f64::min);
    let hi = cases
        .iter()
        .flat_map(|c| series(c).iter().cloned())
        .fold(f64::NEG_INFINITY, f64::max);
    let pad = 0.05 * (hi - lo).max(1.0);
    (lo - pad, hi + pad)
}

#[allow(clippy::too_many_arguments)]
fn plot(
    path: &str,
    caption: &str,
    y_desc: &str,
    y_range: (f64, f64),
    cases: &[Case],
    series: impl Fn(&Case) -> &Vec<f64>,
    fit: impl Fn(&Case) -> (f64, f64),
    bounds: (f64, f64),
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (960, 640)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_min = cases
        .iter()
        .flat_map(|c| c.xs.iter().cloned())
        .fold(f64::INFINITY, f64::min);
    let x_max = cases
        .iter()
        .flat_map(|c| c.xs.iter().cloned())
        .fold(f64::NEG_INFINITY, f64::max);
    let (y_lo, y_hi) = y_range;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("log10(r / d_max)   (small r  <--   -->  flat limit)")
        .y_desc(y_desc)
        .axis_desc_style(("sans-serif", 18))
        .draw()?;

    // Search bounds: solid black verticals with labels.
    for (bx, tag) in [(bounds.0, "r_lo"), (bounds.1, "r_hi")] {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(bx, y_lo), (bx, y_hi)],
            BLACK.mix(0.6).stroke_width(1),
        )))?;
        chart.draw_series(std::iter::once(Text::new(
            tag,
            (bx, y_lo + 0.94 * (y_hi - y_lo)),
            ("sans-serif", 15).into_font().color(&BLACK.mix(0.7)),
        )))?;
    }

    for case in cases {
        let ys = series(case);
        let line: Vec<(f64, f64)> = case.xs.iter().cloned().zip(ys.iter().cloned()).collect();
        chart
            .draw_series(LineSeries::new(line, case.color.stroke_width(2)))?
            .label(case.name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], case.color));

        // Bold cross at the fitted r*, with a guide line down to the axis.
        let (fx, fy) = fit(case);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(fx, y_lo), (fx, fy)],
            case.color.mix(0.4).stroke_width(1),
        )))?;
        chart.draw_series(std::iter::once(Cross::new(
            (fx, fy),
            7,
            case.color.stroke_width(3),
        )))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 18))
        .draw()?;

    root.present()?;
    Ok(())
}
