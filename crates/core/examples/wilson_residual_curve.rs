//! Plot the Wilson signature residual against the radius of curvature
//! `r`, both **normalised** and **unnormalised**, to visualise how the
//! radius fit works and where it degenerates.
//!
//! For each candidate radius `r` we build the constant-curvature Gram
//! matrix `Z(r)` (spherical `r² cos(d/r)`, hyperbolic `−r² cosh(d/r)`)
//! and measure the energy lying outside the rank-`(dim+1)` signature
//! subspace a genuine `dim`-dimensional constant-curvature configuration
//! is allowed to occupy.  Two objectives:
//!
//!   - **normalised** `ρ(r) = E(r) / ‖Z(r)‖_F² ∈ [0,1]` — what the fitter
//!     minimises (`r* = arg min_r ρ`).
//!   - **raw** `E(r) = ρ(r)·‖Z(r)‖_F²` — the unnormalised misfit energy.
//!
//! Sweeping `r` exposes the theory:
//!
//!   - `r ≈ R` (true radius): `Z(R)` collapses to the geometry-mandated
//!     rank+signature, so `ρ` dips toward 0 — the signal the fit locates.
//!   - large `r`: `Z(r) ≈ r²J + K − K'` recovers the classical-MDS
//!     kernel `K` (Wilson et al. 2014, eqs. 24–26), so the normalised
//!     residual → 0 for *every* geometry because the `r²J` spike inflates
//!     `‖Z‖_F² ∼ n²r⁴` — the identifiability boundary.
//!   - the **raw** energy removes that denominator, so the spurious
//!     large-`r` attractor disappears — but for the bounded `cos` kernel
//!     it reintroduces the mirror artifact: `E(r) ∼ r⁴ → 0` as `r → 0`,
//!     biasing the raw objective toward small `r` (high curvature).  This
//!     `r⁴` bias is exactly why the fit normalises by `‖Z‖_F²`.
//!
//! Vertical black lines mark the search bounds each fit actually uses
//! (spherical `[d_max/π, d_max/SPHERICAL_ANGULAR_MIN]`, hyperbolic
//! `[d_max/20, d_max]`); the
//! bounds are fixed fractions of `d_max`, so they are the same on every
//! dataset in the shared `log₁₀(r/d_max)` x-axis.  A bold cross marks the
//! fitted `r*`.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p fitting-core --features plot-examples \
//!     --example wilson_residual_curve
//! ```
//!
//! Writes four PNGs to `plots/` (normalised + raw, spherical + hyperbolic).

use std::error::Error;

use fitting_core::curvature_detection::{
    fit_hyperbolic, fit_spherical, hyperbolic_residual_at, spherical_residual_at, WilsonFit,
    SPHERICAL_ANGULAR_MIN,
};
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere, DataPoints,
};
use plotters::prelude::*;

const N: usize = 300;
const SEED: u64 = 42;
const DIM: usize = 2;
const N_GRID: usize = 70;
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
/// `d_max⁴` so datasets of different diameter overlay (this is a fixed
/// rescale — unlike `ρ` it does **not** divide out the r-dependent scale).
fn raw_log(rho: f64, distances: &[f64], r: f64, d_max: f64, hyperbolic: bool) -> f64 {
    let energy = rho * frob_sq_z(distances, r, hyperbolic) / d_max.powi(4);
    energy.max(10f64.powf(RAW_LOG_FLOOR)).log10()
}

struct Case {
    name: &'static str,
    color: RGBColor,
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
    let fixtures: Vec<(&'static str, RGBColor, DataPoints)> = vec![
        ("hyperbolic", RED, generate_uniform_hyperbolic(N, SEED, 5.0)),
        ("euclidean", BLUE, generate_uniform_ball_2d(N, SEED, 5.0)),
        ("spherical", GREEN, generate_uniform_sphere(N, SEED)),
    ];

    let mut cases: Vec<Case> = Vec::new();
    for (name, color, data) in &fixtures {
        let d = &data.distances;
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

            let rs = spherical_residual_at(d, N, DIM, r);
            let rh = hyperbolic_residual_at(d, N, DIM, r);
            s_norm.push(rs);
            h_norm.push(rh);
            s_raw.push(raw_log(rs, d, r, d_max, false));
            h_raw.push(raw_log(rh, d, r, d_max, true));
        }

        let fit_s = fit_spherical(d, N, DIM);
        let fit_h = fit_hyperbolic(d, N, DIM);
        let fit_s_raw = raw_log(fit_s.residual, d, fit_s.radius, d_max, false);
        let fit_h_raw = raw_log(fit_h.residual, d, fit_h.radius, d_max, true);

        cases.push(Case {
            name,
            color: *color,
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
        println!("\n── {} (d_max = {:.3}) ──", case.name, case.d_max);
        let s = &case.fit_s;
        let h = &case.fit_h;
        println!(
            "  spherical fit:  r* = {:.4}  (r*/d_max = {:.3}, angular d_max/r* = {:.3}),  ρ = {:.4}{}",
            s.radius,
            s.radius / case.d_max,
            case.d_max / s.radius,
            s.residual,
            if s.at_upper_bound { "  [at upper bound → flat]" } else { "" },
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
        "plots/wilson_residual_spherical.png",
        "Wilson spherical residual ρ(r) — normalised",
        "normalised residual ρ(r)",
        (0.0, 1.05),
        &cases,
        |c| &c.s_norm,
        |c| ((c.fit_s.radius / c.d_max).log10(), c.fit_s.residual),
        sph_b,
    )?;
    plot(
        "plots/wilson_residual_hyperbolic.png",
        "Wilson hyperbolic residual ρ(r) — normalised",
        "normalised residual ρ(r)",
        (0.0, 1.05),
        &cases,
        |c| &c.h_norm,
        |c| ((c.fit_h.radius / c.d_max).log10(), c.fit_h.residual),
        hyp_b,
    )?;
    plot(
        "plots/wilson_residual_spherical_raw.png",
        "Wilson spherical raw energy E(r) — unnormalised",
        "log10( raw energy E(r) / d_max^4 )",
        y_range_raw(&cases, |c| &c.s_raw),
        &cases,
        |c| &c.s_raw,
        |c| ((c.fit_s.radius / c.d_max).log10(), c.fit_s_raw),
        sph_b,
    )?;
    plot(
        "plots/wilson_residual_hyperbolic_raw.png",
        "Wilson hyperbolic raw energy E(r) — unnormalised",
        "log10( raw energy E(r) / d_max^4 )",
        y_range_raw(&cases, |c| &c.h_raw),
        &cases,
        |c| &c.h_raw,
        |c| ((c.fit_h.radius / c.d_max).log10(), c.fit_h_raw),
        hyp_b,
    )?;

    println!("\nWrote 4 plots to plots/ (normalised + raw, spherical + hyperbolic)");
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
        .caption(caption, ("sans-serif", 26))
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
