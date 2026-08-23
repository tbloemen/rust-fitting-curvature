//! Plot the Wilson residual-eigenvalue misfit against the radius of
//! curvature `r`, to visualise how the radius fit works and where it
//! degenerates.
//!
//! For each candidate radius `r` we build the constant-curvature Gram
//! matrix `Z(r)` (spherical `r² cos(d/r)`, hyperbolic `−r² cosh(d/r)`)
//! and sum `|λ|` over the eigenvalues lying outside the signature block a
//! genuine `dim`-dimensional constant-curvature configuration is allowed
//! to occupy (Wilson et al. 2014, §V.B: `r* = arg min_r Σ_{i≤n−m} |λᵢ|`).
//!
//! Plotted as `log₁₀(Σ|λ_res| / (n·d_max²))`: the residual spans many
//! decades, `Z` carries units of squared distance, and its eigenvalues are
//! extensive in `n`, so dividing by the dataset constants `n·d_max²`
//! overlays datasets of different diameter *and* different sample size.
//! Neither factor depends on `r`, so this is a pure vertical shift — the
//! curve shape, and hence `r*`, is the paper's objective untouched.  It is
//! also the quantity `detect_geometry` thresholds
//! (`SPHERICAL_RESIDUAL_MAX`), so the plot and the gate read the same axis.
//!
//! Sweeping `r` exposes the theory:
//!
//!   - `r ≈ R` (true radius): `Z(R)` collapses to the geometry-mandated
//!     rank+signature, so both curves plunge toward 0 — a sharp, deep,
//!     unmistakable notch, since on exact manifold data the residual
//!     eigenvalues are pure numerical noise.  This is the signal the fit
//!     locates.
//!   - large `r`: `Z(r) → ±r²J ∓ D²/2` recovers the classical-MDS kernel
//!     (Wilson et al. 2014, eqs. 24–26) and stops carrying curvature
//!     information.  The residual **plateaus** at a finite value there
//!     rather than vanishing — unlike the old `‖Z‖_F²`-normalised
//!     objective, whose denominator grew like `n²r⁴` and so drove the
//!     residual to 0 for *every* geometry, manufacturing a spurious
//!     large-`r` attractor.
//!   - small `r` (spherical only): `cos(d/r)` oscillates once distances
//!     wrap past `πr`, so the residual climbs steeply — which is why the
//!     search floor sits at the feasibility bound `d_max/π`.
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
//! Writes two PNGs to `plots/` (spherical + hyperbolic).

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
/// Floor for the absolute-residual log axis, so exact-fit dips
/// (`Σ|λ_res|` ≈ 0 on manifold data) stay finite.
const ABS_LOG_FLOOR: f64 = -8.0;

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

/// log₁₀ of the residual `Σ|λ_res|`, scaled by the constant `n · d_max²` —
/// the same gauge as [`WilsonFit::residual_normalised`], so a point read off
/// this axis can be compared directly against `SPHERICAL_RESIDUAL_MAX`.
///
/// `d_max²` carries the units of `Z` (squared distance) and `n` removes the
/// extensivity: `Z(r)` is a kernel matrix, so `λᵢ(Z)/n` converges to the
/// eigenvalues of the corresponding integral operator and *every* eigenvalue
/// grows like `n` (exactly: `tr Z = ±n r²`).  `Σ|λ_res| / n` is therefore the
/// Monte-Carlo estimate of a population quantity, and the curves overlay
/// across sample sizes as well as diameters.  Both factors are properties of
/// the dataset, not of `r`, so this is a fixed vertical shift that leaves the
/// paper's objective, and hence `r*`, untouched.
fn abs_log(absolute: f64, n: usize, d_max: f64) -> f64 {
    (absolute / (n as f64 * d_max * d_max))
        .max(10f64.powf(ABS_LOG_FLOOR))
        .log10()
}

struct Case {
    name: &'static str,
    color: RGBColor,
    d_max: f64,
    /// `log₁₀(r / d_max)` grid, shared shape across datasets.
    xs: Vec<f64>,
    s_abs: Vec<f64>,
    h_abs: Vec<f64>,
    fit_s: WilsonFit,
    fit_h: WilsonFit,
    fit_s_abs: f64,
    fit_h_abs: f64,
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
        let (mut s_abs, mut h_abs) = (Vec::new(), Vec::new());
        for i in 0..N_GRID {
            let frac = (log_lo + i as f64 * step).exp(); // r / d_max
            let r = frac * d_max;
            xs.push(frac.log10());

            s_abs.push(abs_log(spherical_residual_at(d, N, DIM, r), N, d_max));
            h_abs.push(abs_log(hyperbolic_residual_at(d, N, DIM, r), N, d_max));
        }

        let fit_s = fit_spherical(d, N, DIM);
        let fit_h = fit_hyperbolic(d, N, DIM);
        let fit_s_abs = abs_log(fit_s.residual, N, d_max);
        let fit_h_abs = abs_log(fit_h.residual, N, d_max);

        cases.push(Case {
            name,
            color: *color,
            d_max,
            xs,
            s_abs,
            h_abs,
            fit_s,
            fit_h,
            fit_s_abs,
            fit_h_abs,
        });
    }

    // ── Console report ──────────────────────────────────────────────
    for case in &cases {
        println!("\n── {} (d_max = {:.3}) ──", case.name, case.d_max);
        let s = &case.fit_s;
        let h = &case.fit_h;
        println!(
            "  spherical fit:  r* = {:.4}  (r*/d_max = {:.3}, angular d_max/r* = {:.3}),  Σ|λ_res| = {:.4e}  (normalised {:.3e}){}",
            s.radius,
            s.radius / case.d_max,
            case.d_max / s.radius,
            s.residual,
            s.residual_normalised,
            if s.at_upper_bound { "  [at upper bound → flat]" } else { "" },
        );
        println!(
            "  hyperbolic fit: r* = {:.4}  (r*/d_max = {:.3}),  Σ|λ_res| = {:.4e}  (normalised {:.3e}){}",
            h.radius,
            h.radius / case.d_max,
            h.residual,
            h.residual_normalised,
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
        "Wilson spherical fit — residual eigenvalues (the fitted objective)",
        "log10( sum |lambda_res| / (n * d_max^2) )",
        y_range_raw(&cases, |c| &c.s_abs),
        &cases,
        |c| &c.s_abs,
        |c| ((c.fit_s.radius / c.d_max).log10(), c.fit_s_abs),
        sph_b,
    )?;
    plot(
        "plots/wilson_residual_hyperbolic.png",
        "Wilson hyperbolic fit — residual eigenvalues (the fitted objective)",
        "log10( sum |lambda_res| / (n * d_max^2) )",
        y_range_raw(&cases, |c| &c.h_abs),
        &cases,
        |c| &c.h_abs,
        |c| ((c.fit_h.radius / c.d_max).log10(), c.fit_h_abs),
        hyp_b,
    )?;
    println!("\nWrote 2 plots to plots/ (spherical + hyperbolic)");
    Ok(())
}

/// Padded [min, max] over a log-residual series across all datasets.
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
