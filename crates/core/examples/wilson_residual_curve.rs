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
//! Datasets are the six synthetic fixtures of the thesis table
//! (`common::synthetic`): `grid 2D`/`grid 10D`, `sphere 2D`/`sphere 10D`,
//! `tree 2D`/`tree 10D`.  Only the `2D` members are exact two-dimensional
//! constant-curvature manifolds, so only they can reach the deep notch; the
//! `10D` members are S⁹ / H⁹ / R¹⁰ objects charged the mass of their surplus
//! dimensions at `dim = 2`, which is why they sit high and flat.  Pass
//! `--all` to add the `H2 exact` control and the remaining generators.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p fitting-core --features plot-examples \
//!     --example wilson_residual_curve
//! cargo run --release -p fitting-core --features plot-examples \
//!     --example wilson_residual_curve -- --n 300
//! ```
//!
//! Writes two SVGs to `plots/` (spherical + hyperbolic).

use std::error::Error;

mod common;
use common::{d_max_of, map_parallel, CommonArgs, Fixture, DIM};

use fitting_core::curvature_detection::{
    fit_hyperbolic, fit_spherical, hyperbolic_residual_at, spherical_residual_at, WilsonFit,
    SPHERICAL_ANGULAR_MIN,
};
use plotters::prelude::*;

const N_GRID: usize = 70;
/// Radius sweep, as a multiple of the dataset's own `d_max`.  Lower bound
/// `d_max/20` matches `fit_hyperbolic`'s overflow-safe floor
/// (`cosh(20) ≈ 2.4·10⁸`); upper bound reaches well into the flat limit.
const R_LO_FRAC: f64 = 1.0 / 20.0;
const R_HI_FRAC: f64 = 10.0;
/// Extra samples placed around each fitted `r*`, spanning
/// `r* · [1/DENSE_BAND_SPAN, DENSE_BAND_SPAN]`, so a narrow exact-manifold
/// notch is drawn rather than straddled.
const DENSE_BAND: usize = 21;
const DENSE_BAND_SPAN: f64 = 1.35;
/// Floor for the absolute-residual log axis, so exact-fit dips
/// (`Σ|λ_res|` ≈ 0 on manifold data) stay finite.
const ABS_LOG_FLOOR: f64 = -8.0;

/// SVG canvas per panel, in user units.  Sized for four panels in a 2×2 grid on
/// A4: at a ~160 mm text width each panel is ~78 mm across, so one unit is
/// ~0.18 mm and the font sizes below land at roughly 9 pt (caption), 7 pt (axis
/// titles) and 6 pt (tick labels) on the page.
const CANVAS: (u32, u32) = (440, 320);

/// Plotted `log₁₀(r/d_max)` window for the spherical panel.  The spherical
/// search window is `[d_max/π, d_max/SPHERICAL_ANGULAR_MIN]`, i.e. `x ≈ −0.50 ..
/// −0.40`, so a decade below the diameter is all that is worth showing; the
/// oscillatory small-`r` tail off to the left only compresses it.
const SPHERICAL_X: (f64, f64) = (-1.0, 0.0);

/// What differs between the two panels: how much of the sweep is shown, where
/// the legend goes, and how finely the grid is subdivided.
struct Panel {
    x_window: Option<(f64, f64)>,
    legend: SeriesLabelPosition,
    /// Light grid lines drawn between two labelled ticks.  Over one decade the
    /// default subdivision is a thicket, so the spherical panel asks for less.
    light_lines: usize,
}

impl Panel {
    fn spherical() -> Self {
        Panel {
            x_window: Some(SPHERICAL_X),
            legend: SeriesLabelPosition::LowerLeft,
            light_lines: 2,
        }
    }
    fn hyperbolic() -> Self {
        Panel {
            x_window: None,
            legend: SeriesLabelPosition::UpperRight,
            light_lines: 10,
        }
    }
}

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
    n: usize,
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

fn build_case(fx: &Fixture) -> Case {
    let (d, n) = (fx.distances.as_slice(), fx.n);
    let d_max = d_max_of(d);

    let fit_s = fit_spherical(d, n, DIM);
    let fit_h = fit_hyperbolic(d, n, DIM);

    // Coarse log sweep, plus a dense band around each fitted r*.  On an exact
    // manifold the minimum is far narrower than the coarse spacing — `tree 2D`
    // fits at r* = 1.0 with a residual three decades below its neighbours — so
    // without the refinement the drawn curve straddles the notch and the r*
    // cross appears to float below the line that is supposed to pass through it.
    let log_lo = R_LO_FRAC.ln();
    let log_hi = R_HI_FRAC.ln();
    let step = (log_hi - log_lo) / (N_GRID - 1) as f64;
    let mut rs: Vec<f64> = (0..N_GRID)
        .map(|i| (log_lo + i as f64 * step).exp() * d_max)
        .collect();
    for r_star in [fit_s.radius, fit_h.radius] {
        for i in 0..DENSE_BAND {
            let t = i as f64 / (DENSE_BAND - 1) as f64;
            rs.push(r_star * DENSE_BAND_SPAN.powf(2.0 * t - 1.0));
        }
    }
    // Exact samples on the spherical panel's frame, so its curves run edge to
    // edge instead of stopping at the nearest coarse grid point inside it.
    for x in [SPHERICAL_X.0, SPHERICAL_X.1] {
        rs.push(10f64.powf(x) * d_max);
    }
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut xs = Vec::with_capacity(rs.len());
    let (mut s_abs, mut h_abs) = (Vec::new(), Vec::new());
    for &r in &rs {
        xs.push((r / d_max).log10());
        s_abs.push(abs_log(spherical_residual_at(d, n, DIM, r), n, d_max));
        h_abs.push(abs_log(hyperbolic_residual_at(d, n, DIM, r), n, d_max));
    }

    Case {
        name: fx.name,
        color: RGBColor(fx.color.0, fx.color.1, fx.color.2),
        n,
        d_max,
        xs,
        s_abs,
        h_abs,
        fit_s_abs: abs_log(fit_s.residual, n, d_max),
        fit_h_abs: abs_log(fit_h.residual, n, d_max),
        fit_s,
        fit_h,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = CommonArgs::parse(&[])?;
    std::fs::create_dir_all("plots")?;

    let mut fixtures = common::synthetic(args.n, args.seed);
    if args.all {
        fixtures.extend(common::controls(args.n, args.seed));
    }
    let cases: Vec<Case> = map_parallel(&fixtures, build_case);

    // ── Console report ──────────────────────────────────────────────
    for case in &cases {
        println!(
            "\n── {} (n = {}, d_max = {:.3}) ──",
            case.name, case.n, case.d_max
        );
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
        "plots/wilson_residual_spherical.svg",
        "Wilson spherical fit — synthetic datasets",
        "log10( sum |lambda_res| / (n * d_max^2) )",
        y_range_raw(
            &cases,
            |c| &c.s_abs,
            |c| ((c.fit_s.radius / c.d_max).log10(), c.fit_s_abs),
            Some(SPHERICAL_X),
        ),
        &cases,
        |c| &c.s_abs,
        |c| ((c.fit_s.radius / c.d_max).log10(), c.fit_s_abs),
        sph_b,
        Panel::spherical(),
    )?;
    plot(
        "plots/wilson_residual_hyperbolic.svg",
        "Wilson hyperbolic fit — synthetic datasets",
        "log10( sum |lambda_res| / (n * d_max^2) )",
        y_range_raw(
            &cases,
            |c| &c.h_abs,
            |c| ((c.fit_h.radius / c.d_max).log10(), c.fit_h_abs),
            None,
        ),
        &cases,
        |c| &c.h_abs,
        |c| ((c.fit_h.radius / c.d_max).log10(), c.fit_h_abs),
        hyp_b,
        Panel::hyperbolic(),
    )?;
    println!("\nWrote 2 plots to plots/ (spherical + hyperbolic)");
    Ok(())
}

/// Is `x` inside the plotted window (everything, if there is no window)?
fn in_x(x: f64, window: Option<(f64, f64)>) -> bool {
    window.is_none_or(|(lo, hi)| (lo..=hi).contains(&x))
}

/// Padded [min, max] over a log-residual series across all datasets, restricted
/// to the plotted `x` window so a clipped panel is scaled by what it shows.
///
/// The fitted `r*` values are folded in as well as the swept curve.  On an
/// exact manifold the fit lands in a notch far narrower than the sweep's grid
/// spacing — `sphere 2D` sweeps down to ~1e-5 but fits at 3e-8 — so a range
/// taken from the curve alone would draw the `r*` cross outside the axes.
fn y_range_raw(
    cases: &[Case],
    series: impl Fn(&Case) -> &Vec<f64>,
    fit: impl Fn(&Case) -> (f64, f64),
    x_window: Option<(f64, f64)>,
) -> (f64, f64) {
    let values = |c: &Case| {
        let (fx, fy) = fit(c);
        c.xs.iter()
            .zip(series(c).iter())
            .filter(|(&x, _)| in_x(x, x_window))
            .map(|(_, &y)| y)
            .chain(in_x(fx, x_window).then_some(fy))
            .collect::<Vec<_>>()
    };
    let lo = cases.iter().flat_map(values).fold(f64::INFINITY, f64::min);
    let hi = cases
        .iter()
        .flat_map(values)
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
    panel: Panel,
) -> Result<(), Box<dyn Error>> {
    let x_window = panel.x_window;
    let root = SVGBackend::new(path, CANVAS).into_drawing_area();
    root.fill(&WHITE)?;

    let (x_min, x_max) = x_window.unwrap_or_else(|| {
        let xs = || cases.iter().flat_map(|c| c.xs.iter().cloned());
        (
            xs().fold(f64::INFINITY, f64::min),
            xs().fold(f64::NEG_INFINITY, f64::max),
        )
    });
    let (y_lo, y_hi) = y_range;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 18))
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(52)
        .build_cartesian_2d(x_min..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("log10(r / d_max)   (small r  <--   -->  flat limit)")
        .y_desc(y_desc)
        // Plotters hands the formatter the raw key point, which at the right
        // edge of the spherical window is a negative zero and prints as "-0.0".
        .x_label_formatter(&|v| {
            let t = (v * 10.0).round() / 10.0;
            format!("{:.1}", if t == 0.0 { 0.0 } else { t })
        })
        .axis_desc_style(("sans-serif", 13))
        .label_style(("sans-serif", 11))
        .max_light_lines(panel.light_lines)
        .draw()?;

    // Search bounds: solid black verticals with labels.
    for (bx, tag) in [(bounds.0, "r_lo"), (bounds.1, "r_hi")] {
        if !in_x(bx, x_window) {
            continue;
        }
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(bx, y_lo), (bx, y_hi)],
            BLACK.mix(0.6).stroke_width(1),
        )))?;
        chart.draw_series(std::iter::once(Text::new(
            tag,
            (bx, y_lo + 0.94 * (y_hi - y_lo)),
            ("sans-serif", 11).into_font().color(&BLACK.mix(0.7)),
        )))?;
    }

    for case in cases {
        let ys = series(case);
        // Points outside the window are dropped rather than clamped onto the
        // frame, which would draw a spurious vertical run up the edge.
        let line: Vec<(f64, f64)> = case
            .xs
            .iter()
            .cloned()
            .zip(ys.iter().cloned())
            .filter(|&(x, _)| in_x(x, x_window))
            .collect();
        chart
            .draw_series(LineSeries::new(line, case.color.stroke_width(2)))?
            .label(case.name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], case.color));

        // Bold cross at the fitted r*, with a guide line down to the axis.
        let (fx, fy) = fit(case);
        if in_x(fx, x_window) {
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
    }

    chart
        .configure_series_labels()
        .position(panel.legend)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()?;

    root.present()?;
    Ok(())
}
