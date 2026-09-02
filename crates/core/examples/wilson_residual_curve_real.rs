//! Plot the Wilson residual-eigenvalue misfit against the radius of
//! curvature `r` for the **real** datasets — the same sweep used by
//! `wilson_residual_curve.rs` for the synthetic fixtures (see that
//! example for the method description and the theory).
//!
//! Datasets are the four real fixtures of the thesis table (`common::real`),
//! in table order:
//!   - `mnist` / `mnist-fashion` — 28×28 pixel vectors (Euclidean distances in pixel space).
//!   - `wordnet-mammals` — the mammal subtree; distances are the graph shortest-path (BFS) matrix, already supplied by the loader.
//!   - `pbmc` — single-cell RNA-seq PCA features (Euclidean distances).
//!
//! For each dataset we sweep `r` and record, for both the spherical
//! (`r² cos(d/r)`) and hyperbolic (`−r² cosh(d/r)`) Gram models, the residual
//! `Σ|λ_res|(r)` the fitter minimises (Wilson et al. 2014 §V.B), scaled by the
//! dataset constants `n · d_max²` so datasets of different diameter and sample
//! size overlay.  That is also the gauge `detect_geometry` thresholds, so a
//! point read off the y-axis compares directly against `SPHERICAL_RESIDUAL_MAX`.
//!
//! Unlike the synthetic fixtures, none of these are exact constant-curvature
//! manifolds, so the interesting question is not whether the residual reaches
//! zero but how *deep* the notch at `r*` is relative to the flat-limit plateau.
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
//!     --example wilson_residual_curve_real -- --data-root path/to/data
//! cargo run --release -p fitting-core --features plot-examples \
//!     --example wilson_residual_curve_real -- --n 500
//! ```
//!
//! Writes two SVGs to `plots/` (spherical + hyperbolic).

use std::error::Error;

mod common;
use common::{d_max_of, map_parallel, CommonArgs, Fixture, DIM};

use fitting_core::curvature_detection::{
    fit_hyperbolic, fit_spherical, hyperbolic_residual_at, spherical_residual_at, WilsonFit,
    SPHERICAL_ANGULAR_MIN, SPHERICAL_RESIDUAL_MAX,
};
use plotters::prelude::*;

const N_GRID: usize = 60;
/// Radius sweep, as a multiple of the dataset's own `d_max`.  Lower bound
/// `d_max/20` matches `fit_hyperbolic`'s overflow-safe floor
/// (`cosh(20) ≈ 2.4·10⁸`); upper bound reaches well into the flat limit.
const R_LO_FRAC: f64 = 1.0 / 20.0;
const R_HI_FRAC: f64 = 10.0;
/// Extra samples around each fitted `r*`, matching `wilson_residual_curve.rs`
/// so the two examples really do run the identical sweep.
const DENSE_BAND: usize = 21;
const DENSE_BAND_SPAN: f64 = 1.35;
/// Floor for the absolute-residual log axis, so near-exact fits stay finite.
const ABS_LOG_FLOOR: f64 = -8.0;

/// SVG canvas per panel, matching `wilson_residual_curve.rs` so all four panels
/// of the 2×2 A4 grid share a scale.
const CANVAS: (u32, u32) = (440, 320);

/// Plotted `log₁₀(r/d_max)` window for the spherical panel, matching
/// `wilson_residual_curve.rs`: the spherical search window sits at `x ≈ −0.50 ..
/// −0.40`, so a decade below the diameter is all that is worth showing.
const SPHERICAL_X: (f64, f64) = (-1.0, 0.0);

/// What differs between the two panels — window, legend corner, grid density —
/// matching `wilson_residual_curve.rs`.
struct Panel {
    x_window: Option<(f64, f64)>,
    legend: SeriesLabelPosition,
    /// Light grid lines drawn between two labelled ticks.
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

/// log₁₀ of the residual `Σ|λ_res|`, scaled by the dataset constants
/// `n · d_max²`: `d_max²` carries the units of `Z` and `n` removes the
/// extensivity of its spectrum (see `wilson_residual_curve.rs`).  Neither
/// factor depends on `r`, so this is a pure vertical shift and leaves `r*`
/// alone.  It matters more here than on the synthetic fixtures, since the
/// loaders can return fewer than `N` points for some datasets.  Same gauge as
/// [`WilsonFit::residual_normalised`], so the axis is directly comparable to
/// [`SPHERICAL_RESIDUAL_MAX`].
fn abs_log(absolute: f64, n: usize, d_max: f64) -> f64 {
    (absolute / (n as f64 * d_max * d_max))
        .max(10f64.powf(ABS_LOG_FLOOR))
        .log10()
}

/// Why `detect_geometry` would or would not accept this spherical fit: it
/// needs an interior `r*` (not pinned at the flat-ward bound) *and* a
/// normalised residual below [`SPHERICAL_RESIDUAL_MAX`].  Both conditions
/// matter — pbmc and wordnet_mammals land interior but miss the residual test.
fn spherical_verdict(fit: &WilsonFit) -> &'static str {
    match (
        fit.at_upper_bound,
        fit.residual_normalised < SPHERICAL_RESIDUAL_MAX,
    ) {
        (false, true) => "  [interior + low residual → spherical]",
        (false, false) => "  [interior but residual too high → not spherical]",
        (true, _) => "  [at upper bound → not spherical]",
    }
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

    // Coarse log sweep plus a dense band around each fitted r*, so a minimum
    // narrower than the coarse spacing is drawn rather than straddled.
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

    let fixtures = common::real(args.n, &args.data_root);
    if fixtures.is_empty() {
        return Err(format!("no real datasets loaded from {}", args.data_root).into());
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
            spherical_verdict(s),
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
        "plots/wilson_residual_real_spherical.svg",
        "Wilson spherical fit — real datasets",
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
        "plots/wilson_residual_real_hyperbolic.svg",
        "Wilson hyperbolic fit — real datasets",
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
    println!("\nWrote 2 plots to plots/ (real datasets: spherical + hyperbolic)");
    Ok(())
}

/// Is `x` inside the plotted window (everything, if there is no window)?
fn in_x(x: f64, window: Option<(f64, f64)>) -> bool {
    window.is_none_or(|(lo, hi)| (lo..=hi).contains(&x))
}

/// Padded [min, max] over a log-residual series across all datasets, restricted
/// to the plotted `x` window so a clipped panel is scaled by what it shows.
/// The fitted `r*` values are folded in so the `r*` cross cannot land outside
/// the axes when the fit sits in a notch narrower than the sweep's grid spacing.
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
