//! The hyperbolic signature residual against the **dimensionless curvature**
//! `κ = |K|·R_rms²` of the model being tested, instead of against the radius `r`.
//!
//! `R_rms` is the RMS geodesic radius of the configuration a candidate radius
//! implies, so each sweep point is reconstructed to place it on the κ axis —
//! the same gauge the fits, `--mode detect` and the embeddings all report.
//! Sweeping `r` still sweeps `κ` monotonically (a larger radius is a flatter
//! model), and the search window becomes a `κ` interval whose flat-ward end is
//! `κ_min = `[`HYPERBOLIC_KAPPA_MIN`] by construction — the same demand for
//! every dataset, which is the point of stating the cap in `κ` rather than as a
//! multiple of `d_max`.  Plotting in this gauge makes two things visible that
//! the `r` axis hides:
//!
//! * a genuine two-dimensional hyperbolic object has a sharp interior minimum at
//!   its true `κ` — the Wilson fit is a good radius estimator when the model is
//!   actually true.  `tree 2D` is the case that shows it;
//! * every **other thesis dataset** instead descends monotonically across the
//!   whole window, so the minimiser stops at the cap and the reported curvature
//!   is just `κ_min` — the bound, not a measurement.  That includes `tree 10D`,
//!   which is the same tree lifted into H⁹: the interior minimum is a property
//!   of matching the fitted `dim`, not of the shape.
//!
//! The Gromov δ(k) verdict is printed alongside: the datasets it calls
//! hyperbolic behave exactly like the ones it does not, which is the empirical
//! case for keeping classification (Gromov) and radius estimation (Wilson)
//! separate.
//!
//! Writes two PNGs to `plots/`.  Data root defaults to `www/public/data`, as in
//! `wilson_residual_curve_real.rs`:
//!
//! ```text
//! cargo run --release -p fitting-core --features plot-examples \
//!     --example residual_vs_kappa
//! ```

use std::error::Error;

mod common;
use common::{d_max_of, d_rms_of, map_parallel, CommonArgs, Fixture, DIM};

use fitting_core::curvature_detection::{
    detect_hyperbolic, fit_hyperbolic, hyperbolic_residual_at, reconstruct_hyperbolic,
    HYPERBOLIC_KAPPA_MIN,
};
use plotters::prelude::*;

const N_GRID: usize = 70;
/// Radius sweep as a multiple of `d_max`, spanning well past both search bounds.
const R_LO_FRAC: f64 = 1.0 / 50.0;
const R_HI_FRAC: f64 = 50.0;
/// Floor for the log residual axis, so the near-exact control stays finite.
const LOG_FLOOR: f64 = -8.0;

// The production minimiser used to be mirrored here so the plotted `r*` would
// be the real one.  It is called directly instead: `fit_hyperbolic` now solves
// its flat-ward cap as a fixed point on `R_rms`, which a local copy would have
// to reproduce exactly, and a copy that drifts plots a fit the detector never
// made.  Calling the real thing makes the agreement structural.

struct Case {
    name: &'static str,
    color: RGBColor,
    d_max: f64,
    d_rms: f64,
    gromov_hyperbolic: bool,
    /// κ implied by the saturated Gromov δ via `δ = ln(1+√2)/√(−K)`, in the same
    /// gauge as `kappa_star`.  Computed for every dataset so the two estimators
    /// are comparable; the pipeline only *uses* it where `gromov_hyperbolic`.
    kappa_gromov: f64,
    /// Log–log slope of the residual across the search window,
    /// `Δlog R / Δlog κ` from the cap to `r = d_max/20`.  This is the
    /// **curvature leverage**: 1 means the residual is proportional to the
    /// curvature mismatch (exactly flat data), 0 means curvature explains none
    /// of the residual and the fit is riding a κ-independent floor.
    slope: f64,
    /// `R(κ_floor) / R(κ_cap)` — how much the residual moves across the whole
    /// window at all.
    leverage: f64,
    /// Log–log slope over the *last decade before the cap* (κ_cap → 3·κ_cap).
    /// This is how hard the objective is still pushing flat-ward where the
    /// constraint binds: large means the bound is strongly active, ≈0 means the
    /// curve has genuinely levelled off and the cap is nearly a stationary point.
    slope_at_cap: f64,
    /// `(log10 κ, log10 normalised residual)` along the sweep.
    curve: Vec<(f64, f64)>,
    /// Where the production fit lands, in the same coordinates.
    fit_x: f64,
    fit_y: f64,
    kappa_star: f64,
    pinned: bool,
}

fn build_case(fx: &Fixture) -> Case {
    let (d, n) = (fx.distances.as_slice(), fx.n);
    let dm = d_max_of(d);
    let dr = d_rms_of(d, n);
    let gauge = n as f64 * dm * dm;
    let norm = |r: f64| hyperbolic_residual_at(d, n, DIM, r) / gauge;

    // κ of the model at radius `r`, on the reported gauge: |K|·R_rms² measured
    // on the configuration that radius implies.
    let kappa_at = |r: f64| reconstruct_hyperbolic(d, n, DIM, r).kappa(n);

    // Take the window and the fit from production rather than mirroring them —
    // the cap is now a fixed-point solve, and a local copy would silently
    // desync the plotted r* from the one the detector actually uses.
    let fit = fit_hyperbolic(d, n, DIM);
    let (r_star, pinned) = (fit.radius, fit.at_upper_bound);
    // The flat-ward cap is where κ falls to the floor, so invert it through κ.
    let cap = {
        let mut r = r_star;
        for _ in 0..12 {
            let next = reconstruct_hyperbolic(d, n, DIM, r).r_rms(n) / HYPERBOLIC_KAPPA_MIN.sqrt();
            if !next.is_finite() || next <= 0.0 {
                break;
            }
            let moved = (next - r).abs() / r;
            r = next;
            if moved < 1e-3 {
                break;
            }
        }
        r.max(dm / 20.0)
    };

    // Residual at the two ends of the search window, in κ.
    let (r_cap, r_floor) = (norm(cap), norm(dm / 20.0));
    let (k_cap, k_floor) = (kappa_at(cap), kappa_at(dm / 20.0));
    let slope = (r_floor.log10() - r_cap.log10()) / (k_floor.log10() - k_cap.log10());
    // Local slope where the constraint binds: κ_cap → 3·κ_cap, i.e. r = cap → cap/√3.
    let r_near = norm(cap / 3.0_f64.sqrt());
    let slope_at_cap = (r_near.log10() - r_cap.log10()) / 3.0_f64.log10();

    let hv = detect_hyperbolic(d, n);
    // The δ estimate names a curvature but no radius-fitting procedure, so its
    // κ is gauged on the hyperbolic configuration that curvature implies.
    let kappa_gromov = if hv.saturated_delta > 1e-12 {
        let k = ((1.0 + 2.0_f64.sqrt()).ln() / hv.saturated_delta).powi(2);
        kappa_at(1.0 / k.sqrt())
    } else {
        f64::INFINITY
    };

    // Coarse sweep, plus a dense band around r* so a narrow interior minimum is
    // drawn rather than straddled.
    let mut rs: Vec<f64> = (0..N_GRID)
        .map(|i| {
            let t = i as f64 / (N_GRID - 1) as f64;
            R_LO_FRAC * (R_HI_FRAC / R_LO_FRAC).powf(t) * dm
        })
        .collect();
    for i in 0..41 {
        let t = i as f64 / 40.0;
        rs.push(r_star * 0.75_f64.powf(1.0 - 2.0 * t));
    }
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let curve: Vec<(f64, f64)> = rs
        .iter()
        .map(|&r| {
            (
                kappa_at(r).max(1e-300).log10(),
                norm(r).max(10f64.powf(LOG_FLOOR)).log10(),
            )
        })
        .collect();

    Case {
        name: fx.name,
        color: RGBColor(fx.color.0, fx.color.1, fx.color.2),
        d_max: dm,
        d_rms: dr,
        gromov_hyperbolic: hv.is_hyperbolic,
        kappa_gromov,
        slope,
        leverage: r_floor / r_cap,
        slope_at_cap,
        curve,
        fit_x: (dr / r_star).powi(2).log10(),
        fit_y: norm(r_star).max(10f64.powf(LOG_FLOOR)).log10(),
        kappa_star: (dr / r_star).powi(2),
        pinned,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = CommonArgs::parse(&[])?;
    std::fs::create_dir_all("plots")?;

    // The ten thesis datasets.  The exact-H² control is deliberately left out:
    // the plot is about what the fit does on real data, not about the reference
    // case where the model is true by construction.
    let mut fixtures = common::thesis(args.n, args.seed, &args.data_root);
    if args.all {
        // Skip `controls()[0]`, the exact-H² control.
        fixtures.extend(common::controls(args.n, args.seed).into_iter().skip(1));
    }

    let cases: Vec<Case> = map_parallel(&fixtures, build_case);

    // ── Console report ──────────────────────────────────────────────
    println!(
        "\n{:<20}{:>9}{:>9}{:>11}{:>12}{:>13}{:>9}{:>13}{:>8}{:>11}{:>12}  gromov",
        "dataset",
        "d_max",
        "d_rms",
        "drms/dmax",
        "kappa*",
        "residual*",
        "pinned",
        "kappa_gromov",
        "slope",
        "leverage",
        "slope@cap",
    );
    println!("{}", "-".repeat(157));
    for c in &cases {
        println!(
            "{:<20}{:>9.4}{:>9.4}{:>11.4}{:>12.4}{:>13.3e}{:>9}{:>13.4}{:>8.3}{:>11.2e}{:>12.3}  {}",
            c.name,
            c.d_max,
            c.d_rms,
            c.d_rms / c.d_max,
            c.kappa_star,
            10f64.powf(c.fit_y),
            c.pinned,
            c.kappa_gromov,
            c.slope,
            c.leverage,
            c.slope_at_cap,
            if c.gromov_hyperbolic { "hyperbolic" } else { "-" },
        );
    }
    println!(
        "\n  kappa*        = curvature the Wilson fit reports (pinned => it is the cap position)"
    );
    println!("  kappa_gromov  = curvature from the saturated Gromov delta; only USED where gromov=hyperbolic");
    println!(
        "  slope         = dlog(residual)/dlog(kappa) across the window: 1 = residual tracks the"
    );
    println!("                  curvature mismatch (exactly flat data); ~0 = curvature explains none of it");
    println!("  leverage      = R(kappa_floor)/R(kappa_cap): total swing of the residual over the window");
    println!("  slope@cap     = local dlog(R)/dlog(kappa) at the cap (kappa_cap -> 3*kappa_cap):");
    println!(
        "                  how hard the objective is still pushing flat-ward where the bound binds"
    );
    let pinned: Vec<f64> = cases
        .iter()
        .filter(|c| c.pinned)
        .map(|c| c.kappa_star)
        .collect();
    let (klo, khi) = (
        pinned.iter().cloned().fold(f64::INFINITY, f64::min),
        pinned.iter().cloned().fold(0.0_f64, f64::max),
    );
    println!(
        "\n{} of {} datasets pinned at the cap; their reported kappa spans {:.4} .. {:.4} ({:.1}x)",
        pinned.len(),
        cases.len(),
        klo,
        khi,
        khi / klo,
    );

    // ── Plots ───────────────────────────────────────────────────────
    plot(
        "plots/residual_vs_kappa.png",
        "Hyperbolic signature residual vs model curvature (full sweep)",
        &cases,
        (-2.4, 2.4),
        (LOG_FLOOR - 0.3, 5.5),
        (klo.log10(), khi.log10()),
        SeriesLabelPosition::UpperLeft,
    )?;
    plot(
        "plots/residual_vs_kappa_caps.png",
        "Zoom: the objective where the cap binds (kappa_min = 0.01)",
        &cases,
        (-2.4, -0.2),
        (-4.4, 0.2),
        (klo.log10(), khi.log10()),
        // The `grid 2D` curve runs through the lower left and every other
        // curve sits in the top band, leaving the lower right clear.
        SeriesLabelPosition::LowerRight,
    )?;
    println!("\nWrote 2 plots to plots/");
    Ok(())
}

fn plot(
    path: &str,
    caption: &str,
    cases: &[Case],
    x_range: (f64, f64),
    y_range: (f64, f64),
    cap_band: (f64, f64),
    legend_pos: SeriesLabelPosition,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (1100, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(58)
        .y_label_area_size(76)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

    chart
        .configure_mesh()
        .x_desc("log10( kappa_model = |K| * R_rms^2 )    (flat  <--   -->  more curved)")
        .y_desc("log10( sum |lambda_res| / (n * d_max^2) )")
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    // Band spanning every pinned dataset's cap position.
    chart.draw_series(std::iter::once(Rectangle::new(
        [(cap_band.0, y_range.0), (cap_band.1, y_range.1)],
        RGBColor(120, 150, 200).mix(0.10).filled(),
    )))?;
    for bx in [cap_band.0, cap_band.1] {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(bx, y_range.0), (bx, y_range.1)],
            RGBColor(80, 110, 170).mix(0.55).stroke_width(1),
        )))?;
    }
    for case in cases {
        chart
            .draw_series(LineSeries::new(
                case.curve.iter().cloned(),
                case.color.stroke_width(2),
            ))?
            .label(case.name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], case.color.stroke_width(3))
            });

        // Where the production fit lands — drawn only when it is actually
        // inside the window.  The caps zoom spans just the pinned datasets'
        // cap positions, so an unpinned fit like `tree 2D` (κ* = 136) would
        // otherwise be clamped onto the frame edge and read as a data point
        // sitting at the boundary.
        let in_view = (x_range.0..=x_range.1).contains(&case.fit_x)
            && (y_range.0..=y_range.1).contains(&case.fit_y);
        if in_view {
            chart.draw_series(std::iter::once(Circle::new(
                (case.fit_x, case.fit_y),
                5,
                case.color.filled(),
            )))?;
        }
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.88))
        .border_style(BLACK.mix(0.35))
        .label_font(("sans-serif", 14))
        .position(legend_pos)
        .draw()?;

    root.present()?;
    Ok(())
}
