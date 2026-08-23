//! The hyperbolic signature residual against the **dimensionless curvature**
//! `κ = |K|·d_rms²` of the model being tested, instead of against the radius `r`.
//!
//! Since `κ_model = (d_rms/r)²`, sweeping `r` is sweeping `κ`, and the search
//! window `[d_max/20, d_rms/√κ_min]` becomes a `κ` interval.  The flat-ward end
//! is `κ_min = `[`HYPERBOLIC_KAPPA_MIN`] by construction — the same demand for
//! every dataset — which is the point of stating the cap in `κ` rather than as
//! a multiple of `d_max`.  Plotting in this gauge makes two things visible that
//! the `r` axis hides:
//!
//! * an **exact H² manifold** has a sharp interior minimum at its true `κ` — the
//!   Wilson fit is a good radius estimator when the model is actually true;
//! * every **thesis dataset** instead descends monotonically across the whole
//!   window, so the minimiser stops at the cap and the reported curvature is
//!   just `κ_min` — the bound, not a measurement.
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

use fitting_core::curvature_detection::{
    detect_hyperbolic, hyperbolic_residual_at, HYPERBOLIC_KAPPA_MIN,
};
use fitting_core::data::{load_fashion_mnist, load_mnist, load_pbmc, load_wordnet_mammals};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::synthetic_data::{
    generate_hd_antipodal_clusters, generate_hd_hyperbolic_shells, generate_hd_sphere,
    generate_hd_tree, generate_uniform_grid, generate_uniform_hyperbolic, DataPoints,
};
use plotters::prelude::*;

const N: usize = 400;
const SEED: u64 = 42;
const DIM: usize = 2;
/// Ambient dimension the curved synthetics are lifted into, matching
/// `optimizer::Dataset::load_synthetic`.
const HD: usize = 10;
const N_GRID: usize = 70;
/// Radius sweep as a multiple of `d_max`, spanning well past both search bounds.
const R_LO_FRAC: f64 = 1.0 / 50.0;
const R_HI_FRAC: f64 = 50.0;
/// Floor for the log residual axis, so the near-exact control stays finite.
const LOG_FLOOR: f64 = -8.0;

fn distances_for(d: &DataPoints) -> Vec<f64> {
    if !d.distances.is_empty() {
        d.distances.clone()
    } else {
        compute_euclidean_distance_matrix(&d.x, d.n_points, d.ambient_dim)
    }
}

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}

fn d_rms_of(d: &[f64], n: usize) -> f64 {
    let s: f64 = d.iter().map(|x| x * x).sum();
    (s / (n as f64 * (n as f64 - 1.0))).sqrt()
}

// ── The production minimiser, mirrored so the plotted r* is the real one ────

fn golden(a: f64, b: f64, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
    let phi = 0.618_033_988_749_894_9_f64;
    let (mut a, mut b) = (a, b);
    let mut r1 = a + (1.0 - phi) * (b - a);
    let mut r2 = a + phi * (b - a);
    let (mut f1, mut f2) = (f(r1), f(r2));
    for _ in 0..50 {
        if f1 < f2 {
            b = r2;
            r2 = r1;
            f2 = f1;
            r1 = a + (1.0 - phi) * (b - a);
            f1 = f(r1);
        } else {
            a = r1;
            r1 = r2;
            f1 = f2;
            r2 = a + phi * (b - a);
            f2 = f(r2);
        }
        if (b - a) / (a + b).max(1e-12) < 1e-7 {
            break;
        }
    }
    if f1 < f2 {
        (r1, f1)
    } else {
        (r2, f2)
    }
}

/// Coarse log grid over the window, golden-refine every local minimum, keep the
/// best — `signature::minimise_log_spaced`.  Returns `(r*, value, pinned)`.
fn fit_window(lo: f64, hi: f64, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64, bool) {
    const G: usize = 30;
    let step = (hi.ln() - lo.ln()) / (G - 1) as f64;
    let gr: Vec<f64> = (0..G).map(|i| (lo.ln() + i as f64 * step).exp()).collect();
    let gv: Vec<f64> = gr.iter().map(|&r| f(r)).collect();
    let mut mins: Vec<usize> = Vec::new();
    if gv[0] < gv[1] {
        mins.push(0);
    }
    for i in 1..G - 1 {
        if gv[i] < gv[i - 1] && gv[i] < gv[i + 1] {
            mins.push(i);
        }
    }
    if gv[G - 1] < gv[G - 2] {
        mins.push(G - 1);
    }
    if mins.is_empty() {
        mins.push(0);
    }
    let (mut br, mut bv, mut bi) = (gr[mins[0]], gv[mins[0]], mins[0]);
    for &i in &mins {
        let a = gr[i.saturating_sub(1)];
        let b = gr[(i + 1).min(G - 1)];
        let (r, v) = if a < b {
            golden(a, b, f)
        } else {
            (gr[i], gv[i])
        };
        if v < bv {
            bv = v;
            br = r;
            bi = i;
        }
    }
    (br, bv, bi == G - 1)
}

struct Case {
    name: &'static str,
    color: RGBColor,
    /// Drawn heavier: the exact-manifold control.
    control: bool,
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

fn build_case(name: &'static str, color: RGBColor, control: bool, d: &[f64], n: usize) -> Case {
    let dm = d_max_of(d);
    let dr = d_rms_of(d, n);
    let gauge = n as f64 * dm * dm;
    let norm = |r: f64| hyperbolic_residual_at(d, n, DIM, r) / gauge;

    // Production window: [d_max/20, d_rms/√κ_min] with κ_min = HYPERBOLIC_KAPPA_MIN.
    let cap = dr / HYPERBOLIC_KAPPA_MIN.sqrt();
    let mut obj = |r: f64| hyperbolic_residual_at(d, n, DIM, r);
    let (r_star, _, pinned) = fit_window(dm / 20.0, cap, &mut obj);

    // Residual at the two ends of the search window, in κ.
    let (r_cap, r_floor) = (norm(cap), norm(dm / 20.0));
    let (k_cap, k_floor) = ((dr / cap).powi(2), (dr / (dm / 20.0)).powi(2));
    let slope = (r_floor.log10() - r_cap.log10()) / (k_floor.log10() - k_cap.log10());
    // Local slope where the constraint binds: κ_cap → 3·κ_cap, i.e. r = cap → cap/√3.
    let r_near = norm(cap / 3.0_f64.sqrt());
    let slope_at_cap = (r_near.log10() - r_cap.log10()) / 3.0_f64.log10();

    let hv = detect_hyperbolic(d, n);
    let kappa_gromov = if hv.saturated_delta > 1e-12 {
        ((1.0 + 2.0_f64.sqrt()).ln() / hv.saturated_delta).powi(2) * dr * dr
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
                (dr / r).powi(2).log10(),
                norm(r).max(10f64.powf(LOG_FLOOR)).log10(),
            )
        })
        .collect();

    Case {
        name,
        color,
        control,
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
    let data_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "www/public/data".to_string());

    std::fs::create_dir_all("plots")?;

    let mut cases: Vec<Case> = Vec::new();

    // Control: an exact H² manifold of curvature radius 1, so its true κ is d_rms².
    let h2 = generate_uniform_hyperbolic(N, SEED, 5.0);
    let dh = distances_for(&h2);
    let true_kappa = d_rms_of(&dh, N).powi(2);
    cases.push(build_case("H2 exact (control)", BLACK, true, &dh, N));

    let synth: [(&'static str, RGBColor, DataPoints); 5] = [
        ("tree", RGBColor(214, 39, 40), generate_hd_tree(N, HD, SEED)),
        (
            "hyperbolic_shells",
            RGBColor(255, 127, 14),
            generate_hd_hyperbolic_shells(N, HD, SEED),
        ),
        (
            "sphere",
            RGBColor(44, 160, 44),
            generate_hd_sphere(N, HD, SEED),
        ),
        (
            "antipodal_clusters",
            RGBColor(23, 190, 207),
            generate_hd_antipodal_clusters(N, HD, SEED),
        ),
        (
            "grid",
            RGBColor(148, 103, 189),
            generate_uniform_grid(N, SEED),
        ),
    ];
    for (name, color, dp) in synth {
        let d = distances_for(&dp);
        let n = dp.n_points;
        cases.push(build_case(name, color, false, &d, n));
    }

    let reals: [(&'static str, RGBColor, Result<DataPoints, String>); 4] = [
        (
            "mnist",
            RGBColor(31, 119, 180),
            load_mnist(&format!("{data_root}/mnist"), N),
        ),
        (
            "fashion_mnist",
            RGBColor(227, 119, 194),
            load_fashion_mnist(&format!("{data_root}/fashion-mnist"), N),
        ),
        (
            "pbmc",
            RGBColor(140, 86, 75),
            load_pbmc(&format!("{data_root}/pbmc"), N),
        ),
        (
            "wordnet_mammals",
            RGBColor(127, 127, 127),
            load_wordnet_mammals(&format!("{data_root}/wordnet"), N),
        ),
    ];
    for (name, color, loaded) in reals {
        match loaded {
            Ok(dp) => {
                let d = distances_for(&dp);
                let n = dp.n_points;
                cases.push(build_case(name, color, false, &d, n));
            }
            Err(e) => eprintln!("{name} skipped: {e}"),
        }
    }

    // ── Console report ──────────────────────────────────────────────
    println!(
        "\n{:<20}{:>9}{:>9}{:>11}{:>12}{:>13}{:>9}{:>13}{:>8}{:>11}{:>12}  {}",
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
        "gromov"
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
    println!("control's true kappa = {true_kappa:.3}");

    // ── Plots ───────────────────────────────────────────────────────
    plot(
        "plots/residual_vs_kappa.png",
        "Hyperbolic signature residual vs model curvature (full sweep)",
        &cases,
        (-2.4, 2.4),
        (LOG_FLOOR - 0.3, 5.5),
        Some(true_kappa.log10()),
        (klo.log10(), khi.log10()),
        SeriesLabelPosition::LowerLeft,
    )?;
    plot(
        "plots/residual_vs_kappa_caps.png",
        "Zoom: the objective where the cap binds (kappa_min = 0.01)",
        &cases,
        (-2.4, -0.2),
        (-4.4, 0.2),
        None,
        (klo.log10(), khi.log10()),
        // The `grid` curve runs through the lower left here.
        SeriesLabelPosition::UpperRight,
    )?;
    println!("\nWrote 2 plots to plots/");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn plot(
    path: &str,
    caption: &str,
    cases: &[Case],
    x_range: (f64, f64),
    y_range: (f64, f64),
    true_kappa_log: Option<f64>,
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
        .x_desc("log10( kappa_model = |K| * d_rms^2 )    (flat  <--   -->  more curved)")
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
    // With the cap stated as a κ floor the band collapses to one line: every
    // dataset is held to the same κ_min, which is the point of the change.
    let band_label = if cap_band.1 - cap_band.0 < 0.01 {
        format!("cap for every dataset: kappa_min = {HYPERBOLIC_KAPPA_MIN}")
    } else {
        "every cap lands in this band".to_string()
    };
    chart.draw_series(std::iter::once(Text::new(
        band_label,
        (
            cap_band.0 + 0.03,
            y_range.0 + 0.965 * (y_range.1 - y_range.0),
        ),
        ("sans-serif", 14).into_font().color(&RGBColor(60, 90, 150)),
    )))?;

    // The control's true curvature.
    if let Some(tk) = true_kappa_log {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(tk, y_range.0), (tk, y_range.1)],
            BLACK.mix(0.55).stroke_width(2),
        )))?;
        // Sits in the empty strip below every curve and left of the control's
        // dive, so it clears the bundle where the curves converge.
        chart.draw_series(std::iter::once(Text::new(
            "true kappa of the control",
            (tk - 0.78, y_range.0 + 0.36 * (y_range.1 - y_range.0)),
            ("sans-serif", 14).into_font().color(&BLACK.mix(0.75)),
        )))?;
    }

    for case in cases {
        let w = if case.control { 3 } else { 2 };
        chart
            .draw_series(LineSeries::new(
                case.curve.iter().cloned(),
                case.color.stroke_width(w),
            ))?
            .label(if case.pinned {
                format!("{} (pinned)", case.name)
            } else {
                case.name.to_string()
            })
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], case.color.stroke_width(3))
            });

        // Where the production fit lands.
        chart.draw_series(std::iter::once(Circle::new(
            (case.fit_x, case.fit_y),
            5,
            case.color.filled(),
        )))?;
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
