//! The two knobs of the spherical gate, plotted against what they cost and buy.
//!
//! `detect_geometry` calls data spherical when
//! `residual_normalised < SPHERICAL_RESIDUAL_MAX && !at_upper_bound`, and the
//! second condition is really a third knob in disguise: the search window's
//! flat-ward edge is `d_max / SPHERICAL_ANGULAR_MIN`, so loosening `A` changes
//! which fits are pinned as well as where they land.  The two constants
//! interact, and neither can be chosen alone.
//!
//! **Panel 1 — where the ceiling on the threshold comes from.**  Every
//! dataset's normalised residual against `A`.  A dataset only constrains the
//! threshold if it reaches the residual test at all, i.e. if its fit lands
//! *interior*; pinned fits are rejected by `at_upper_bound` first and are drawn
//! faded.  Two ceilings are drawn:
//!
//!   - **interior ceiling** — the lowest-scoring non-spherical dataset that
//!     lands interior.  Raise the threshold above this and that dataset is
//!     labelled spherical.
//!   - **conservative ceiling** — the lowest-scoring non-spherical dataset of
//!     any kind, pinned or not.  Staying under this is safe even if the pin
//!     test fails, which the `SPHERICAL_RESIDUAL_MAX` docs argue it can, since
//!     which edge a flat objective settles on is decided by noise.
//!
//! The usable band is between the spherical fixture's score (below it even a
//! perfect sphere is rejected) and whichever ceiling you choose to trust.
//!
//! **Panel 2 — what a threshold buys.**  Noise tolerance against threshold, by
//! inverting the noise sweep (see `spherical_noise_sweep.rs` for the models).
//! Read a threshold off panel 1's ceiling, then read the σ it affords here.
//!
//! Run with:
//!
//! ```bash
//! cargo run --release -p fitting-core --features plot-examples \
//!     --example spherical_gate_tradeoff
//! ```

use std::error::Error;
use std::f64::consts::PI;

use fitting_core::curvature_detection::{
    fit_spherical, spherical_residual_at, SPHERICAL_ANGULAR_MIN, SPHERICAL_RESIDUAL_MAX,
};
use fitting_core::data::{load_fashion_mnist, load_mnist, load_pbmc, load_wordnet_mammals};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::rng::Rng;
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere, DataPoints,
};
use plotters::prelude::*;

const N_SYNTH: usize = 300;
const N_REAL: usize = 500;
const SEED: u64 = 42;
const DIM: usize = 2;
const N_GRID: usize = 30; // matches fit_spherical

/// Candidate values for `SPHERICAL_ANGULAR_MIN`, current value first.
const A_VALUES: [f64; 7] = [2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.5];

/// The genuinely spherical fixture; everything else is an impostor.
const SPHERE_FIXTURE: &str = "S2 (exact)";

// ── Noise sweep (panel 2) ───────────────────────────────────────────────────

const NOISE_N: usize = 300;
const NOISE_SEEDS: u64 = 3;
const SIGMAS: [f64; 13] = [
    0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.04, 0.08, 0.16,
];

// ── Search, copied from signature.rs so the window bound can vary ───────────

fn golden_section(a: f64, b: f64, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
    let phi = 0.6180339887498949_f64;
    let mut a = a;
    let mut b = b;
    let mut r1 = a + (1.0 - phi) * (b - a);
    let mut r2 = a + phi * (b - a);
    let mut f1 = f(r1);
    let mut f2 = f(r2);
    for _ in 0..40 {
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
        if (b - a) / (a + b).max(1e-12) < 1e-5 {
            break;
        }
    }
    let r_star = 0.5 * (a + b);
    let f_star = f(r_star);
    if f_star < f1 && f_star < f2 {
        (r_star, f_star)
    } else if f1 < f2 {
        (r1, f1)
    } else {
        (r2, f2)
    }
}

fn minimise_log_spaced(
    lo: f64,
    hi: f64,
    n_grid: usize,
    f: &mut dyn FnMut(f64) -> f64,
) -> (f64, bool) {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    let step = (log_hi - log_lo) / (n_grid - 1) as f64;
    let grid_r: Vec<f64> = (0..n_grid)
        .map(|i| (log_lo + i as f64 * step).exp())
        .collect();
    let grid_res: Vec<f64> = grid_r.iter().map(|&r| f(r)).collect();

    let mut local_min_indices: Vec<usize> = Vec::new();
    if n_grid >= 2 && grid_res[0] < grid_res[1] {
        local_min_indices.push(0);
    }
    for i in 1..n_grid.saturating_sub(1) {
        if grid_res[i] < grid_res[i - 1] && grid_res[i] < grid_res[i + 1] {
            local_min_indices.push(i);
        }
    }
    if n_grid >= 2 && grid_res[n_grid - 1] < grid_res[n_grid - 2] {
        local_min_indices.push(n_grid - 1);
    }
    if local_min_indices.is_empty() {
        local_min_indices.push(
            grid_res
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
        );
    }

    let mut best_r = grid_r[local_min_indices[0]];
    let mut best_res = grid_res[local_min_indices[0]];
    let mut best_i = local_min_indices[0];
    for &i in &local_min_indices {
        let a = grid_r[i.saturating_sub(1)];
        let b = grid_r[(i + 1).min(n_grid - 1)];
        let (r, res) = if a < b {
            golden_section(a, b, f)
        } else {
            (grid_r[i], grid_res[i])
        };
        if res < best_res {
            best_res = res;
            best_r = r;
            best_i = i;
        }
    }
    (best_r, best_i == n_grid - 1)
}

// ── Panel 1 data ────────────────────────────────────────────────────────────

struct Case {
    name: &'static str,
    color: RGBColor,
    /// Per `A_VALUES` entry: `(residual_normalised, at_upper_bound)`.
    rows: Vec<(f64, bool)>,
}

fn distances_for(data: &DataPoints) -> Vec<f64> {
    if !data.distances.is_empty() {
        data.distances.clone()
    } else {
        compute_euclidean_distance_matrix(&data.x, data.n_points, data.ambient_dim)
    }
}

fn analyse(name: &'static str, color: RGBColor, data: &DataPoints) -> Case {
    let n = data.n_points;
    let d = distances_for(data);
    let d_max = d.iter().cloned().fold(0.0_f64, f64::max);
    let scale = n as f64 * d_max * d_max;

    let rows = A_VALUES
        .iter()
        .map(|&a| {
            let mut objective = |r: f64| -> f64 { spherical_residual_at(&d, n, DIM, r) };
            let (r_star, at_upper) =
                minimise_log_spaced(d_max / PI, d_max / a, N_GRID, &mut objective);
            (spherical_residual_at(&d, n, DIM, r_star) / scale, at_upper)
        })
        .collect();

    println!("  analysed {name}");
    Case { name, color, rows }
}

/// Lowest impostor score at each `A`.  `interior_only` skips fits pinned at the
/// flat-ward bound, which `at_upper_bound` rejects before the residual test.
fn ceiling(cases: &[Case], interior_only: bool) -> Vec<f64> {
    (0..A_VALUES.len())
        .map(|i| {
            cases
                .iter()
                .filter(|c| c.name != SPHERE_FIXTURE)
                .filter(|c| !interior_only || !c.rows[i].1)
                .map(|c| c.rows[i].0)
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

// ── Panel 2 data ────────────────────────────────────────────────────────────

/// Mean normalised residual of the noisy sphere at each σ, for one noise model.
/// `geodesic_corrected` displaces the points in `R³` and maps the resulting
/// chords back to arcs; otherwise the geodesic matrix itself is jittered.
fn noise_curve(geodesic_corrected: bool) -> Vec<f64> {
    SIGMAS
        .iter()
        .map(|&sigma| {
            let mut total = 0.0;
            for s in 0..NOISE_SEEDS {
                let data = generate_uniform_sphere(NOISE_N, SEED + s);
                let mut rng = Rng::new(1000 + s);
                let d = if geodesic_corrected {
                    let mut x = data.x.clone();
                    for v in x.iter_mut() {
                        *v += sigma * rng.normal();
                    }
                    let mut d = compute_euclidean_distance_matrix(&x, NOISE_N, data.ambient_dim);
                    for v in d.iter_mut() {
                        *v = 2.0 * (*v / 2.0).clamp(-1.0, 1.0).asin();
                    }
                    d
                } else {
                    let mut d = data.distances.clone();
                    for i in 0..NOISE_N {
                        for j in (i + 1)..NOISE_N {
                            let p = (d[i * NOISE_N + j] * (1.0 + sigma * rng.normal())).max(0.0);
                            d[i * NOISE_N + j] = p;
                            d[j * NOISE_N + i] = p;
                        }
                    }
                    d
                };
                total += fit_spherical(&d, NOISE_N, DIM).residual_normalised;
            }
            total / NOISE_SEEDS as f64
        })
        .collect()
}

/// Invert a noise curve: the σ at which the residual reaches `threshold`,
/// linearly interpolated between the bracketing samples.  `None` if the curve
/// is already above `threshold` at σ = 0.
fn sigma_at(curve: &[f64], threshold: f64) -> Option<f64> {
    if curve[0] >= threshold {
        return None;
    }
    for i in 1..curve.len() {
        if curve[i] >= threshold {
            let (s0, r0) = (SIGMAS[i - 1], curve[i - 1]);
            let (s1, r1) = (SIGMAS[i], curve[i]);
            return Some(s0 + (threshold - r0) * (s1 - s0) / (r1 - r0));
        }
    }
    None
}

// ── Plot ────────────────────────────────────────────────────────────────────

const OUT: &str = "plots/spherical_gate_tradeoff.png";

fn main() -> Result<(), Box<dyn Error>> {
    let data_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "www/public/data".to_string());

    println!("Sweeping A over {} values...", A_VALUES.len());
    let mut cases = vec![
        analyse(
            SPHERE_FIXTURE,
            GREEN,
            &generate_uniform_sphere(N_SYNTH, SEED),
        ),
        analyse(
            "E2 (flat)",
            BLUE,
            &generate_uniform_ball_2d(N_SYNTH, SEED, 5.0),
        ),
        analyse(
            "H2",
            RGBColor(148, 0, 211),
            &generate_uniform_hyperbolic(N_SYNTH, SEED, 5.0),
        ),
    ];
    let real: Vec<(&'static str, RGBColor, Result<DataPoints, _>)> = vec![
        (
            "mnist",
            RED,
            load_mnist(&format!("{data_root}/mnist"), N_REAL),
        ),
        (
            "fashion_mnist",
            RGBColor(255, 140, 0),
            load_fashion_mnist(&format!("{data_root}/fashion-mnist"), N_REAL),
        ),
        (
            "pbmc",
            RGBColor(0, 150, 136),
            load_pbmc(&format!("{data_root}/pbmc"), N_REAL),
        ),
        (
            "wordnet_mammals",
            RGBColor(121, 85, 72),
            load_wordnet_mammals(&format!("{data_root}/wordnet"), N_REAL),
        ),
    ];
    for (name, color, loaded) in real {
        match loaded {
            Ok(data) => cases.push(analyse(name, color, &data)),
            Err(e) => println!("  skipping {name}: {e}"),
        }
    }

    let ceil_interior = ceiling(&cases, true);
    let ceil_all = ceiling(&cases, false);

    println!("Running the noise sweep...");
    let metric = noise_curve(false);
    let ambient_geo = noise_curve(true);

    // ── Console summary: the numbers the plot shows ──
    println!("\n     A    max safe T (interior)   max safe T (conservative)   binding dataset");
    for (i, a) in A_VALUES.iter().enumerate() {
        let binding = cases
            .iter()
            .filter(|c| c.name != SPHERE_FIXTURE)
            .min_by(|x, y| x.rows[i].0.partial_cmp(&y.rows[i].0).unwrap())
            .map(|c| c.name)
            .unwrap_or("-");
        println!(
            "  {:4.2}          {:11.3e}               {:11.3e}   {}",
            a, ceil_interior[i], ceil_all[i], binding
        );
    }
    println!("\n  threshold T    metric sigma    ambient-geodesic sigma");
    for t in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1] {
        let f = |v: Option<f64>| match v {
            Some(s) => format!("{:.3}%", 100.0 * s),
            None => "n/a".to_string(),
        };
        println!(
            "  {:11.0e}    {:12}    {}",
            t,
            f(sigma_at(&metric, t)),
            f(sigma_at(&ambient_geo, t))
        );
    }

    // ── Figure ──
    let root = BitMapBackend::new(OUT, (1320, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let (top, bottom) = root.split_vertically(560);
    // The panel-1 legend needs ~12 entries and every corner of the plot area is
    // occupied by a curve, so it gets its own strip rather than covering data.
    let (top_chart, top_legend) = top.split_horizontally(990);

    // Panel 1: residual vs A.
    let mut c1 = ChartBuilder::on(&top_chart)
        .caption(
            "Loosening the window: which datasets reach the residual test, and how high it may sit",
            ("sans-serif", 21),
        )
        .margin(18)
        .x_label_area_size(52)
        .y_label_area_size(78)
        .build_cartesian_2d(0.4f64..2.65f64, -8.5f64..0.0f64)?;
    c1.configure_mesh()
        .x_desc("SPHERICAL_ANGULAR_MIN  A     (<-- looser window          stricter -->)")
        .y_desc("log10 residual / (n d_max^2)")
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    // Two regimes, shaded.  Below the conservative ceiling every impostor is
    // rejected on residual alone; between the two ceilings the flat data is
    // only rejected because its fit happens to be pinned, which is the test
    // `SPHERICAL_RESIDUAL_MAX`'s own docs call noise-driven.
    let mut safe: Vec<(f64, f64)> = vec![(A_VALUES[A_VALUES.len() - 1], -8.5), (A_VALUES[0], -8.5)];
    safe.extend(
        A_VALUES
            .iter()
            .enumerate()
            .map(|(i, &a)| (a, ceil_all[i].log10())),
    );
    c1.draw_series(std::iter::once(Polygon::new(safe, GREEN.mix(0.13))))?;

    let mut pin_dependent: Vec<(f64, f64)> = A_VALUES
        .iter()
        .enumerate()
        .map(|(i, &a)| (a, ceil_all[i].log10()))
        .collect();
    pin_dependent.extend(
        A_VALUES
            .iter()
            .enumerate()
            .rev()
            .map(|(i, &a)| (a, ceil_interior[i].log10())),
    );
    c1.draw_series(std::iter::once(Polygon::new(
        pin_dependent,
        RGBColor(255, 193, 7).mix(0.22),
    )))?;

    for case in &cases {
        let pts: Vec<(f64, f64)> = A_VALUES
            .iter()
            .enumerate()
            .map(|(i, &a)| (a, case.rows[i].0.log10()))
            .collect();
        c1.draw_series(LineSeries::new(
            pts.clone(),
            case.color.mix(0.45).stroke_width(2),
        ))?;
        // Filled circle = interior (constrains the threshold); open = pinned.
        for (i, p) in pts.iter().enumerate() {
            if case.rows[i].1 {
                c1.draw_series(std::iter::once(Circle::new(
                    *p,
                    4,
                    case.color.mix(0.5).stroke_width(1),
                )))?;
            } else {
                c1.draw_series(std::iter::once(Circle::new(*p, 5, case.color.filled())))?;
            }
        }
    }

    for (vals, style) in [
        (&ceil_interior, BLACK.stroke_width(3)),
        (&ceil_all, BLACK.mix(0.45).stroke_width(2)),
    ] {
        let pts: Vec<(f64, f64)> = A_VALUES
            .iter()
            .enumerate()
            .map(|(i, &a)| (a, vals[i].log10()))
            .collect();
        c1.draw_series(LineSeries::new(pts, style))?;
    }

    for (t, tag, col) in [
        (SPHERICAL_RESIDUAL_MAX, "current T", RGBColor(200, 0, 0)),
        (1e-2, "T = 1e-2", RGBColor(0, 120, 200)),
    ] {
        c1.draw_series(std::iter::once(PathElement::new(
            vec![(0.4, t.log10()), (2.65, t.log10())],
            col.mix(0.8).stroke_width(2),
        )))?;
        c1.draw_series(std::iter::once(Text::new(
            tag,
            (0.45, t.log10() + 0.12),
            ("sans-serif", 15).into_font().color(&col),
        )))?;
    }
    c1.draw_series(std::iter::once(PathElement::new(
        vec![(SPHERICAL_ANGULAR_MIN, -8.5), (SPHERICAL_ANGULAR_MIN, 0.0)],
        BLACK.mix(0.35).stroke_width(1),
    )))?;
    // Legend, drawn in its own strip in backend pixels so it never overlaps a
    // curve.  `Swatch` entries are the two shaded regimes, `Line` entries the
    // dataset curves and the reference levels.
    enum Mark {
        Line(RGBColor, u32),
        Swatch(RGBColor, f64),
    }
    let mut entries: Vec<(String, Mark)> = vec![(
        "datasets (filled = interior,".to_string(),
        Mark::Line(WHITE, 0),
    )];
    entries.push(("hollow = pinned)".to_string(), Mark::Line(WHITE, 0)));
    for case in &cases {
        entries.push((case.name.to_string(), Mark::Line(case.color, 3)));
    }
    entries.push(("".to_string(), Mark::Line(WHITE, 0)));
    entries.push((
        "hard ceiling (lowest interior".to_string(),
        Mark::Line(BLACK, 3),
    ));
    entries.push(("  impostor: admits pbmc)".to_string(), Mark::Line(WHITE, 0)));
    entries.push((
        "safe ceiling (lowest impostor".to_string(),
        Mark::Line(RGBColor(140, 140, 140), 2),
    ));
    entries.push((
        "  of any kind: admits E2)".to_string(),
        Mark::Line(WHITE, 0),
    ));
    entries.push(("".to_string(), Mark::Line(WHITE, 0)));
    entries.push((
        "safe on residual alone".to_string(),
        Mark::Swatch(GREEN, 0.13),
    ));
    entries.push((
        "needs the pin test".to_string(),
        Mark::Swatch(RGBColor(255, 193, 7), 0.22),
    ));
    entries.push(("".to_string(), Mark::Line(WHITE, 0)));
    entries.push((
        format!("current T = {SPHERICAL_RESIDUAL_MAX:.0e}"),
        Mark::Line(RGBColor(200, 0, 0), 2),
    ));
    entries.push((
        "proposed T = 1e-2".to_string(),
        Mark::Line(RGBColor(0, 120, 200), 2),
    ));

    let mut y = 46i32;
    for (label, mark) in &entries {
        match mark {
            Mark::Line(col, w) if *w > 0 => {
                top_legend.draw(&PathElement::new(
                    vec![(12, y), (40, y)],
                    col.stroke_width(*w),
                ))?;
            }
            Mark::Swatch(col, alpha) => {
                top_legend.draw(&Rectangle::new(
                    [(12, y - 6), (40, y + 6)],
                    col.mix(*alpha).filled(),
                ))?;
                top_legend.draw(&Rectangle::new(
                    [(12, y - 6), (40, y + 6)],
                    BLACK.mix(0.3).stroke_width(1),
                ))?;
            }
            _ => {}
        }
        if !label.is_empty() {
            top_legend.draw(&Text::new(
                label.as_str(),
                (48, y - 7),
                ("sans-serif", 14).into_font().color(&BLACK),
            ))?;
        }
        y += 24;
    }

    // Panel 2: noise tolerance vs threshold.
    let t_lo = -4.0f64;
    let t_hi = -0.5f64;
    let mut c2 = ChartBuilder::on(&bottom)
        .caption(
            "What the threshold buys: tolerable noise on the spherical fixture",
            ("sans-serif", 21),
        )
        .margin(18)
        .x_label_area_size(52)
        .y_label_area_size(78)
        .build_cartesian_2d(t_lo..t_hi, -4.0f64..-0.6f64)?;
    c2.configure_mesh()
        .x_desc("log10 SPHERICAL_RESIDUAL_MAX")
        .y_desc("log10 tolerable sigma")
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    for (curve, tag, col) in [
        (
            &metric,
            "metric noise (jittered geodesics)",
            RGBColor(200, 0, 0),
        ),
        (
            &ambient_geo,
            "ambient noise, geodesic-corrected",
            RGBColor(0, 120, 200),
        ),
    ] {
        let pts: Vec<(f64, f64)> = (0..=140)
            .map(|k| t_lo + (t_hi - t_lo) * k as f64 / 140.0)
            .filter_map(|lt| sigma_at(curve, 10f64.powf(lt)).map(|s| (lt, s.log10())))
            .collect();
        c2.draw_series(LineSeries::new(pts, col.stroke_width(3)))?
            .label(tag)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], col.stroke_width(3)));
    }

    // Labels staggered in y so the two ceilings, which sit close together on a
    // log axis, stay readable.
    for (t, tag, col, label_y) in [
        (
            SPHERICAL_RESIDUAL_MAX,
            "current T",
            RGBColor(120, 120, 120),
            -3.9,
        ),
        (
            ceil_all[0],
            "safe ceiling @ A=2.5",
            RGBColor(80, 80, 80),
            -3.9,
        ),
        (ceil_interior[0], "hard ceiling (pbmc)", BLACK, -3.65),
    ] {
        c2.draw_series(std::iter::once(PathElement::new(
            vec![(t.log10(), -4.0), (t.log10(), -0.6)],
            col.mix(0.7).stroke_width(2),
        )))?;
        c2.draw_series(std::iter::once(Text::new(
            tag,
            (t.log10() + 0.03, label_y),
            ("sans-serif", 15).into_font().color(&col),
        )))?;
    }
    c2.configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 15))
        .draw()?;

    root.present()?;
    println!("\nWrote {OUT}");
    Ok(())
}
