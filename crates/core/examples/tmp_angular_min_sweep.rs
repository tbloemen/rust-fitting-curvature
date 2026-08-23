//! Throwaway: how the spherical gate statistic behaves as
//! `SPHERICAL_ANGULAR_MIN` is loosened.
//!
//! `fit_spherical` hardcodes the constant, so this reimplements its search
//! (same 30-point log grid + golden-section refinement, same objective
//! `Σ|λ_res|`) with the flat-ward bound parameterised by `A`, and reports the
//! gate statistic `Σ|λ_res| / (n · d_max²)` at each `r*`.  This is what
//! `SPHERICAL_RESIDUAL_MAX` is calibrated against; re-run it after changing
//! `SPHERICAL_ANGULAR_MIN` or adding datasets.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_angular_min_sweep`

use std::error::Error;
use std::f64::consts::PI;

use fitting_core::curvature_detection::{
    fit_hyperbolic, spherical_residual_at, SPHERICAL_RESIDUAL_MAX,
};
use fitting_core::data::{load_fashion_mnist, load_mnist, load_pbmc, load_wordnet_mammals};
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere, DataPoints,
};

const N_SYNTH: usize = 300;
const N_REAL: usize = 500;
const SEED: u64 = 42;
const DIM: usize = 2;
const N_GRID: usize = 30; // matches fit_spherical

/// Candidate values for `SPHERICAL_ANGULAR_MIN`, current value first.
const A_VALUES: [f64; 5] = [2.5, 2.0, 1.5, 1.0, 0.5];

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
) -> (f64, f64, bool) {
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
    (best_r, best_res, best_i == n_grid - 1)
}

// ── One (dataset, A) result ─────────────────────────────────────────────────

struct Row {
    a: f64,
    r_over_dmax: f64,
    at_upper: bool,
    gauge: f64,
}

struct Case {
    name: String,
    n: usize,
    d_max: f64,
    rows: Vec<Row>,
    /// Hyperbolic fit in its own (A-independent) window, for reference.
    hyp_r_over_dmax: f64,
    hyp_gauge: f64,
}

fn distances_for(data: &DataPoints) -> Vec<f64> {
    if !data.distances.is_empty() {
        data.distances.clone()
    } else {
        compute_euclidean_distance_matrix(&data.x, data.n_points, data.ambient_dim)
    }
}

fn analyse(name: &str, data: &DataPoints) -> Case {
    let n = data.n_points;
    let d = distances_for(data);
    let d_max = d.iter().cloned().fold(0.0_f64, f64::max);
    let scale = n as f64 * d_max * d_max;

    let mut rows = Vec::new();
    for a in A_VALUES {
        let r_lower = d_max / PI;
        let r_upper = d_max / a;
        let mut objective = |r: f64| -> f64 { spherical_residual_at(&d, n, DIM, r) };
        let (r_star, _, at_upper) = minimise_log_spaced(r_lower, r_upper, N_GRID, &mut objective);
        rows.push(Row {
            a,
            r_over_dmax: r_star / d_max,
            at_upper,
            gauge: spherical_residual_at(&d, n, DIM, r_star) / scale,
        });
    }

    let fh = fit_hyperbolic(&d, n, DIM);

    Case {
        name: name.to_string(),
        n,
        d_max,
        rows,
        hyp_r_over_dmax: fh.radius / d_max,
        hyp_gauge: fh.residual_normalised,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "www/public/data".to_string());

    let mut cases: Vec<Case> = vec![
        analyse(
            "synth_hyperbolic",
            &generate_uniform_hyperbolic(N_SYNTH, SEED, 5.0),
        ),
        analyse(
            "synth_euclidean",
            &generate_uniform_ball_2d(N_SYNTH, SEED, 5.0),
        ),
        analyse("synth_spherical", &generate_uniform_sphere(N_SYNTH, SEED)),
    ];

    // Real datasets are optional: skip cleanly if the data directory is absent.
    let real: Vec<(&str, Result<DataPoints, _>)> = vec![
        ("mnist", load_mnist(&format!("{data_root}/mnist"), N_REAL)),
        (
            "fashion_mnist",
            load_fashion_mnist(&format!("{data_root}/fashion-mnist"), N_REAL),
        ),
        ("pbmc", load_pbmc(&format!("{data_root}/pbmc"), N_REAL)),
        (
            "wordnet_mammals",
            load_wordnet_mammals(&format!("{data_root}/wordnet"), N_REAL),
        ),
    ];
    for (name, loaded) in real {
        match loaded {
            Ok(data) => cases.push(analyse(name, &data)),
            Err(e) => println!("skipping {name}: {e}"),
        }
    }

    for c in &cases {
        println!(
            "\n{}  (n = {}, d_max = {:.4})\n    A     r*/d_max  coverage    at_upper   \
             gauge = e/(n dmax^2)   verdict",
            c.name, c.n, c.d_max
        );
        for r in &c.rows {
            let coverage_deg = (1.0 / r.r_over_dmax) * 180.0 / PI;
            let spherical = !r.at_upper && r.gauge < SPHERICAL_RESIDUAL_MAX;
            println!(
                "  {:4.1}   {:8.4}   {:5.1}deg   {:8}   {:18.3e}   {}",
                r.a,
                r.r_over_dmax,
                coverage_deg,
                if r.at_upper { "yes" } else { "no" },
                r.gauge,
                if spherical { "SPHERICAL" } else { "-" },
            );
        }
        println!(
            "  hyperbolic fit: r*/d_max = {:.4}   gauge = {:.3e}",
            c.hyp_r_over_dmax, c.hyp_gauge
        );
    }

    // Separation: synth_spherical is the only genuinely spherical fixture, so
    // for each A the usable threshold band is (its score, min score of the rest).
    println!("\n\nUsable threshold band per A  —  SPHERICAL_RESIDUAL_MAX must sit between the");
    println!("spherical fixture's score and the nearest impostor's.  Current value: {SPHERICAL_RESIDUAL_MAX:.0e}");
    println!("Impostors pinned at the flat-ward bound are excluded: `at_upper_bound` rejects");
    println!("them before the residual test runs, so they cannot bind the threshold.");
    println!(
        "    A    spherical fixture   nearest interior impostor      ratio   current value fits"
    );
    for (ai, a) in A_VALUES.iter().enumerate() {
        let sph = cases.iter().find(|c| c.name == "synth_spherical").unwrap();
        let s_gauge = sph.rows[ai].gauge;
        let mut best_gauge = f64::INFINITY;
        let mut best_name = "-";
        for c in &cases {
            if c.name == "synth_spherical" || c.rows[ai].at_upper {
                continue;
            }
            if c.rows[ai].gauge < best_gauge {
                best_gauge = c.rows[ai].gauge;
                best_name = &c.name;
            }
        }
        let fits = s_gauge < SPHERICAL_RESIDUAL_MAX && SPHERICAL_RESIDUAL_MAX < best_gauge;
        println!(
            "  {:4.1}         {:11.3e}   {:11.3e} ({:15})   {:8.1}x   {}",
            a,
            s_gauge,
            best_gauge,
            best_name,
            best_gauge / s_gauge,
            if fits { "yes" } else { "NO" },
        );
    }

    Ok(())
}
