//! Is `d_max` the right variable to scale the *hyperbolic* search window by?
//!
//! For the sphere it is forced: `d_max ≤ πr` is a hard geometric constraint, so
//! `d_max` is intrinsically the quantity the window must be expressed in.
//! Hyperbolic space is unbounded and imposes no analogous constraint, so the
//! choice is free — and `d_max` has a defect the sphere case tolerates because
//! it has no alternative: it is an extreme order statistic over `n(n−1)/2`
//! pairs, so it **grows with `n`** for a fixed underlying distribution.
//!
//! A window written as `[d_max/20, d_max]` therefore *moves with sample size*,
//! and so does the smallest reportable curvature `κ_min = (d_rms/d_max)²`.
//! For a thesis that sweeps `N`, that couples the detector's resolution to the
//! sweep axis.
//!
//! Two `n`-stable alternatives are checked here:
//!   * `d_rms` — the same statistic the thesis κ gauge already uses;
//!   * `r_δ = δ_sat / ln(1+√2)` — the curvature radius the Gromov growing-ball
//!     test already produces, which is what gates the hyperbolic verdict anyway.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_scale_variable`

use fitting_core::curvature_detection::detect_hyperbolic;
use fitting_core::synthetic_data::{generate_uniform_ball_2d, generate_uniform_hyperbolic};

const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}

fn d_rms_of(d: &[f64], n: usize) -> f64 {
    let s: f64 = d.iter().map(|x| x * x).sum();
    (s / (n as f64 * (n as f64 - 1.0))).sqrt()
}

fn main() {
    println!("══ 1 — how the scale statistics move with n (same distribution) ══");
    println!("  H² fixture, true curvature radius ρ = 1, max_rho = 5\n");
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>10}  {:>10}  {:>12}",
        "n", "d_max", "d_rms", "d_rms/dmax", "cap=d_max", "κ_min"
    );
    let mut first_dmax = 0.0;
    for &n in &[100_usize, 200, 400, 800, 1600] {
        let d = generate_uniform_hyperbolic(n, SEED, 5.0).distances;
        let dm = d_max_of(&d);
        let dr = d_rms_of(&d, n);
        if first_dmax == 0.0 {
            first_dmax = dm;
        }
        println!(
            "  {n:>6}  {dm:>8.4}  {dr:>8.4}  {:>10.4}  {:>10.4}  {:>12.4}",
            dr / dm,
            dm,
            (dr / dm).powi(2)
        );
    }
    let d1600 = generate_uniform_hyperbolic(1600, SEED, 5.0).distances;
    let d100 = generate_uniform_hyperbolic(100, SEED, 5.0).distances;
    println!(
        "\n  drift from n=100 to n=1600:   d_max {:+.1}%   d_rms {:+.1}%",
        100.0 * (d_max_of(&d1600) / d_max_of(&d100) - 1.0),
        100.0 * (d_rms_of(&d1600, 1600) / d_rms_of(&d100, 100) - 1.0)
    );

    println!("\n  Same for exactly Euclidean data (no curvature to confound it):");
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>10}",
        "n", "d_max", "d_rms", "d_rms/dmax"
    );
    for &n in &[100_usize, 200, 400, 800, 1600] {
        let d = generate_uniform_ball_2d(n, SEED, E_RADIUS).distances;
        let dm = d_max_of(&d);
        let dr = d_rms_of(&d, n);
        println!("  {n:>6}  {dm:>8.4}  {dr:>8.4}  {:>10.4}", dr / dm);
    }

    // ── 2. The δ-derived radius, which uses no extreme statistic ───────────
    println!("\n══ 2 — the Gromov δ radius r_δ = δ_sat / ln(1+√2), true ρ = 1 ══");
    println!(
        "  {:>6}  {:>8}  {:>10}  {:>10}  {:>10}  {:>8}",
        "n", "δ_sat", "r_δ", "r_δ / ρ", "d_max/ρ", "hyp?"
    );
    for &n in &[100_usize, 200, 400, 800] {
        let d = generate_uniform_hyperbolic(n, SEED, 5.0).distances;
        let v = detect_hyperbolic(&d, n);
        let r_delta = v.saturated_delta / (1.0 + 2.0_f64.sqrt()).ln();
        println!(
            "  {n:>6}  {:>8.4}  {r_delta:>10.4}  {:>10.4}  {:>10.4}  {:>8}",
            v.saturated_delta,
            r_delta,
            d_max_of(&d),
            v.is_hyperbolic
        );
    }

    // Across curvature scales: does r_δ track ρ, and does d_max?
    println!("\n  Across extents (ρ = 1 fixed, extent shrinking):");
    println!(
        "  {:>8}  {:>8}  {:>10}  {:>8}",
        "max_rho", "d_max", "r_δ (ρ=1)", "hyp?"
    );
    for &mr in &[5.0_f64, 3.0, 2.0, 1.0, 0.5] {
        let d = generate_uniform_hyperbolic(400, SEED, mr).distances;
        let v = detect_hyperbolic(&d, 400);
        let r_delta = v.saturated_delta / (1.0 + 2.0_f64.sqrt()).ln();
        println!(
            "  {mr:>8.2}  {:>8.4}  {r_delta:>10.4}  {:>8}",
            d_max_of(&d),
            v.is_hyperbolic
        );
    }
}
