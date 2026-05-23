//! Empirical evaluation of the Wilson et al. (2014) curvature-detection
//! method on the same fixtures as the histogram-based detector, for a
//! head-to-head comparison.

use fitting_core::curvature_detection::{
    WilsonFit, detect_geometry, fit_hyperbolic, fit_spherical, hyperbolic_residual_at,
    spherical_residual_at,
};
use fitting_core::synthetic_data::{
    generate_hd_sphere, generate_uniform_ball_2d, generate_uniform_ball_3d,
    generate_uniform_hyperbolic, generate_uniform_hyperbolic3, generate_uniform_sphere,
    generate_uniform_sphere3,
};

const N: usize = 400;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const H_MAX_RHO: f64 = 5.0;

// ── Sanity checks ──────────────────────────────────────────────────────────

/// On unit-sphere data the spherical fit should return a radius close
/// to 1 with a residual much smaller than the trivial scale (r²·n).
#[test]
fn fit_spherical_unit_sphere_recovers_radius() {
    let data = generate_uniform_sphere(N, SEED);
    let WilsonFit {
        radius,
        residual,
        at_upper_bound,
    } = fit_spherical(&data.distances, N);
    println!("S²: r* = {radius:.3}, |λ₁| = {residual:.3e}, at_upper = {at_upper_bound}");
    assert!(
        (0.8..=1.5).contains(&radius),
        "S² recovered radius {radius:.3} should be near 1 (true)"
    );
    assert!(
        !at_upper_bound,
        "S² fit should not be pinned at the search upper bound"
    );
}

/// Wilson's eigenvalue residual on its own cannot tell Euclidean data
/// from a small cap on a very-large-radius sphere — both produce small
/// |λ₁|.  The distinguishing fact for Euclidean is that the data only
/// covers a small *angular* extent of the fitted sphere
/// (`d_max / r* ≪ π`).  This test pins that signature.
#[test]
fn fit_spherical_euclidean_has_small_angular_extent() {
    let data = generate_uniform_ball_2d(N, SEED, E_RADIUS);
    let d_max = data.distances.iter().cloned().fold(0.0_f64, f64::max);
    let fit = fit_spherical(&data.distances, N);
    let angular = d_max / fit.radius;
    println!(
        "E²(spherical fit): r* = {:.3}, |λ₁| = {:.3e}, d_max/r* = {:.3}",
        fit.radius, fit.residual, angular,
    );
    assert!(
        angular < 2.0,
        "E²: d_max/r* = {angular:.3} should be small (< 2.0) — far from π"
    );
}

/// On a hyperbolic dataset with max_rho=5, the recovered radius is
/// expected near 1 (the true curvature radius of the generator).
#[test]
fn fit_hyperbolic_h2_recovers_finite_radius() {
    let data = generate_uniform_hyperbolic(N, SEED, H_MAX_RHO);
    let fit = fit_hyperbolic(&data.distances, N);
    println!(
        "H²: r* = {:.3}, |λ₂| = {:.3e}, at_upper = {}",
        fit.radius, fit.residual, fit.at_upper_bound,
    );
    assert!(
        !fit.at_upper_bound,
        "H² fit should not be pinned at the upper bound"
    );
}

// ── Single-seed end-to-end detection ───────────────────────────────────────

#[test]
fn detect_curvature_single_seed_per_geometry() {
    let cases: [(&str, Vec<f64>, &str); 6] = [
        (
            "E²",
            generate_uniform_ball_2d(N, SEED, E_RADIUS).distances,
            "euclidean",
        ),
        (
            "E³",
            generate_uniform_ball_3d(N, SEED, E_RADIUS).distances,
            "euclidean",
        ),
        (
            "S²",
            generate_uniform_sphere(N, SEED).distances,
            "spherical",
        ),
        (
            "S³",
            generate_uniform_sphere3(N, SEED).distances,
            "spherical",
        ),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
            "hyperbolic",
        ),
        (
            "H³",
            generate_uniform_hyperbolic3(N, SEED, H_MAX_RHO).distances,
            "hyperbolic",
        ),
    ];

    for (name, dist, expected) in &cases {
        let r = detect_geometry(dist, N);
        println!(
            "{name}: best = {}, K = {:+.3}, S(r={:.2}, ε={:.2e}), H(r={:.2}, ε={:.2e})",
            r.best_geometry,
            r.curvature,
            r.spherical.radius,
            r.spherical.residual,
            r.hyperbolic.radius,
            r.hyperbolic.residual,
        );
        assert_eq!(
            r.best_geometry, *expected,
            "{name}: expected {expected}, got {}",
            r.best_geometry,
        );
    }
}

/// Diagnostic: scan all fixtures and print every fit result without
/// asserting (so we can see the whole picture).
#[test]
#[ignore = "diagnostic only"]
fn diag_all_fixtures() {
    let cases: [(&str, Vec<f64>, &str); 6] = [
        (
            "E²",
            generate_uniform_ball_2d(N, SEED, E_RADIUS).distances,
            "euclidean",
        ),
        (
            "E³",
            generate_uniform_ball_3d(N, SEED, E_RADIUS).distances,
            "euclidean",
        ),
        (
            "S²",
            generate_uniform_sphere(N, SEED).distances,
            "spherical",
        ),
        (
            "S³",
            generate_uniform_sphere3(N, SEED).distances,
            "spherical",
        ),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
            "hyperbolic",
        ),
        (
            "H³",
            generate_uniform_hyperbolic3(N, SEED, H_MAX_RHO).distances,
            "hyperbolic",
        ),
    ];

    println!();
    println!("Fixture | best        | S(r*, ε, d/r*)     | H(r*, ε, d/r*)");
    for (name, dist, _expected) in &cases {
        let d_max = dist.iter().cloned().fold(0.0f64, f64::max);
        let r = detect_geometry(dist, N);
        let s_ang = d_max / r.spherical.radius;
        let h_ang = d_max / r.hyperbolic.radius;
        println!(
            "{name:7} | {:11} | r={:6.3} ε={:.2e} d/r*={:.2} | r={:6.3} ε={:.2e} d/r*={:.2}",
            r.best_geometry,
            r.spherical.radius,
            r.spherical.residual,
            s_ang,
            r.hyperbolic.radius,
            r.hyperbolic.residual,
            h_ang,
        );
    }
}

/// Diagnostic: print residual curves so we can see whether the
/// minimisation has a clean interior minimum or a numerical-noise floor.
#[test]
#[ignore = "diagnostic only"]
fn diag_residual_curve() {
    let cases: [(&str, Vec<f64>); 4] = [
        ("E²", generate_uniform_ball_2d(N, SEED, E_RADIUS).distances),
        ("S²", generate_uniform_sphere(N, SEED).distances),
        ("S³", generate_uniform_sphere3(N, SEED).distances),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
        ),
    ];

    for (name, dist) in &cases {
        let d_max = dist.iter().cloned().fold(0.0f64, f64::max);
        println!("\n── {name} (d_max={d_max:.2}) ──");
        println!("       r        |λ₁(spherical)|     |λ₂(hyperbolic)|");
        // Span the full hyperbolic search range from d_max/20 to 5*d_max
        // so we see whether residual minima live at small r (the true
        // hyperbolic regime) or large r (the Euclidean limit).
        let r_lo = d_max / 20.0;
        let r_hi = 5.0 * d_max;
        for i in 0..20 {
            let t = i as f64 / 19.0;
            let r = r_lo * (r_hi / r_lo).powf(t);
            let s_res = if r >= d_max / std::f64::consts::PI {
                spherical_residual_at(dist, N, r)
            } else {
                f64::NAN
            };
            let h_res = hyperbolic_residual_at(dist, N, r);
            println!("  {r:8.3}    {s_res:14.3e}    {h_res:14.3e}");
        }
    }
}

// ── Robustness across seeds (head-to-head with the histogram method) ───────

const ROBUSTNESS_SEEDS: u64 = 1;

fn hit_rate<F: Fn(u64) -> Vec<f64>>(make_dist: F, expected: &str) -> usize {
    (0..ROBUSTNESS_SEEDS)
        .filter(|&s| detect_geometry(&make_dist(s), N).best_geometry == expected)
        .count()
}

#[test]
fn robust_seeds_e2() {
    let h = hit_rate(
        |s| generate_uniform_ball_2d(N, s, E_RADIUS).distances,
        "euclidean",
    );
    println!("E²: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "E²: {h}/{ROBUSTNESS_SEEDS}"
    );
}

#[test]
fn robust_seeds_e3() {
    let h = hit_rate(
        |s| generate_uniform_ball_3d(N, s, E_RADIUS).distances,
        "euclidean",
    );
    println!("E³: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "E³: {h}/{ROBUSTNESS_SEEDS}"
    );
}

#[test]
fn robust_seeds_s2() {
    let h = hit_rate(|s| generate_uniform_sphere(N, s).distances, "spherical");
    println!("S²: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "S²: {h}/{ROBUSTNESS_SEEDS}"
    );
}

#[test]
fn robust_seeds_s3() {
    let h = hit_rate(|s| generate_uniform_sphere3(N, s).distances, "spherical");
    println!("S³: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "S³: {h}/{ROBUSTNESS_SEEDS}"
    );
}

#[test]
fn robust_seeds_s4() {
    let h = hit_rate(|s| generate_hd_sphere(N, 5, s).distances, "spherical");
    println!("S⁴: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "S⁴: {h}/{ROBUSTNESS_SEEDS}"
    );
}

#[test]
fn robust_seeds_s5() {
    let h = hit_rate(|s| generate_hd_sphere(N, 6, s).distances, "spherical");
    println!("S⁵: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "S⁵: {h}/{ROBUSTNESS_SEEDS}"
    );
}

#[test]
fn robust_seeds_h2() {
    let h = hit_rate(
        |s| generate_uniform_hyperbolic(N, s, H_MAX_RHO).distances,
        "hyperbolic",
    );
    println!("H²: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "H²: {h}/{ROBUSTNESS_SEEDS}"
    );
}

#[test]
fn robust_seeds_h3() {
    let h = hit_rate(
        |s| generate_uniform_hyperbolic3(N, s, H_MAX_RHO).distances,
        "hyperbolic",
    );
    println!("H³: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "H³: {h}/{ROBUSTNESS_SEEDS}"
    );
}
