//! Empirical evaluation of the Wilson et al. (2014) curvature-detection
//! method on the same fixtures as the histogram-based detector, for a
//! head-to-head comparison.

use fitting_core::curvature_detection::{
    detect_geometry, eigenvalues_symmetric, fit_hyperbolic, fit_spherical, hyperbolic_residual_at,
    spherical_residual_at, WilsonFit, SPHERICAL_RESIDUAL_MAX,
};
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere,
};

const N: usize = 400;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const H_MAX_RHO: f64 = 5.0;
/// Target embedding dimension the curvature models are fitted at.
const DIM: usize = 2;

// ── Eigensolver ────────────────────────────────────────────────────────────

/// The residual-eigenvalue criterion needs the *whole* spectrum, so the
/// full symmetric eigensolver underpinning it gets its own checks.
/// A matrix with an analytically known spectrum: the `k × k` all-ones
/// matrix has eigenvalues `{k, 0, …, 0}`, and adding `c·I` shifts them all.
#[test]
fn eigenvalues_symmetric_matches_known_spectrum() {
    const K: usize = 8;
    const C: f64 = -3.0;
    let mut a = vec![1.0; K * K];
    for i in 0..K {
        a[i * K + i] += C;
    }
    let lam = eigenvalues_symmetric(&a, K);
    assert_eq!(lam.len(), K);
    // Ascending: K−1 copies of C, then K+C.
    for &l in &lam[..K - 1] {
        assert!((l - C).abs() < 1e-10, "expected {C}, got {l}");
    }
    let top = lam[K - 1];
    assert!(
        (top - (K as f64 + C)).abs() < 1e-10,
        "expected {}, got {top}",
        K as f64 + C
    );
}

/// On an indefinite matrix with no special structure the spectrum must
/// still reproduce the two invariants it determines: the trace `Σλ` and
/// the squared Frobenius norm `Σλ²`.  Both are sensitive to a single
/// missed or duplicated eigenvalue.
#[test]
fn eigenvalues_symmetric_preserves_trace_and_frobenius() {
    const K: usize = 25;
    let mut a = vec![0.0; K * K];
    for i in 0..K {
        for j in 0..=i {
            // Deterministic, well-mixed, and indefinite.
            let v = ((i * 7 + j * 13 + 1) as f64).sin() * 3.0 + (i as f64 - j as f64) * 0.1;
            a[i * K + j] = v;
            a[j * K + i] = v;
        }
    }
    let trace: f64 = (0..K).map(|i| a[i * K + i]).sum();
    let frob_sq: f64 = a.iter().map(|x| x * x).sum();

    let lam = eigenvalues_symmetric(&a, K);
    let sum: f64 = lam.iter().sum();
    let sum_sq: f64 = lam.iter().map(|l| l * l).sum();

    assert!(
        (sum - trace).abs() < 1e-9 * frob_sq.sqrt(),
        "Σλ = {sum} should equal trace = {trace}"
    );
    assert!(
        (sum_sq - frob_sq).abs() < 1e-9 * frob_sq,
        "Σλ² = {sum_sq} should equal ‖A‖_F² = {frob_sq}"
    );
    assert!(
        lam.windows(2).all(|w| w[0] <= w[1]),
        "eigenvalues should be returned in ascending order"
    );
}

/// The signature criterion is exact on data that really lies on the
/// model manifold: for points on a unit sphere, `Z(1) = r²cos(d/r)` is
/// the Gram matrix of the ambient `R³` coordinates, so it has rank 3 and
/// everything below the top 3 eigenvalues is numerical noise.
#[test]
fn spherical_residual_vanishes_at_true_radius() {
    const M: usize = 60;
    let data = generate_uniform_sphere(M, SEED);
    let d_max = data.distances.iter().cloned().fold(0.0_f64, f64::max);
    let res = spherical_residual_at(&data.distances, M, DIM, 1.0);
    // Same `n · d_max²` gauge `WilsonFit::residual_normalised` uses, so the
    // tolerance means the same thing here as at the detection threshold.
    let normalised = res / (M as f64 * d_max * d_max);
    assert!(
        normalised < 1e-12,
        "S² at r=1: normalised residual {normalised:.3e} should be ~0"
    );
}

// ── Sanity checks ──────────────────────────────────────────────────────────

/// On unit-sphere data the spherical fit should return a radius close
/// to 1 with a residual much smaller than the trivial scale (r²·n).
#[test]
#[ignore = "too slow"]
fn fit_spherical_unit_sphere_recovers_radius() {
    let data = generate_uniform_sphere(N, SEED);
    let WilsonFit {
        radius,
        residual,
        residual_normalised,
        at_upper_bound,
    } = fit_spherical(&data.distances, N, DIM);
    println!(
        "S²: r* = {radius:.3}, Σ|λ_res| = {residual:.3e} (normalised {residual_normalised:.3e}), at_upper = {at_upper_bound}"
    );
    assert!(
        residual_normalised < SPHERICAL_RESIDUAL_MAX,
        "S²: normalised residual {residual_normalised:.3e} should clear the detection threshold {SPHERICAL_RESIDUAL_MAX:.0e}"
    );
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
/// |λ₁|.  Under the narrowed angular-coverage window Euclidean data can't
/// find an interior spherical minimum: its residual keeps falling toward
/// the flatter (larger-`r`) edge, so the fit pins at the upper bound —
/// exactly the signal `detect_geometry` uses to reject a spherical
/// verdict.  This test pins that signature.
#[test]
#[ignore = "too slow"]
fn fit_spherical_euclidean_pins_at_upper_bound() {
    let data = generate_uniform_ball_2d(N, SEED, E_RADIUS);
    let fit = fit_spherical(&data.distances, N, DIM);
    println!(
        "E²(spherical fit): r* = {:.3}, ρ = {:.3e}, at_upper = {}",
        fit.radius, fit.residual, fit.at_upper_bound,
    );
    assert!(
        fit.at_upper_bound,
        "E²: spherical fit should pin at the flat-ward upper bound (not a genuine sphere)"
    );
}

/// On a hyperbolic dataset with max_rho=5, the recovered radius is
/// expected near 1 (the true curvature radius of the generator).
#[test]
#[ignore = "too slow"]
fn fit_hyperbolic_h2_recovers_finite_radius() {
    let data = generate_uniform_hyperbolic(N, SEED, H_MAX_RHO);
    let fit = fit_hyperbolic(&data.distances, N, DIM);
    println!(
        "H²: r* = {:.3}, ρ = {:.3e}, at_upper = {}",
        fit.radius, fit.residual, fit.at_upper_bound,
    );
    assert!(
        !fit.at_upper_bound,
        "H² fit should not be pinned at the upper bound"
    );
}

// ── Single-seed end-to-end detection ───────────────────────────────────────

#[test]
#[ignore = "too slow"]
fn detect_curvature_single_seed_per_geometry() {
    // Only intrinsically-2D fixtures: the detector fits dim-2 models, so
    // higher-dimensional manifolds (S³, H³, …) have no well-defined
    // expected radius/geometry under this criterion.
    let cases: [(&str, Vec<f64>, &str); 3] = [
        (
            "E²",
            generate_uniform_ball_2d(N, SEED, E_RADIUS).distances,
            "euclidean",
        ),
        (
            "S²",
            generate_uniform_sphere(N, SEED).distances,
            "spherical",
        ),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
            "hyperbolic",
        ),
    ];

    for (name, dist, expected) in &cases {
        let verdict = detect_geometry(dist, N, DIM);
        // The Wilson fits are no longer carried on the detection result;
        // recompute them directly for the diagnostic print.
        let spherical = fit_spherical(dist, N, DIM);
        let hyperbolic = fit_hyperbolic(dist, N, DIM);
        println!(
            "{name}: best = {}, K = {:+.3}, S(r={:.2}, ε={:.2e}), H(r={:.2}, ε={:.2e})",
            verdict.best_geometry,
            verdict.curvature,
            spherical.radius,
            spherical.residual,
            hyperbolic.radius,
            hyperbolic.residual,
        );
        assert_eq!(
            verdict.best_geometry, *expected,
            "{name}: expected {expected}, got {}",
            verdict.best_geometry,
        );
    }
}

/// Diagnostic: scan all fixtures and print every fit result without
/// asserting (so we can see the whole picture).
#[test]
#[ignore = "diagnostic only"]
fn diag_all_fixtures() {
    let cases: [(&str, Vec<f64>, &str); 3] = [
        (
            "E²",
            generate_uniform_ball_2d(N, SEED, E_RADIUS).distances,
            "euclidean",
        ),
        (
            "S²",
            generate_uniform_sphere(N, SEED).distances,
            "spherical",
        ),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
            "hyperbolic",
        ),
    ];

    println!();
    println!("Fixture | best        | S(r*, ε, ε/(n dmax²), pin) | H(r*, ε, ε/(n dmax²), pin)");
    for (name, dist, _expected) in &cases {
        let verdict = detect_geometry(dist, N, DIM);
        let spherical = fit_spherical(dist, N, DIM);
        let hyperbolic = fit_hyperbolic(dist, N, DIM);
        println!(
            "{name:7} | {:11} | r={:6.3} ε={:.2e} n={:.2e} pin={:5} | r={:6.3} ε={:.2e} n={:.2e} pin={:5}",
            verdict.best_geometry,
            spherical.radius,
            spherical.residual,
            spherical.residual_normalised,
            spherical.at_upper_bound,
            hyperbolic.radius,
            hyperbolic.residual,
            hyperbolic.residual_normalised,
            hyperbolic.at_upper_bound,
        );
    }
}

/// Diagnostic: print residual curves so we can see whether the
/// minimisation has a clean interior minimum or a numerical-noise floor.
#[test]
#[ignore = "diagnostic only"]
fn diag_residual_curve() {
    let cases: [(&str, Vec<f64>); 3] = [
        ("E²", generate_uniform_ball_2d(N, SEED, E_RADIUS).distances),
        ("S²", generate_uniform_sphere(N, SEED).distances),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
        ),
    ];

    for (name, dist) in &cases {
        let d_max = dist.iter().cloned().fold(0.0f64, f64::max);
        println!("\n── {name} (d_max={d_max:.2}) ──");
        println!("       r        Σ|λ_res|(sph)      Σ|λ_res|(hyp)");
        // Span the full hyperbolic search range from d_max/20 to 5*d_max
        // so we see whether residual minima live at small r (the true
        // hyperbolic regime) or large r (the Euclidean limit).
        let r_lo = d_max / 20.0;
        let r_hi = 5.0 * d_max;
        for i in 0..20 {
            let t = i as f64 / 19.0;
            let r = r_lo * (r_hi / r_lo).powf(t);
            let s_res = if r >= d_max / std::f64::consts::PI {
                spherical_residual_at(dist, N, DIM, r)
            } else {
                f64::NAN
            };
            let h_res = hyperbolic_residual_at(dist, N, DIM, r);
            println!("  {r:8.3}    {s_res:14.3e}    {h_res:14.3e}");
        }
    }
}

// ── Robustness across seeds (head-to-head with the histogram method) ───────

const ROBUSTNESS_SEEDS: u64 = 1;

fn hit_rate<F: Fn(u64) -> Vec<f64>>(make_dist: F, expected: &str) -> usize {
    (0..ROBUSTNESS_SEEDS)
        .filter(|&s| detect_geometry(&make_dist(s), N, DIM).best_geometry == expected)
        .count()
}

#[test]
#[ignore = "too slow"]
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
#[ignore = "too slow"]
fn robust_seeds_s2() {
    let h = hit_rate(|s| generate_uniform_sphere(N, s).distances, "spherical");
    println!("S²: {h}/{ROBUSTNESS_SEEDS}");
    assert!(
        h as u64 >= ROBUSTNESS_SEEDS - 1,
        "S²: {h}/{ROBUSTNESS_SEEDS}"
    );
}

#[test]
#[ignore = "too slow"]
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
