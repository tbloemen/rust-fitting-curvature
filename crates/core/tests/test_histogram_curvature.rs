//! Tests for the naive shell-density geometry detector.
//!
//! This detector is intentionally unreliable at recovering the curvature
//! *sign* (see the module docs): Euclidean is the `c → 0` limit of both the
//! spherical and hyperbolic densities, so the three curves are nearly
//! degenerate on finite samples.  These tests therefore assert only that
//!
//!  - the shell-density profile and the three curve fits are well-formed,
//!  - the fit machinery recovers a sensible intrinsic dimension when the
//!    *matching* model is fit to matching-curvature data, and
//!  - [`detect_geometry`] returns a coherent verdict (valid label, with a
//!    curvature sign consistent with that label).
//!
//! Classification *correctness* is deliberately NOT asserted — distinguishing
//! the curvature sign is the job of the Gromov growing-ball detector.

use fitting_core::curvature_detection::histogram::{
    detect_geometry, fit_geometries, shell_density_profile,
};
use fitting_core::curvature_detection::old_detection::gromov_hyperbolicity;
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_ball_3d, generate_uniform_hyperbolic,
    generate_uniform_sphere,
};

const N: usize = 400;
const BINS: usize = 35;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const H_MAX_RHO: f64 = 5.0;

// ── Shell-density profile ─────────────────────────────────────────────────────

#[test]
fn shell_profile_is_well_formed() {
    let data = generate_uniform_ball_2d(N, SEED, E_RADIUS);
    let p = shell_density_profile(&data.distances, N, BINS);

    assert_eq!(p.bin_centers.len(), BINS);
    assert_eq!(p.density.len(), BINS);
}

// ── Fit machinery ─────────────────────────────────────────────────────────────

#[test]
fn fits_are_well_formed() {
    let data = generate_uniform_ball_2d(N, SEED, E_RADIUS);
    let fits = fit_geometries(&data.distances, N, BINS);

    for (name, f) in [
        ("euclidean", &fits.euclidean),
        ("spherical", &fits.spherical),
        ("hyperbolic", &fits.hyperbolic),
    ] {
        assert!(
            (0.0..=1.0).contains(&f.r_squared),
            "{name}: R²={} out of [0,1]",
            f.r_squared
        );
        assert!(f.dim.is_finite(), "{name}: dim not finite");
        assert!(
            f.curvature_scale >= 0.0,
            "{name}: curvature_scale {} should be ≥ 0",
            f.curvature_scale
        );
    }
    assert_eq!(
        fits.euclidean.curvature_scale, 0.0,
        "Euclidean fit must report zero curvature scale"
    );
}

/// The matching model fits its own data well (high R²) and recovers a
/// plausible intrinsic dimension.  This checks the regression machinery,
/// not the classifier.
#[test]
fn matching_model_recovers_dimension() {
    // (label, distances, expected dim, fit selector)
    let euc2 = generate_uniform_ball_2d(N, SEED, E_RADIUS);
    let euc3 = generate_uniform_ball_3d(N, SEED, E_RADIUS);

    let f2 = fit_geometries(&euc2.distances, N, BINS).euclidean;
    let f3 = fit_geometries(&euc3.distances, N, BINS).euclidean;

    println!("E² euclidean fit: dim={:.2} R²={:.3}", f2.dim, f2.r_squared);
    println!("E³ euclidean fit: dim={:.2} R²={:.3}", f3.dim, f3.r_squared);

    assert!(
        f2.r_squared > 0.8,
        "E² euclidean R²={:.3} too low",
        f2.r_squared
    );
    assert!(
        f3.r_squared > 0.8,
        "E³ euclidean R²={:.3} too low",
        f3.r_squared
    );
    assert!(
        (1.0..=3.5).contains(&f2.dim),
        "E² recovered dim {:.2} implausible",
        f2.dim
    );
    assert!(
        f3.dim > f2.dim,
        "E³ dim {:.2} should exceed E² dim {:.2}",
        f3.dim,
        f2.dim
    );
}

// ── Verdict coherence (NOT correctness) ───────────────────────────────────────

#[test]
fn verdict_is_coherent() {
    let cases = [
        ("E²", generate_uniform_ball_2d(N, SEED, E_RADIUS).distances),
        ("S²", generate_uniform_sphere(N, SEED).distances),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
        ),
    ];

    for (name, dist) in &cases {
        let v = detect_geometry(dist, N, BINS);
        assert!(
            matches!(v.best_geometry, "euclidean" | "spherical" | "hyperbolic"),
            "{name}: unexpected label {}",
            v.best_geometry
        );
        match v.best_geometry {
            "euclidean" => assert_eq!(v.curvature, 0.0, "{name}: euclidean must have K=0"),
            "spherical" => assert!(v.curvature >= 0.0, "{name}: spherical must have K≥0"),
            "hyperbolic" => assert!(v.curvature <= 0.0, "{name}: hyperbolic must have K≤0"),
            _ => unreachable!(),
        }
    }
}

/// Diagnostic: print the three fits and the verdict for each fixture to
/// see how close the residuals are (the reason the sign is unreliable).
#[test]
#[ignore = "diagnostic only"]
fn diag_fit_degeneracy() {
    let cases = [
        ("E²", generate_uniform_ball_2d(N, SEED, E_RADIUS).distances),
        ("S²", generate_uniform_sphere(N, SEED).distances),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
        ),
    ];
    println!("\nfixture | euc R² | sph R² | hyp R² | verdict");
    for (name, dist) in &cases {
        let f = fit_geometries(dist, N, BINS);
        let v = detect_geometry(dist, N, BINS);
        println!(
            "{name:7} | {:.4} | {:.4} | {:.4} | {} (K={:+.3})",
            f.euclidean.r_squared,
            f.spherical.r_squared,
            f.hyperbolic.r_squared,
            v.best_geometry,
            v.curvature,
        );
    }
}

// ── Gromov δ (old_detection global percentile) ────────────────────────────────

/// A tree is δ-hyperbolic with δ=0 for every 4-tuple.
/// Center node 0, leaves 1..n-1: d(center, leaf) = 1, d(leaf, leaf) = 2.
#[test]
fn gromov_tree_metric_is_zero() {
    let n = 10usize;
    let mut dist = vec![0.0f64; n * n];
    for i in 1..n {
        dist[i] = 1.0; // d(0, i)
        dist[i * n] = 1.0; // d(i, 0)
        for j in 1..n {
            if i != j {
                dist[i * n + j] = 2.0;
            }
        }
    }
    let delta = gromov_hyperbolicity(&dist, n, 2000);
    assert!(
        delta < 1e-10,
        "star-graph (tree) should have normalised δ=0, got {delta}"
    );
}

/// Too few points to form a 4-tuple: must return 0.0 without panicking.
#[test]
fn gromov_small_n_returns_zero() {
    let dist3 = vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    assert_eq!(gromov_hyperbolicity(&dist3, 3, 500), 0.0);

    let dist1 = vec![0.0];
    assert_eq!(gromov_hyperbolicity(&dist1, 1, 500), 0.0);
}

/// Hyperbolic data should produce a smaller normalised δ than Euclidean data
/// of the same scale — the defining property of negative curvature.
#[test]
fn gromov_hyperbolic_smaller_than_euclidean() {
    let hyp = generate_uniform_hyperbolic(N, SEED, H_MAX_RHO);
    let euc = generate_uniform_ball_2d(N, SEED, E_RADIUS);

    let delta_hyp = gromov_hyperbolicity(&hyp.distances, N, 5000);
    let delta_euc = gromov_hyperbolicity(&euc.distances, N, 5000);

    assert!(
        delta_hyp < delta_euc,
        "hyperbolic δ={delta_hyp:.3} should be < euclidean δ={delta_euc:.3}"
    );
}
