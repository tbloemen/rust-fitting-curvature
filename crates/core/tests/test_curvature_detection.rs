use fitting_core::curvature_detection::detect_geometry;
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_ball_3d, generate_uniform_hyperbolic,
    generate_uniform_hyperbolic3, generate_uniform_sphere, generate_uniform_sphere3,
};

const N: usize = 400;
const BINS: usize = 35;
const SEED: u64 = 42;
/// k_neighbors = 0 means "auto" (N/4 = 100 for N=400)
const K_NN: usize = 0;
/// Euclidean ball radius — matches the geodesic scale of the curved spaces
/// (H²/H³ use max_rho=5, S²/S³ span [0,π]).  At r=5: sinh(5)/5 ≈ 14.8,
/// making H² clearly distinguishable from E².
const E_RADIUS: f64 = 5.0;
/// Hyperbolic max radius — must be ≥5 so that sinh(r)/r ≥ 14.8, giving
/// enough curvature signal for the density-based classifier.
const H_MAX_RHO: f64 = 5.0;

/// Helper: run detection and print a summary (visible with --nocapture).
fn detect_and_print(name: &str, distances: &[f64], n: usize) -> &'static str {
    let result = detect_geometry(distances, n, BINS, K_NN);
    println!(
        "{name}: best={}, R²(E={:.3}, S={:.3}, H={:.3}), dim(E={:.2}, S={:.2}, H={:.2})",
        result.best_geometry,
        result.euclidean.r_squared,
        result.spherical.r_squared,
        result.hyperbolic.r_squared,
        result.euclidean.dim,
        result.spherical.dim,
        result.hyperbolic.dim,
    );
    result.best_geometry
}

#[test]
fn test_detects_euclidean_2d() {
    let data = generate_uniform_ball_2d(N, SEED, E_RADIUS);
    let best = detect_and_print("E²", &data.distances, N);
    assert_eq!(best, "euclidean", "E² should be detected as euclidean");
}

#[test]
fn test_detects_euclidean_3d() {
    let data = generate_uniform_ball_3d(N, SEED, E_RADIUS);
    let best = detect_and_print("E³", &data.distances, N);
    assert_eq!(best, "euclidean", "E³ should be detected as euclidean");
}

#[test]
fn test_detects_spherical_2d() {
    let data = generate_uniform_sphere(N, SEED);
    let best = detect_and_print("S²", &data.distances, N);
    assert_eq!(best, "spherical", "S² should be detected as spherical");
}

#[test]
fn test_detects_spherical_3d() {
    let data = generate_uniform_sphere3(N, SEED);
    let best = detect_and_print("S³", &data.distances, N);
    assert_eq!(best, "spherical", "S³ should be detected as spherical");
}

#[test]
fn test_detects_hyperbolic_2d() {
    let data = generate_uniform_hyperbolic(N, SEED, H_MAX_RHO);
    let best = detect_and_print("H²", &data.distances, N);
    assert_eq!(best, "hyperbolic", "H² should be detected as hyperbolic");
}

#[test]
fn test_detects_hyperbolic_3d() {
    let data = generate_uniform_hyperbolic3(N, SEED, H_MAX_RHO);
    let best = detect_and_print("H³", &data.distances, N);
    assert_eq!(best, "hyperbolic", "H³ should be detected as hyperbolic");
}

/// Verify that the estimated intrinsic dimension is in the right ballpark.
#[test]
fn test_dimension_estimates() {
    // We check loose bounds (±1.5) because the boundary of the sampled ball
    // introduces bias, especially for the higher-dimensional cases.
    let cases = [
        (generate_uniform_ball_2d(N, SEED, E_RADIUS).distances, "euclidean", 2.0_f64),
        (generate_uniform_ball_3d(N, SEED, E_RADIUS).distances, "euclidean", 3.0),
        (generate_uniform_sphere(N, SEED).distances, "spherical", 2.0),
        (generate_uniform_sphere3(N, SEED).distances, "spherical", 3.0),
        (generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances, "hyperbolic", 2.0),
        (generate_uniform_hyperbolic3(N, SEED, H_MAX_RHO).distances, "hyperbolic", 3.0),
    ];

    for (distances, expected_geom, expected_dim) in &cases {
        let result = detect_geometry(distances, N, BINS, K_NN);
        let dim = match *expected_geom {
            "euclidean" => result.euclidean.dim,
            "spherical" => result.spherical.dim,
            "hyperbolic" => result.hyperbolic.dim,
            _ => unreachable!(),
        };
        assert!(
            (dim - expected_dim).abs() < 1.5,
            "{expected_geom} d={expected_dim}: estimated dim={dim:.2}, expected within 1.5"
        );
    }
}
