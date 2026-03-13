//! Tests for manifold implementations.
//! Ported from Python test/test_grad.py and test/test_rsgd.py (manifold-related parts).

use fitting_core::manifolds::{Euclidean, Hyperboloid, Manifold, Sphere};
use fitting_core::synthetic_data::Rng;

// ---------------------------------------------------------------------------
// Hyperboloid tests
// ---------------------------------------------------------------------------

#[test]
fn test_hyperboloid_constraint() {
    let h = Hyperboloid::new(-1.0);
    let pts = h.init_points(10, 2, 0.01, 42);
    let ambient = 3;
    for i in 0..10 {
        let x0 = pts[i * ambient];
        let x1 = pts[i * ambient + 1];
        let x2 = pts[i * ambient + 2];
        let lorentz = -x0 * x0 + x1 * x1 + x2 * x2;
        assert!((lorentz + 1.0).abs() < 1e-6, "Point {i} lorentz={lorentz}");
    }
}

#[test]
fn test_hyperboloid_distances_non_negative() {
    let h = Hyperboloid::new(-1.0);
    let pts = h.init_points(20, 2, 0.01, 42);
    let dist = h.pairwise_distances(&pts, 20, 3);
    for i in 0..20 {
        for j in 0..20 {
            assert!(
                dist[i * 20 + j] >= 0.0,
                "Negative distance at ({i},{j}): {}",
                dist[i * 20 + j]
            );
        }
    }
}

#[test]
fn test_hyperboloid_distances_symmetric() {
    let h = Hyperboloid::new(-1.0);
    let pts = h.init_points(20, 2, 0.01, 42);
    let dist = h.pairwise_distances(&pts, 20, 3);
    for i in 0..20 {
        for j in 0..20 {
            let diff = (dist[i * 20 + j] - dist[j * 20 + i]).abs();
            assert!(diff < 1e-10, "Not symmetric at ({i},{j})");
        }
    }
}

#[test]
fn test_hyperboloid_diagonal_zero() {
    let h = Hyperboloid::new(-1.0);
    let pts = h.init_points(20, 2, 0.01, 42);
    let dist = h.pairwise_distances(&pts, 20, 3);
    for i in 0..20 {
        assert!(
            dist[i * 20 + i].abs() < 1e-10,
            "Diagonal not zero at {i}: {}",
            dist[i * 20 + i]
        );
    }
}

#[test]
fn test_hyperboloid_exp_map_preserves_constraint() {
    let h = Hyperboloid::new(-1.0);
    let n = 10;
    let ambient = 3;
    let mut pts = h.init_points(n, 2, 0.01, 42);

    // Create tangent vectors (random, then project)
    let mut rng = Rng::new(42);
    let mut tangent: Vec<f64> = (0..n * ambient).map(|_| rng.normal() * 0.01).collect();
    h.project_to_tangent(&pts, &mut tangent, n, ambient);

    // Apply exp map
    h.exp_map(&mut pts, &tangent, n, ambient);

    // Check constraint is preserved
    for i in 0..n {
        let x0 = pts[i * ambient];
        let x1 = pts[i * ambient + 1];
        let x2 = pts[i * ambient + 2];
        let lorentz = -x0 * x0 + x1 * x1 + x2 * x2;
        assert!(
            (lorentz + 1.0).abs() < 1e-4,
            "Constraint violated after exp_map: point {i}, lorentz={lorentz}"
        );
    }
}

#[test]
fn test_hyperboloid_different_curvatures() {
    for &k in &[-0.5, -1.0, -2.0] {
        let h = Hyperboloid::new(k);
        let r_sq = 1.0 / (-k);
        let pts = h.init_points(10, 2, 0.01, 42);
        let ambient = 3;

        for i in 0..10 {
            let x0 = pts[i * ambient];
            let spatial_sq: f64 = (1..ambient).map(|d| pts[i * ambient + d].powi(2)).sum();
            let lorentz = -x0 * x0 + spatial_sq;
            assert!(
                (lorentz + r_sq).abs() < 1e-4,
                "Constraint violated for k={k}: point {i}, lorentz={lorentz}, expected {}",
                -r_sq
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Sphere tests
// ---------------------------------------------------------------------------

#[test]
fn test_sphere_constraint() {
    let s = Sphere::new(1.0);
    let pts = s.init_points(10, 2, 0.01, 42);
    let ambient = 3;
    for i in 0..10 {
        let mut norm_sq = 0.0;
        for d in 0..ambient {
            norm_sq += pts[i * ambient + d].powi(2);
        }
        assert!((norm_sq - 1.0).abs() < 1e-6, "Point {i} norm_sq={norm_sq}");
    }
}

#[test]
fn test_sphere_distances_non_negative() {
    let s = Sphere::new(1.0);
    let pts = s.init_points(20, 2, 0.01, 42);
    let dist = s.pairwise_distances(&pts, 20, 3);
    for i in 0..20 {
        for j in 0..20 {
            assert!(dist[i * 20 + j] >= 0.0, "Negative distance at ({i},{j})");
        }
    }
}

#[test]
fn test_sphere_distances_symmetric() {
    let s = Sphere::new(1.0);
    let pts = s.init_points(20, 2, 0.01, 42);
    let dist = s.pairwise_distances(&pts, 20, 3);
    for i in 0..20 {
        for j in 0..20 {
            let diff = (dist[i * 20 + j] - dist[j * 20 + i]).abs();
            assert!(diff < 1e-10, "Not symmetric at ({i},{j})");
        }
    }
}

#[test]
fn test_sphere_exp_map_preserves_constraint() {
    let s = Sphere::new(1.0);
    let n = 10;
    let ambient = 3;
    let mut pts = s.init_points(n, 2, 0.01, 42);

    let mut rng = Rng::new(42);
    let mut tangent: Vec<f64> = (0..n * ambient).map(|_| rng.normal() * 0.01).collect();
    s.project_to_tangent(&pts, &mut tangent, n, ambient);

    s.exp_map(&mut pts, &tangent, n, ambient);

    for i in 0..n {
        let mut norm_sq = 0.0;
        for d in 0..ambient {
            norm_sq += pts[i * ambient + d].powi(2);
        }
        assert!(
            (norm_sq - 1.0).abs() < 1e-4,
            "Sphere constraint violated after exp_map: point {i}, norm_sq={norm_sq}"
        );
    }
}

// ---------------------------------------------------------------------------
// Euclidean tests
// ---------------------------------------------------------------------------

#[test]
fn test_euclidean_distances() {
    let e = Euclidean;
    // (0, 0) and (3, 4) -> distance = 5
    let pts = vec![0.0, 0.0, 3.0, 4.0];
    let dist = e.pairwise_distances(&pts, 2, 2);
    assert!((dist[1] - 5.0).abs() < 1e-10);
}

#[test]
fn test_euclidean_distances_symmetric() {
    let e = Euclidean;
    let mut rng = Rng::new(42);
    let pts: Vec<f64> = (0..20 * 3).map(|_| rng.normal()).collect();
    let dist = e.pairwise_distances(&pts, 20, 3);
    for i in 0..20 {
        for j in 0..20 {
            let diff = (dist[i * 20 + j] - dist[j * 20 + i]).abs();
            assert!(diff < 1e-10);
        }
    }
}

#[test]
fn test_euclidean_center() {
    let e = Euclidean;
    let mut pts = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 points in 2D
    e.center(&mut pts, 3, 2);

    // After centering, mean should be ~0
    for d in 0..2 {
        let mean: f64 = (0..3).map(|i| pts[i * 2 + d]).sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-10, "Mean not zero after centering: {mean}");
    }
}

// ---------------------------------------------------------------------------
// Tangent space projection tests
// ---------------------------------------------------------------------------

#[test]
fn test_hyperboloid_tangent_is_tangent() {
    let h = Hyperboloid::new(-1.0);
    let n = 5;
    let ambient = 3;
    let pts = h.init_points(n, 2, 0.1, 42);

    let mut rng = Rng::new(42);
    let mut grad: Vec<f64> = (0..n * ambient).map(|_| rng.normal()).collect();
    h.project_to_tangent(&pts, &mut grad, n, ambient);

    // Tangent vector v at x should satisfy <x, v>_L = 0
    for i in 0..n {
        let offset = i * ambient;
        let lorentz_inner = -pts[offset] * grad[offset]
            + pts[offset + 1] * grad[offset + 1]
            + pts[offset + 2] * grad[offset + 2];
        assert!(
            lorentz_inner.abs() < 1e-6,
            "Tangent not perpendicular at point {i}: <x,v>_L = {lorentz_inner}"
        );
    }
}

#[test]
fn test_sphere_tangent_is_tangent() {
    let s = Sphere::new(1.0);
    let n = 5;
    let ambient = 3;
    let pts = s.init_points(n, 2, 0.1, 42);

    let mut rng = Rng::new(42);
    let mut grad: Vec<f64> = (0..n * ambient).map(|_| rng.normal()).collect();
    s.project_to_tangent(&pts, &mut grad, n, ambient);

    // Tangent vector v at x should satisfy <x, v> = 0
    for i in 0..n {
        let offset = i * ambient;
        let inner: f64 = (0..ambient)
            .map(|d| pts[offset + d] * grad[offset + d])
            .sum();
        assert!(
            inner.abs() < 1e-6,
            "Tangent not perpendicular at point {i}: <x,v> = {inner}"
        );
    }
}
