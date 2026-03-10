//! Tests for t-distribution kernel functions.
//! Ported from Python test/test_tsne.py::TestKernels

use fitting_core::config::ScalingLossType;
use fitting_core::kernels::{compute_q_matrix, t_distribution_kernel};
use fitting_core::manifolds::{Euclidean, Hyperboloid, Sphere};

#[test]
fn test_t_distribution_values() {
    let distances = vec![0.0, 1.0, 2.0, 10.0];
    // With dof=1 (Cauchy kernel): k(d) = 1 / (1 + d^2)
    let kernel_vals = t_distribution_kernel(&distances, 1.0);
    let expected = [1.0, 0.5, 0.2, 1.0 / 101.0];

    for (i, (&got, &exp)) in kernel_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "Kernel value {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_t_distribution_monotonic() {
    let distances: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let kernel_vals = t_distribution_kernel(&distances, 1.0);

    for i in 1..kernel_vals.len() {
        assert!(
            kernel_vals[i] <= kernel_vals[i - 1],
            "Kernel not monotonically decreasing at index {i}"
        );
    }
}

#[test]
fn test_q_matrix_normalization() {
    let n = 50;
    let manifold = Euclidean;
    let mut rng = fitting_core::synthetic_data::Rng::new(42);
    let points: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();

    let q = compute_q_matrix(&manifold, &points, n, 2, 1.0);
    let sum: f64 = q.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Q matrix sums to {sum}, not 1");
}

#[test]
fn test_q_matrix_symmetry() {
    let n = 50;
    let manifold = Euclidean;
    let mut rng = fitting_core::synthetic_data::Rng::new(42);
    let points: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();

    let q = compute_q_matrix(&manifold, &points, n, 2, 1.0);
    for i in 0..n {
        for j in 0..n {
            let diff = (q[i * n + j] - q[j * n + i]).abs();
            assert!(diff < 1e-6, "Q not symmetric at ({i},{j})");
        }
    }
}

#[test]
fn test_q_matrix_diagonal_zero() {
    let n = 50;
    let manifold = Euclidean;
    let mut rng = fitting_core::synthetic_data::Rng::new(42);
    let points: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();

    let q = compute_q_matrix(&manifold, &points, n, 2, 1.0);
    for i in 0..n {
        assert_eq!(q[i * n + i], 0.0, "Q diagonal not zero at {i}");
    }
}

#[test]
fn test_q_matrix_hyperbolic() {
    let n = 30;
    let h = Hyperboloid::new(-1.0, ScalingLossType::HardBarrier);
    let ambient = 3;

    // Initialize points on hyperboloid
    let mut rng = fitting_core::synthetic_data::Rng::new(42);
    let mut points = vec![0.0; n * ambient];
    for i in 0..n {
        let x1 = rng.normal() * 0.1;
        let x2 = rng.normal() * 0.1;
        let x0 = (1.0 + x1 * x1 + x2 * x2).sqrt();
        points[i * 3] = x0;
        points[i * 3 + 1] = x1;
        points[i * 3 + 2] = x2;
    }

    let q = compute_q_matrix(&h, &points, n, ambient, 1.0);
    let sum: f64 = q.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(!q.iter().any(|v| v.is_nan()));
}

#[test]
fn test_q_matrix_spherical() {
    let n = 30;
    let s = Sphere::new(1.0);
    let ambient = 3;

    let mut rng = fitting_core::synthetic_data::Rng::new(42);
    let mut points = vec![0.0; n * ambient];
    for i in 0..n {
        let x1 = rng.normal() * 0.1;
        let x2 = rng.normal() * 0.1;
        let sq = (x1 * x1 + x2 * x2).min(0.99);
        let x0 = (1.0 - sq).sqrt();
        points[i * 3] = x0;
        points[i * 3 + 1] = x1;
        points[i * 3 + 2] = x2;
    }

    let q = compute_q_matrix(&s, &points, n, ambient, 1.0);
    let sum: f64 = q.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(!q.iter().any(|v| v.is_nan()));
}
