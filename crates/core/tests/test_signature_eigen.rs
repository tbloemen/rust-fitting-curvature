//! Tests for the eigenvector-accumulating decomposition added for
//! `curvature_detection::reconstruct`.
//!
//! The load-bearing test here is [`eigen_agrees_with_eigenvalue_only`]: the
//! radius search uses the cheap eigenvalue-only path and the reconstruction
//! uses the full one, so the two must not drift. They share `tridiagonalise`
//! and `ql_implicit`, and this pins that they keep agreeing.

use fitting_core::curvature_detection::{eigen_symmetric, eigenvalues_symmetric, Eigen};
use fitting_core::synthetic_data::{generate_uniform_hyperbolic, generate_uniform_sphere};

/// xoshiro-free deterministic filler — these matrices only need to be
/// reproducible, not statistically good.
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn random_symmetric(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = lcg(&mut s);
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
    a
}

/// `max |(VᵀV)_{ij} − δ_{ij}|`.
fn orthogonality_error(e: &Eigen, n: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for p in 0..n {
        for q in 0..n {
            let mut dot = 0.0;
            for i in 0..n {
                dot += e.vectors[i * n + p] * e.vectors[i * n + q];
            }
            let target = if p == q { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    worst
}

/// `max |A − V Λ Vᵀ|`.
fn reconstruction_error(a: &[f64], e: &Eigen, n: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut v = 0.0;
            for k in 0..n {
                v += e.vectors[i * n + k] * e.values[k] * e.vectors[j * n + k];
            }
            worst = worst.max((a[i * n + j] - v).abs());
        }
    }
    worst
}

#[test]
fn eigen_agrees_with_eigenvalue_only() {
    for (n, seed) in [(4, 1), (17, 2), (60, 3)] {
        let a = random_symmetric(n, seed);
        let cheap = eigenvalues_symmetric(&a, n);
        let full = eigen_symmetric(&a, n);
        assert_eq!(cheap.len(), full.values.len());
        for (i, (c, f)) in cheap.iter().zip(&full.values).enumerate() {
            assert!(
                (c - f).abs() < 1e-10 * (1.0 + c.abs()),
                "n={n} eigenvalue {i} drifted: eigenvalues_symmetric={c}, eigen_symmetric={f}"
            );
        }
    }
}

#[test]
fn eigenvectors_are_orthonormal() {
    for (n, seed) in [(4, 11), (17, 12), (60, 13)] {
        let a = random_symmetric(n, seed);
        let e = eigen_symmetric(&a, n);
        let err = orthogonality_error(&e, n);
        assert!(err < 1e-10, "n={n}: VᵀV deviates from I by {err}");
    }
}

#[test]
fn decomposition_reconstructs_the_matrix() {
    for (n, seed) in [(4, 21), (17, 22), (60, 23)] {
        let a = random_symmetric(n, seed);
        let e = eigen_symmetric(&a, n);
        let err = reconstruction_error(&a, &e, n);
        assert!(err < 1e-9, "n={n}: A - VΛVᵀ off by {err}");
    }
}

#[test]
fn values_are_ascending() {
    let a = random_symmetric(30, 31);
    let e = eigen_symmetric(&a, 30);
    for w in e.values.windows(2) {
        assert!(w[0] <= w[1], "eigenvalues not ascending: {:?}", e.values);
    }
}

#[test]
fn handles_degenerate_sizes() {
    let empty = eigen_symmetric(&[], 0);
    assert!(empty.values.is_empty() && empty.vectors.is_empty());

    let one = eigen_symmetric(&[3.5], 1);
    assert_eq!(one.values, vec![3.5]);
    assert_eq!(one.vectors, vec![1.0]);

    // A zero matrix has no distinguished directions; the basis must still be
    // orthonormal rather than degenerate.
    let n = 5;
    let e = eigen_symmetric(&vec![0.0; n * n], n);
    assert!(e.values.iter().all(|v| v.abs() < 1e-12));
    assert!(orthogonality_error(&e, n) < 1e-12);
}

/// The decomposition is used on `Z(r)` matrices, which are far from generic:
/// near-rank-deficient with a big spectral gap. Exercise a real one.
#[test]
fn works_on_a_real_z_matrix() {
    let data = generate_uniform_sphere(60, 42);
    let n = data.n_points;
    // `Z_spherical(r) = r² cos(d/r)` at the unit radius the generator uses.
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            z[i * n + j] = data.distances[i * n + j].cos();
        }
    }
    let e = eigen_symmetric(&z, n);
    assert!(orthogonality_error(&e, n) < 1e-9);
    assert!(reconstruction_error(&z, &e, n) < 1e-9);

    // Points on S² span R³, so Z is PSD of rank 3: three big eigenvalues and
    // a tail that is numerically zero.
    let tail: f64 = e.values[..n - 3].iter().map(|v| v.abs()).sum();
    assert!(tail < 1e-8, "S² Z-matrix tail should vanish, got {tail}");
    assert!(e.values[n - 3] > 1.0);
}

#[test]
fn works_on_a_hyperbolic_z_matrix() {
    let data = generate_uniform_hyperbolic(60, 42, 3.0);
    let n = data.n_points;
    // `Z_hyperbolic(r) = −r² cosh(d/r)` at unit radius.
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            z[i * n + j] = -data.distances[i * n + j].cosh();
        }
    }
    let e = eigen_symmetric(&z, n);
    assert!(orthogonality_error(&e, n) < 1e-9);
    assert!(reconstruction_error(&z, &e, n) < 1e-8);

    // Signature (2 positive, 1 negative) for H²: one strongly negative
    // eigenvalue, two positive, and a vanishing middle.
    let middle: f64 = e.values[1..n - 2].iter().map(|v| v.abs()).sum();
    assert!(e.values[0] < -1.0, "expected a Lorentzian time axis");
    assert!(middle < 1e-7, "H² Z-matrix middle should vanish, got {middle}");
}
