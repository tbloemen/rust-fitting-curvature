//! Tests for affinity computation.
//! Ported from Python test/test_tsne.py::TestAffinities

use fitting_core::affinities::{binary_search_sigma, compute_perplexity_affinities};
use fitting_core::synthetic_data::Rng;

fn random_data(n: usize, d: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n * d).map(|_| rng.normal()).collect()
}

#[test]
fn test_binary_search_convergence() {
    let mut rng = Rng::new(42);
    let distances: Vec<f64> = (0..30).map(|_| rng.uniform() * 10.0).collect();
    let target_perplexity = 15.0;

    let sigma = binary_search_sigma(&distances, target_perplexity);

    assert!(sigma > 0.0);
    assert!(sigma < 1e4);

    // Verify perplexity is close to target
    let two_sigma_sq = 2.0 * sigma * sigma;
    let max_val = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = distances
        .iter()
        .map(|&d| (-d / two_sigma_sq - (-max_val / two_sigma_sq)).exp())
        .collect();
    let sum_exp: f64 = exps.iter().sum();
    let mut entropy = 0.0;
    for &e in &exps {
        let p = e / sum_exp;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    let perplexity = entropy.exp();

    assert!(
        (perplexity - target_perplexity).abs() < 1.0,
        "Perplexity {perplexity} not close to target {target_perplexity}"
    );
}

#[test]
fn test_affinity_symmetry() {
    let data = random_data(50, 10, 42);
    let v = compute_perplexity_affinities(&data, 50, 10, 15.0);

    for i in 0..50 {
        for j in 0..50 {
            let diff = (v[i * 50 + j] - v[j * 50 + i]).abs();
            assert!(
                diff < 1e-6,
                "Affinity not symmetric at ({i},{j}): {} vs {}",
                v[i * 50 + j],
                v[j * 50 + i]
            );
        }
    }
}

#[test]
fn test_affinity_normalization() {
    let data = random_data(50, 10, 42);
    let v = compute_perplexity_affinities(&data, 50, 10, 15.0);
    let sum: f64 = v.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Affinities sum to {sum}, not 1");
}

#[test]
fn test_affinity_non_negative() {
    let data = random_data(50, 10, 42);
    let v = compute_perplexity_affinities(&data, 50, 10, 15.0);
    for (i, &val) in v.iter().enumerate() {
        assert!(val >= 0.0, "Negative affinity at index {i}: {val}");
    }
}

#[test]
fn test_affinity_diagonal_near_zero() {
    let data = random_data(50, 10, 42);
    let v = compute_perplexity_affinities(&data, 50, 10, 15.0);
    for i in 0..50 {
        assert!(
            v[i * 50 + i] < 0.01,
            "Diagonal too large at {i}: {}",
            v[i * 50 + i]
        );
    }
}
