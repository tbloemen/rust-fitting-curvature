//! Utility functions for data normalization and initialization.
//!
//! Ported from Python `src/matrices.py`.

use crate::synthetic_data::Rng;

/// Normalize data so that mean pairwise distance equals 1.
///
/// This makes embeddings geometry-agnostic w.r.t. initialization scale.
pub fn normalize_data(data: &mut [f64], n_points: usize, n_features: usize, seed: u64) {
    let mut rng = Rng::new(seed);

    // Sample random pairs to estimate mean distance
    let n_samples = 10000.min(n_points * (n_points - 1) / 2);
    let mut total_dist = 0.0;

    for _ in 0..n_samples {
        let i = (rng.uniform() * n_points as f64) as usize % n_points;
        let mut j = (rng.uniform() * n_points as f64) as usize % n_points;
        while j == i {
            j = (rng.uniform() * n_points as f64) as usize % n_points;
        }

        let mut sq = 0.0;
        for f in 0..n_features {
            let diff = data[i * n_features + f] - data[j * n_features + f];
            sq += diff * diff;
        }
        total_dist += sq.sqrt();
    }

    let mean_dist = total_dist / n_samples as f64;
    if mean_dist > 1e-12 {
        for val in data.iter_mut() {
            *val /= mean_dist;
        }
    }
}

/// Get a principled initialization scale for normalized data.
///
/// For data with mean pairwise distance = 1, returns σ such that
/// points initialized as N(0, σ²I) will have expected pairwise distance ≈ 1.
///
/// For Gaussian points in d dimensions: E[||x-y||] ≈ σ * sqrt(2d),
/// so σ = 1 / sqrt(2d) gives E[||x-y||] = 1.
pub fn get_default_init_scale(embed_dim: usize) -> f64 {
    1.0 / (2.0 * embed_dim as f64).sqrt()
}

/// Compute pairwise Euclidean distance matrix from data.
///
/// Returns flat n × n row-major matrix.
pub fn compute_euclidean_distance_matrix(
    data: &[f64],
    n_points: usize,
    n_features: usize,
) -> Vec<f64> {
    let mut dist = vec![0.0; n_points * n_points];
    for i in 0..n_points {
        for j in (i + 1)..n_points {
            let mut sq = 0.0;
            for f in 0..n_features {
                let diff = data[i * n_features + f] - data[j * n_features + f];
                sq += diff * diff;
            }
            let d = sq.sqrt();
            dist[i * n_points + j] = d;
            dist[j * n_points + i] = d;
        }
    }
    dist
}
