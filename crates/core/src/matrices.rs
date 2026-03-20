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

/// PCA via power iteration with deflation.
///
/// Returns the projection of `data` onto its top `embed_dim` principal
/// components as a flat `n_points × embed_dim` row-major array.
/// Each output column has mean zero (inherited from centering the input).
pub fn pca(
    data: &[f64],
    n_points: usize,
    n_features: usize,
    embed_dim: usize,
    seed: u64,
) -> Vec<f64> {
    // Center the data column-wise.
    let mut centered = data.to_vec();
    for f in 0..n_features {
        let mean = (0..n_points).map(|i| data[i * n_features + f]).sum::<f64>() / n_points as f64;
        for i in 0..n_points {
            centered[i * n_features + f] -= mean;
        }
    }

    let mut rng = Rng::new(seed);
    let mut components: Vec<Vec<f64>> = Vec::with_capacity(embed_dim);

    for _ in 0..embed_dim {
        // Random unit vector in feature space.
        let mut v: Vec<f64> = (0..n_features).map(|_| rng.normal()).collect();
        vec_normalize(&mut v);

        for _ in 0..200 {
            // w = X v  (n_points)
            let mut w = vec![0.0f64; n_points];
            for i in 0..n_points {
                for f in 0..n_features {
                    w[i] += centered[i * n_features + f] * v[f];
                }
            }

            // v_new = X^T w  (n_features)
            let mut v_new = vec![0.0f64; n_features];
            for i in 0..n_points {
                for f in 0..n_features {
                    v_new[f] += centered[i * n_features + f] * w[i];
                }
            }

            // Orthogonalize against previously found components (deflation).
            for prev in &components {
                let dot: f64 = v_new.iter().zip(prev).map(|(a, b)| a * b).sum();
                for (a, b) in v_new.iter_mut().zip(prev) {
                    *a -= dot * b;
                }
            }

            vec_normalize(&mut v_new);

            // Check convergence (sign-agnostic).
            let change: f64 = v
                .iter()
                .zip(&v_new)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            v = v_new;
            if change < 1e-9 {
                break;
            }
        }
        components.push(v);
    }

    // Project centered data onto the components: result[i, k] = centered[i, :] · components[k]
    let mut result = vec![0.0f64; n_points * embed_dim];
    for i in 0..n_points {
        for k in 0..embed_dim {
            result[i * embed_dim + k] = (0..n_features)
                .map(|f| centered[i * n_features + f] * components[k][f])
                .sum();
        }
    }
    result
}

fn vec_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
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
