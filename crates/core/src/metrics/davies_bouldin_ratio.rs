//! The Davies-Bouldin ratio (Di Caro et al. 2010), its underlying DB index,
//! and its registry entry.

use super::helpers::euclidean_dist_2d;
use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::cast::count_to_f64;
use crate::context::EmbeddingContext;

/// Davies-Bouldin index from precomputed distance matrix.
/// Lower = better separated, more compact clusters.
#[must_use]
pub fn davies_bouldin(distances: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    let cluster_indices: Vec<Vec<usize>> = unique_labels
        .iter()
        .map(|&lbl| (0..n).filter(|&i| labels[i] == lbl).collect())
        .collect();

    // Compute scatters and medoids
    let mut scatters = vec![0.0f64; k];
    let mut medoid_indices = vec![0usize; k];

    for (ci, indices) in cluster_indices.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        // Find medoid (point with smallest total distance to others)
        let mut best_total = f64::INFINITY;
        for &candidate in indices {
            let total: f64 = indices.iter().map(|&j| distances[candidate * n + j]).sum();
            if total < best_total {
                best_total = total;
                medoid_indices[ci] = candidate;
            }
        }
        // Scatter: mean distance to medoid
        let medoid = medoid_indices[ci];
        scatters[ci] = indices
            .iter()
            .map(|&j| distances[medoid * n + j])
            .sum::<f64>()
            / count_to_f64(indices.len());
    }

    // DB index
    let mut db = 0.0;
    for i in 0..k {
        let mut max_ratio = 0.0f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let d_ij = distances[medoid_indices[i] * n + medoid_indices[j]];
            if d_ij < 1e-12 {
                continue;
            }
            let ratio = (scatters[i] + scatters[j]) / d_ij;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        db += max_ratio;
    }
    db / count_to_f64(k)
}

/// Davies-Bouldin ratio: `DB_high` / `DB_projected`.
///
/// Computes the DB index on both the high-dimensional data distances and
/// the 2D projected Euclidean distances. A higher ratio indicates the
/// projection preserves or improves cluster separation relative to the
/// original data (Di Caro et al. 2010).
#[must_use]
pub fn davies_bouldin_ratio(
    high_dim_distances: &[f64],
    pts_2d: &[f64],
    labels: &[u32],
    n: usize,
) -> f64 {
    let dist_2d = euclidean_dist_2d(pts_2d, n);
    davies_bouldin_ratio_from(high_dim_distances, &dist_2d, labels, n)
}

/// [`davies_bouldin_ratio`] against a projected distance matrix the caller
/// already holds.
///
/// The wrapper above derives that matrix from coordinates on every call, which
/// is a wasted `O(n²)` for a caller — [`EmbeddingContext`] — that has it
/// cached. The two agree bit-for-bit: `euclidean_dist_2d` and
/// `matrices::compute_euclidean_distance_matrix` differ only by a leading
/// `0.0 +`, which is exact.
#[must_use]
pub fn davies_bouldin_ratio_from(
    high_dim_distances: &[f64],
    dist_2d: &[f64],
    labels: &[u32],
    n: usize,
) -> f64 {
    let db_high = davies_bouldin(high_dim_distances, labels, n);
    let db_proj = davies_bouldin(dist_2d, labels, n);
    if db_proj < 1e-12 {
        return 0.0;
    }
    db_high / db_proj
}

/// Davies-Bouldin ratio `DB_high / DB_projected` (Di Caro et al. 2010).
///
/// A ratio, so unbounded above — see [`QualityMetric::is_objective`].
pub struct DaviesBouldinRatio;
impl QualityMetric for DaviesBouldinRatio {
    fn name(&self) -> &'static str {
        "davies_bouldin_ratio"
    }
    fn base(&self) -> &'static str {
        "davies_bouldin_ratio"
    }
    fn space(&self) -> Space {
        Space::Projected
    }
    fn family(&self) -> Family {
        Family::ClassSeparation
    }
    fn direction(&self) -> Direction {
        Direction::Maximize
    }
    fn is_objective(&self) -> bool {
        false
    }
    fn short(&self) -> &'static str {
        "db"
    }
    fn label(&self) -> &'static str {
        "DB Ratio"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        match c.labels {
            Some(l) => MetricValue::measured(davies_bouldin_ratio_from(
                c.high_dim_dist,
                c.dist_2d(),
                l,
                c.n,
            )),
            None => MetricValue::NotApplicable,
        }
    }
}
