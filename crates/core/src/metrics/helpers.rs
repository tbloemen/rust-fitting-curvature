//! The handful of routines shared by more than one metric.
//!
//! A helper only earns a place here once a second metric needs it; anything
//! used by exactly one lives in that metric's own module, next to the function
//! it serves — `fractional_rank_vector` in [`super::shepard_goodness`],
//! `davies_bouldin` in [`super::davies_bouldin_ratio`].

/// Compute ranks: for each point i, ranks[i*n + j] = rank of j sorted by
/// distance from i (0 = self, 1 = nearest neighbor, etc.).
pub(super) fn compute_ranks(distances: &[f64], n: usize) -> Vec<usize> {
    let mut ranks = vec![0usize; n * n];
    for i in 0..n {
        let mut indices: Vec<usize> = (0..n).collect();
        // total_cmp gives a proper total order (NaN/inf sort last), so sort_by
        // cannot panic when a diverged embedding produces non-finite distances.
        indices.sort_by(|&a, &b| distances[i * n + a].total_cmp(&distances[i * n + b]));
        for (rank, &j) in indices.iter().enumerate() {
            ranks[i * n + j] = rank;
        }
    }
    ranks
}

/// Compute k-nearest neighbor index sets (excluding self).
pub(super) fn knn_index_sets(dist: &[f64], n: usize, k: usize) -> Vec<Vec<usize>> {
    (0..n)
        .map(|i| {
            let mut indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            indices.sort_by(|&a, &b| dist[i * n + a].total_cmp(&dist[i * n + b]));
            indices.truncate(k);
            indices
        })
        .collect()
}

/// Compute pairwise Euclidean distance matrix from 2D points (flat [x,y] pairs).
///
/// Use this to obtain "after-projection" distances from the output of
/// `visualisation::project_to_2d`, so that any metric can be evaluated on
/// what the viewer actually sees rather than on the manifold geometry.
#[must_use]
pub fn euclidean_dist_2d(pts_2d: &[f64], n: usize) -> Vec<f64> {
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pts_2d[i * 2] - pts_2d[j * 2];
            let dy = pts_2d[i * 2 + 1] - pts_2d[j * 2 + 1];
            let d = (dx * dx + dy * dy).sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}
