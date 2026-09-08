//! The pre-registry metrics aggregate.
//!
//! Superseded by [`super::MetricValues`], which covers the same quantities plus
//! the three spread diagnostics and carries its own wire format. Kept only
//! until `EmbeddingState` and `crates/web` are moved over.

use super::functions::*;

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// All quality metrics for a completed embedding, in both manifold and
/// 2D-projected variants (the "before" / "after projecting" distinction).
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    // A. Local structure preservation
    pub trustworthiness_manifold: f64,
    pub trustworthiness_2d: f64,
    pub continuity_manifold: f64,
    pub continuity_2d: f64,
    // B. Distance preservation
    pub normalized_stress_manifold: f64,
    pub normalized_stress_2d: f64,
    pub shepard_goodness_manifold: f64,
    pub shepard_goodness_2d: f64,
    // C. Label-dependent (None when no labels provided)
    pub neighborhood_hit_manifold: Option<f64>,
    pub neighborhood_hit_2d: Option<f64>,
    // D. Class separation — 2D only, label-dependent
    pub cluster_density_measure: Option<f64>,
    pub davies_bouldin_ratio: Option<f64>,
}

/// Compute a full metrics snapshot.
///
/// `high_dim_dist` — pairwise distances in input space.
/// `embed_dist`    — manifold geodesic distances (before projection).
/// `pts_2d`        — flat (x,y) pairs from `project_to_2d` (after projection).
/// `labels`        — optional class labels; label-dependent metrics are `None` when absent.
/// `k`             — neighbourhood size used for kNN metrics.
pub fn compute_snapshot(
    high_dim_dist: &[f64],
    embed_dist: &[f64],
    pts_2d: &[f64],
    labels: Option<&[u32]>,
    n: usize,
    k: usize,
) -> MetricsSnapshot {
    let dist_2d = euclidean_dist_2d(pts_2d, n);

    let (neighborhood_hit_manifold, neighborhood_hit_2d, cluster_density, db_ratio) =
        if let Some(lbl) = labels {
            (
                Some(neighborhood_hit(embed_dist, lbl, n, k)),
                Some(neighborhood_hit(&dist_2d, lbl, n, k)),
                Some(cluster_density_measure(pts_2d, lbl, n)),
                Some(davies_bouldin_ratio(high_dim_dist, pts_2d, lbl, n)),
            )
        } else {
            (None, None, None, None)
        };

    MetricsSnapshot {
        trustworthiness_manifold: trustworthiness(high_dim_dist, embed_dist, n, k),
        trustworthiness_2d: trustworthiness(high_dim_dist, &dist_2d, n, k),
        continuity_manifold: continuity(high_dim_dist, embed_dist, n, k),
        continuity_2d: continuity(high_dim_dist, &dist_2d, n, k),
        normalized_stress_manifold: normalized_stress(high_dim_dist, embed_dist, n),
        normalized_stress_2d: normalized_stress(high_dim_dist, &dist_2d, n),
        shepard_goodness_manifold: shepard_goodness(high_dim_dist, embed_dist, n),
        shepard_goodness_2d: shepard_goodness(high_dim_dist, &dist_2d, n),
        neighborhood_hit_manifold,
        neighborhood_hit_2d,
        cluster_density_measure: cluster_density,
        davies_bouldin_ratio: db_ratio,
    }
}
