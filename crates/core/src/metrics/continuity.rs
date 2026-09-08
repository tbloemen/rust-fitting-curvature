//! Continuity (Venna & Kaski 2006), and its two registry entries.

use super::helpers::{compute_ranks, knn_index_sets};
use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::context::EmbeddingContext;

/// Continuity (Venna & Kaski 2006).
///
/// Measures whether points that are neighbors in the original space remain
/// neighbors in the embedding. Penalizes "missed neighbors" from the original.
/// Returns a value in [0, 1], higher is better.
#[must_use]
pub fn continuity(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    let k = k.min(n - 2);
    if k == 0 || n < 3 {
        return 1.0;
    }

    let ranks_embed = compute_ranks(embedded_distances, n);
    let high_knn = knn_index_sets(high_dim_distances, n, k);

    let denom = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0);
    if denom < 1e-12 {
        return 1.0;
    }

    let mut penalty = 0.0;
    for i in 0..n {
        for &j in &high_knn[i] {
            let r = ranks_embed[i * n + j];
            if r > k {
                penalty += (r - k) as f64;
            }
        }
    }

    1.0 - (2.0 / denom) * penalty
}

/// Continuity on the 2-D projection.
pub struct Continuity;
impl QualityMetric for Continuity {
    fn name(&self) -> &'static str {
        "continuity"
    }
    fn base(&self) -> &'static str {
        "continuity"
    }
    fn space(&self) -> Space {
        Space::Projected
    }
    fn family(&self) -> Family {
        Family::Structure
    }
    fn direction(&self) -> Direction {
        Direction::Maximize
    }
    fn is_objective(&self) -> bool {
        true
    }
    fn short(&self) -> &'static str {
        "cont"
    }
    fn label(&self) -> &'static str {
        "Continuity"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        MetricValue::measured(continuity(c.high_dim_dist, c.dist_2d(), c.n, c.k))
    }
}

/// Continuity read on the manifold geodesics. See
/// [`TrustworthinessManifold`](super::trustworthiness::TrustworthinessManifold).
pub struct ContinuityManifold;
impl QualityMetric for ContinuityManifold {
    fn name(&self) -> &'static str {
        "continuity_manifold"
    }
    fn base(&self) -> &'static str {
        "continuity"
    }
    fn space(&self) -> Space {
        Space::Manifold
    }
    fn family(&self) -> Family {
        Family::Structure
    }
    fn direction(&self) -> Direction {
        Direction::Maximize
    }
    fn is_objective(&self) -> bool {
        false
    }
    fn short(&self) -> &'static str {
        "cont_m"
    }
    fn label(&self) -> &'static str {
        "Continuity"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        MetricValue::measured(continuity(c.high_dim_dist, c.manifold_dist(), c.n, c.k))
    }
}
