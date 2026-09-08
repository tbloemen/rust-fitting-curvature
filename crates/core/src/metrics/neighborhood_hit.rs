//! Neighborhood hit (van der Maaten 2009), and its two registry entries.

use super::helpers::knn_index_sets;
use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::context::EmbeddingContext;

/// Neighborhood hit (van der Maaten 2009).
///
/// Measures how well the k-nearest-neighbor structure in the embedding aligns
/// with class labels: for each point, the fraction of its k-NN in the
/// embedding that share the same label, averaged over all points.
///
/// `M_NH = (1/N) * sum_i |{j ∈ kNN_embed(i) : label[j] = label[i]}| / k`
///
/// Returns a value in [0, 1], with **1 being best** (all k-NN same-class).
/// Requires labeled data. Pass manifold geodesic or 2D Euclidean distances
/// for the before/after projection distinction.
#[must_use]
pub fn neighborhood_hit(embedded_distances: &[f64], labels: &[u32], n: usize, k: usize) -> f64 {
    let k = k.min(n - 1);
    if k == 0 {
        return 1.0;
    }

    let knn = knn_index_sets(embedded_distances, n, k);
    let mut total = 0.0;
    for i in 0..n {
        let same = knn[i].iter().filter(|&&j| labels[j] == labels[i]).count();
        total += same as f64 / k as f64;
    }
    total / n as f64
}

/// Neighborhood hit on the 2-D projection.
///
/// It sits in [`Family::ClassSeparation`], not `Structure`: it is the fraction
/// of a point's k nearest neighbours sharing its label, so it reads `labels`
/// and never touches the high-dimensional data. Its resemblance to
/// trustworthiness/continuity is that it is a k-NN statistic at the same `k` —
/// a computational similarity, not a semantic one.
pub struct NeighborhoodHit;
impl QualityMetric for NeighborhoodHit {
    fn name(&self) -> &'static str {
        "neighborhood_hit"
    }
    fn base(&self) -> &'static str {
        "neighborhood_hit"
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
        true
    }
    fn short(&self) -> &'static str {
        "nh"
    }
    fn label(&self) -> &'static str {
        "Neighborhood Hit"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        match c.labels {
            Some(l) => MetricValue::measured(neighborhood_hit(c.dist_2d(), l, c.n, c.k)),
            None => MetricValue::NotApplicable,
        }
    }
}

/// Neighborhood hit read on the manifold geodesics. See
/// [`TrustworthinessManifold`](super::trustworthiness::TrustworthinessManifold).
pub struct NeighborhoodHitManifold;
impl QualityMetric for NeighborhoodHitManifold {
    fn name(&self) -> &'static str {
        "neighborhood_hit_manifold"
    }
    fn base(&self) -> &'static str {
        "neighborhood_hit"
    }
    fn space(&self) -> Space {
        Space::Manifold
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
        "nh_m"
    }
    fn label(&self) -> &'static str {
        "Neighborhood Hit"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        match c.labels {
            Some(l) => MetricValue::measured(neighborhood_hit(c.manifold_dist(), l, c.n, c.k)),
            None => MetricValue::NotApplicable,
        }
    }
}
