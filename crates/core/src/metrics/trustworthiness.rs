//! Trustworthiness (Venna & Kaski 2006), and its two registry entries.

use super::helpers::{compute_ranks, knn_index_sets};
use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::context::EmbeddingContext;

/// Trustworthiness (Venna & Kaski 2006).
///
/// Measures whether points that appear as neighbors in the embedding are also
/// neighbors in the original space. Penalizes "false neighbors" in the embedding.
/// Returns a value in [0, 1], higher is better.
#[must_use]
pub fn trustworthiness(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    let k = k.min(n - 2);
    if k == 0 || n < 3 {
        return 1.0;
    }

    let ranks_high = compute_ranks(high_dim_distances, n);
    let embed_knn = knn_index_sets(embedded_distances, n, k);

    let denom = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0);
    if denom < 1e-12 {
        return 1.0;
    }

    let mut penalty = 0.0;
    for i in 0..n {
        for &j in &embed_knn[i] {
            // ranks_high uses 0-based ranks where 0 = self, so rank > k means
            // j is NOT among i's k nearest in high-dim space
            let r = ranks_high[i * n + j];
            if r > k {
                penalty += (r - k) as f64;
            }
        }
    }

    1.0 - (2.0 / denom) * penalty
}

/// Trustworthiness on the 2-D projection.
pub struct Trustworthiness;
impl QualityMetric for Trustworthiness {
    fn name(&self) -> &'static str {
        "trustworthiness"
    }
    fn base(&self) -> &'static str {
        "trustworthiness"
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
        "trust"
    }
    fn label(&self) -> &'static str {
        "Trustworthiness"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        MetricValue::measured(trustworthiness(c.high_dim_dist, c.dist_2d(), c.n, c.k))
    }
}

/// Trustworthiness read on the manifold geodesics, before projection.
///
/// Measured but not optimised: what the thesis judges is the 2-D visualisation,
/// so the manifold half of the objective budget was optimising a surface no
/// reader looks at. `figures/exp4.rs` plots this against its twin, and is the
/// evidence for having dropped it.
pub struct TrustworthinessManifold;
impl QualityMetric for TrustworthinessManifold {
    fn name(&self) -> &'static str {
        "trustworthiness_manifold"
    }
    fn base(&self) -> &'static str {
        "trustworthiness"
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
        "trust_m"
    }
    fn label(&self) -> &'static str {
        "Trustworthiness"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        MetricValue::measured(trustworthiness(
            c.high_dim_dist,
            c.manifold_dist(),
            c.n,
            c.k,
        ))
    }
}
