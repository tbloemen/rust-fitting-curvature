//! Scale-normalized stress (Damrich & Hamprecht 2022), and its two registry
//! entries.

use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::context::EmbeddingContext;

/// Scale-normalized stress (SNS) from Damrich & Hamprecht 2022.
///
/// Standard normalized stress is scale-sensitive: embeddings that preserve
/// distance shape but differ in overall scale appear poor. SNS removes this
/// by finding the optimal scale factor α before computing stress:
///
/// `α  = Σ_{i,j} d(x,x') · ‖y−y'‖ / Σ_{i,j} d(x,x')²`
/// `SNS = Σ_{i,j} [d(x,x') − α·‖y−y'‖]² / Σ_{i,j} d(x,x')²`
///
/// Returns a value in [0, 1], with **0 being best**. Because α is optimised
/// out, manifold and 2D variants are identical for Euclidean embeddings where
/// `project_to_2d` only rescales coordinates for display.
#[must_use]
pub fn normalized_stress(high_dim_distances: &[f64], embedded_distances: &[f64], n: usize) -> f64 {
    let mut cross = 0.0;
    let mut embed_sq = 0.0;
    let mut high_sq = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_h = high_dim_distances[i * n + j];
            let d_e = embedded_distances[i * n + j];
            cross += d_h * d_e;
            embed_sq += d_e * d_e;
            high_sq += d_h * d_h;
        }
    }
    if high_sq < 1e-12 {
        return 0.0;
    }
    let alpha = if embed_sq < 1e-12 {
        1.0
    } else {
        cross / embed_sq
    };

    let mut numerator = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_h = high_dim_distances[i * n + j];
            let d_e = embedded_distances[i * n + j];
            let diff = d_h - alpha * d_e;
            numerator += diff * diff;
        }
    }
    (numerator / high_sq).sqrt()
}

/// Scale-normalized stress on the 2-D projection.
///
/// The one minimised metric in the set, and the reason `Direction` exists:
/// `oriented_value` orients it as `1 − x` and `metrics_to_vec` substitutes the
/// *upper* bound for a diverged trial rather than the lower one.
pub struct NormalizedStress;
impl QualityMetric for NormalizedStress {
    fn name(&self) -> &'static str {
        "normalized_stress"
    }
    fn base(&self) -> &'static str {
        "normalized_stress"
    }
    fn space(&self) -> Space {
        Space::Projected
    }
    fn family(&self) -> Family {
        Family::Distance
    }
    fn direction(&self) -> Direction {
        Direction::Minimize
    }
    fn is_objective(&self) -> bool {
        true
    }
    fn short(&self) -> &'static str {
        "stress"
    }
    fn label(&self) -> &'static str {
        "Norm. Stress"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        MetricValue::measured(normalized_stress(c.high_dim_dist, c.dist_2d(), c.n))
    }
}

/// Scale-normalized stress on the manifold geodesics.
///
/// Because the optimal scale α is divided out, this is *identical* to its twin
/// for Euclidean embeddings, where `project_to_2d` only rescales coordinates
/// for display. The two readings separate only under curvature.
pub struct NormalizedStressManifold;
impl QualityMetric for NormalizedStressManifold {
    fn name(&self) -> &'static str {
        "normalized_stress_manifold"
    }
    fn base(&self) -> &'static str {
        "normalized_stress"
    }
    fn space(&self) -> Space {
        Space::Manifold
    }
    fn family(&self) -> Family {
        Family::Distance
    }
    fn direction(&self) -> Direction {
        Direction::Minimize
    }
    fn is_objective(&self) -> bool {
        false
    }
    fn short(&self) -> &'static str {
        "stress_m"
    }
    fn label(&self) -> &'static str {
        "Norm. Stress"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        MetricValue::measured(normalized_stress(c.high_dim_dist, c.manifold_dist(), c.n))
    }
}
