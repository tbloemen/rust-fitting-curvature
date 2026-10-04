//! The Dunn index, and its registry entry.

use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::context::EmbeddingContext;

/// Dunn index: ratio of minimum inter-cluster distance to maximum intra-cluster diameter.
/// Higher = better clustering.
#[must_use]
pub fn dunn_index(embedded_distances: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    // Cluster indices
    let cluster_indices: Vec<Vec<usize>> = unique_labels
        .iter()
        .map(|&lbl| (0..n).filter(|&i| labels[i] == lbl).collect())
        .collect();

    // Max intra-cluster diameter
    let mut max_intra = 0.0f64;
    for indices in &cluster_indices {
        for &a in indices {
            for &b in indices {
                let d = embedded_distances[a * n + b];
                if d > max_intra {
                    max_intra = d;
                }
            }
        }
    }
    if max_intra < 1e-12 {
        return 0.0;
    }

    // Min inter-cluster distance
    let mut min_inter = f64::INFINITY;
    for ci in 0..k {
        for cj in (ci + 1)..k {
            for &a in &cluster_indices[ci] {
                for &b in &cluster_indices[cj] {
                    let d = embedded_distances[a * n + b];
                    if d < min_inter {
                        min_inter = d;
                    }
                }
            }
        }
    }

    min_inter / max_intra
}

/// Dunn index: min inter-cluster distance over max intra-cluster diameter.
///
/// A ratio, so unbounded above — see [`QualityMetric::is_objective`].
pub struct DunnIndex;
impl QualityMetric for DunnIndex {
    fn name(&self) -> &'static str {
        "dunn_index"
    }
    fn base(&self) -> &'static str {
        "dunn_index"
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
        "dunn"
    }
    fn label(&self) -> &'static str {
        "Dunn Index"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        match c.labels {
            Some(l) => MetricValue::measured(dunn_index(c.dist_2d(), l, c.n)),
            None => MetricValue::NotApplicable,
        }
    }
}
