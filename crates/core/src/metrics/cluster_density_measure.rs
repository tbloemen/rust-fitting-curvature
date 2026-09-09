//! The Cluster Density Measure (Albuquerque et al. 2010), and its registry
//! entry.

use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::cast::count_to_f64;
use crate::context::EmbeddingContext;

/// Cluster Density Measure (`ClDM`) from Albuquerque et al. (2010).
///
/// Uses the label-based cluster formula on 2D projected coordinates:
/// `ClDM` = (1/K) * sum_{k<l} d²_{k,l} / (`r_k` * `r_l`)
///
/// Measures how well-separated and compact the clusters are.
/// Higher values = better separated clusters.
///
/// # Panics
///
/// Panics if `labels[i]` is not found in the deduplicated label set.
#[must_use]
pub fn cluster_density_measure(pts_2d: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    // Compute centroids and radii per cluster
    let mut centroids = vec![(0.0f64, 0.0f64); k];
    let mut counts = vec![0usize; k];
    let mut radii = vec![0.0f64; k];

    for i in 0..n {
        let label_idx = unique_labels.iter().position(|&l| l == labels[i]).unwrap();
        centroids[label_idx].0 += pts_2d[i * 2];
        centroids[label_idx].1 += pts_2d[i * 2 + 1];
        counts[label_idx] += 1;
    }
    for ci in 0..k {
        if counts[ci] > 0 {
            centroids[ci].0 /= count_to_f64(counts[ci]);
            centroids[ci].1 /= count_to_f64(counts[ci]);
        }
    }

    // Compute average radius per cluster
    for i in 0..n {
        let label_idx = unique_labels.iter().position(|&l| l == labels[i]).unwrap();
        let dx = pts_2d[i * 2] - centroids[label_idx].0;
        let dy = pts_2d[i * 2 + 1] - centroids[label_idx].1;
        radii[label_idx] += (dx * dx + dy * dy).sqrt();
    }
    for ci in 0..k {
        radii[ci] = if counts[ci] > 0 {
            (radii[ci] / count_to_f64(counts[ci])).max(1e-12)
        } else {
            1e-12
        };
    }

    let mut cldm = 0.0;
    for ki in 0..k {
        for kj in (ki + 1)..k {
            let dx = centroids[ki].0 - centroids[kj].0;
            let dy = centroids[ki].1 - centroids[kj].1;
            let d_sq = dx * dx + dy * dy;
            cldm += d_sq / (radii[ki] * radii[kj]);
        }
    }
    cldm / count_to_f64(k)
}

/// Cluster Density Measure (Albuquerque et al. 2010).
///
/// A ratio, so unbounded above — the worst of the three, reaching 2e24 over
/// `results/` when collapsed clusters hit the `1e-12` radius floor in the
/// formula. See [`QualityMetric::is_objective`].
pub struct ClusterDensityMeasure;
impl QualityMetric for ClusterDensityMeasure {
    fn name(&self) -> &'static str {
        "cluster_density_measure"
    }
    fn base(&self) -> &'static str {
        "cluster_density_measure"
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
        "cldm"
    }
    fn label(&self) -> &'static str {
        "Cluster Density"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        match c.labels {
            Some(l) => MetricValue::measured(cluster_density_measure(c.coords_2d(), l, c.n)),
            None => MetricValue::NotApplicable,
        }
    }
}
