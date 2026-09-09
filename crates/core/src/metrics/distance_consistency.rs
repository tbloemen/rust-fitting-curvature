//! Distance consistency (Sips et al. 2009), and its registry entry.

use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::cast::count_to_f64;
use crate::context::EmbeddingContext;

/// Distance consistency (`DSC`) from Sips et al. (2009), as surveyed in
/// Bernard et al. (2018).
///
/// The fraction of points lying nearer to their own class centroid than to any
/// other class centroid:
///
/// `DSC = (1/N) * |{i : ||u_i - mu_{l_i}|| < min_{c != l_i} ||u_i - mu_c||}|`
///
/// where `u_i` is point `i` in the visualisation and `mu_c` is the centroid of
/// class `c`. In `[0, 1]` by construction, with **1 being best**, and it needs
/// neither a neighbourhood size nor a clustering step.
///
/// It is the *global* counterpart to [`neighborhood_hit`](super::neighborhood_hit),
/// which asks only about a point's immediate neighbours and so cannot tell a
/// cleanly separated class from one that merely fails to interleave. Comparing
/// each point against every class centroid instead makes `DSC` answer whether
/// the classes occupy separated regions of the visualisation as a whole. The
/// price is the reliance on centroids: a class that wraps around another, or is
/// otherwise strongly non-convex, can score poorly while still being perfectly
/// separable.
///
/// Ties count as a miss — the comparison is strict, as in the definition — and
/// a single-class input scores 1.0, the minimum over an empty set of rival
/// centroids being infinite.
#[must_use]
pub fn distance_consistency(pts_2d: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let k = unique_labels.len();
    if n == 0 {
        return 0.0;
    }
    if k < 2 {
        return 1.0;
    }

    let class_of = |i: usize| {
        unique_labels
            .binary_search(&labels[i])
            .expect("every label is in the deduplicated list")
    };

    let mut centroids = vec![(0.0f64, 0.0f64); k];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let c = class_of(i);
        centroids[c].0 += pts_2d[i * 2];
        centroids[c].1 += pts_2d[i * 2 + 1];
        counts[c] += 1;
    }
    for c in 0..k {
        if counts[c] > 0 {
            centroids[c].0 /= count_to_f64(counts[c]);
            centroids[c].1 /= count_to_f64(counts[c]);
        }
    }

    // Squared distances throughout: the comparison is monotone in the square
    // root, so `n * k` of them are wasted work.
    let dist_sq = |i: usize, c: usize| {
        let dx = pts_2d[i * 2] - centroids[c].0;
        let dy = pts_2d[i * 2 + 1] - centroids[c].1;
        dx * dx + dy * dy
    };

    let mut hits = 0usize;
    for i in 0..n {
        let own = class_of(i);
        let d_own = dist_sq(i, own);
        let nearest_rival = (0..k)
            .filter(|&c| c != own)
            .map(|c| dist_sq(i, c))
            .fold(f64::INFINITY, f64::min);
        if d_own < nearest_rival {
            hits += 1;
        }
    }

    count_to_f64(hits) / count_to_f64(n)
}

/// Distance consistency (Sips et al. 2009), on the 2-D projection.
///
/// Bounded in `[0, 1]` by construction, so it is a qParEGO objective — the
/// second one in [`Family::ClassSeparation`], beside `neighborhood_hit`, whose
/// local reading it complements.
///
/// It has **no manifold twin**: the definition is stated over positions in the
/// visualisation, and a centroid is an arithmetic mean, which is not the
/// intrinsic mean of points on a curved manifold. A pre-projection reading
/// would need a Fréchet mean, and would then be a different statistic rather
/// than the same one read earlier.
pub struct DistanceConsistency;
impl QualityMetric for DistanceConsistency {
    fn name(&self) -> &'static str {
        "distance_consistency"
    }
    fn base(&self) -> &'static str {
        "distance_consistency"
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
        "dsc"
    }
    fn label(&self) -> &'static str {
        "Distance Consistency"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        match c.labels {
            Some(l) => MetricValue::measured(distance_consistency(c.coords_2d(), l, c.n)),
            None => MetricValue::NotApplicable,
        }
    }
}
