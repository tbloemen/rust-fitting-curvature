//! Shepard goodness (Espadoto et al.), and its two registry entries.

use super::quality::{Direction, Family, QualityMetric, Space};
use super::values::MetricValue;
use crate::cast::count_to_f64;
use crate::context::EmbeddingContext;

/// The rank variable `R[X]` of `values`, using **fractional ranks**: identical
/// values "are each assigned fractional ranks equal to the average of their
/// positions" (Spearman's rank correlation coefficient, *Definition and
/// calculation*). 0-based here, so rank 0 is the smallest value.
///
/// Ties are not exotic in this codebase, which is why the fractional
/// convention matters rather than being a formality.
fn fractional_rank_vector(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < indices.len() {
        let mut end = start;
        while end + 1 < indices.len()
            && values[indices[end + 1]].partial_cmp(&values[indices[start]])
                == Some(std::cmp::Ordering::Equal)
        {
            end += 1;
        }
        let fractional_rank = count_to_f64(start + end) / 2.0;
        for &idx in &indices[start..=end] {
            ranks[idx] = fractional_rank;
        }
        start = end + 1;
    }
    ranks
}

/// Shepard goodness `M_shep` (Espadoto et al.): Spearman's rank correlation
/// coefficient `r_s` between the pairwise distances of the original space and
/// those of the embedding.
///
/// The observations `(X_i, Y_i)` are the `m = n(n−1)/2` upper-triangle point
/// pairs — note that the statistical sample size is `m`, not this function's
/// `n`, which counts *points*, not observations. `r_s` is then a scalar measure
/// of how well the global rank-order of distances is preserved.
///
/// Computed from the general definition, as the Pearson correlation
/// coefficient of the two rank variables:
///
/// `r_s = ρ(R[X], R[Y]) = cov(R[X], R[Y]) / (σ_R[X] · σ_R[Y])`
///
/// and deliberately **not** via the familiar shortcut
///
/// `r_s = 1 − 6·Σd_i² / (m(m²−1))`,  `d_i = R[X_i] − R[Y_i]`
///
/// which "applies only when all n ranks are distinct integers (no ties)".
/// These distance vectors are routinely full of ties (see
/// [`fractional_rank_vector`]), so the standard guidance holds: with ties
/// present that formula "should not be used" and "the Pearson correlation
/// coefficient should be calculated on the ranks" instead. Using it anyway
/// with ordinal ranks scored a fully collapsed embedding — every pairwise
/// distance identical, so no rank information at all — at 0.63 on
/// `tree_structured` and 0.51 on `hyperbolic_shells`, purely from tie-breaking
/// by point index.
///
/// Two project-specific deviations from textbook `r_s ∈ [−1, 1]`:
///
/// - The result is **normalised onto [0, 1]** by the order-preserving affine
///   map `(r_s + 1) / 2`, so that every reported metric shares one range.
///   1 is perfect rank-order preservation, 0.5 is rank-order independence
///   (the no-skill value of a rank correlation), and 0 would be exact rank
///   reversal. The map is strictly monotone, so no information in `r_s` is
///   lost and the induced ordering of embeddings is unchanged; note that it
///   moves the no-skill point off zero, so **0.5, not 0, is the score of an
///   embedding that preserves nothing**.
/// - Degenerate input — either side constant, so `σ_R = 0` and `r_s` is
///   undefined (`scipy.stats.spearmanr` returns NaN here) — returns 0.5, the
///   image of "no rank information", rather than NaN, because this feeds a
///   Pareto objective where NaN is a hazard.
#[must_use]
pub fn shepard_goodness(high_dim_distances: &[f64], embedded_distances: &[f64], n: usize) -> f64 {
    let m = n * (n - 1) / 2;
    if m < 2 {
        return 1.0;
    }

    let mut d_high = Vec::with_capacity(m);
    let mut d_embed = Vec::with_capacity(m);
    for i in 0..n {
        for j in (i + 1)..n {
            d_high.push(high_dim_distances[i * n + j]);
            d_embed.push(embedded_distances[i * n + j]);
        }
    }

    let r_x = fractional_rank_vector(&d_high);
    let r_y = fractional_rank_vector(&d_embed);

    // cov(R[X], R[Y]) / (σ_R[X] · σ_R[Y]), with the 1/m factors cancelling.
    // Both mean ranks are (m-1)/2 whatever the tie pattern, since fractional
    // ranks redistribute 0..m-1 without changing their sum; the σ do change
    // — ties shrink them — which is exactly what the shortcut cannot see.
    let mean_rank = count_to_f64(m - 1) / 2.0;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&rx, &ry) in r_x.iter().zip(r_y.iter()) {
        let (dx, dy) = (rx - mean_rank, ry - mean_rank);
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    // A constant distance vector collapses every rank onto `mean_rank`, so
    // σ_R = 0 and r_s is undefined. That is a total loss of rank structure,
    // which on this scale is the no-skill value 0.5 — not 1, and not the 0
    // that exact rank *reversal* would earn.
    let sigma_product = (var_x * var_y).sqrt();
    if sigma_product < 1e-12 {
        return 0.5;
    }

    // (r_s + 1) / 2, clamped only against floating-point overshoot at the ends.
    ((cov / sigma_product + 1.0) / 2.0).clamp(0.0, 1.0)
}

/// Shepard goodness on the 2-D projection.
pub struct ShepardGoodness;
impl QualityMetric for ShepardGoodness {
    fn name(&self) -> &'static str {
        "shepard_goodness"
    }
    fn base(&self) -> &'static str {
        "shepard_goodness"
    }
    fn space(&self) -> Space {
        Space::Projected
    }
    fn family(&self) -> Family {
        Family::Distance
    }
    fn direction(&self) -> Direction {
        Direction::Maximize
    }
    fn is_objective(&self) -> bool {
        true
    }
    fn short(&self) -> &'static str {
        "shep"
    }
    fn label(&self) -> &'static str {
        "Shepard Goodness"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        MetricValue::measured(shepard_goodness(c.high_dim_dist, c.dist_2d(), c.n))
    }
}

/// Shepard goodness on the manifold geodesics. See
/// [`TrustworthinessManifold`](super::trustworthiness::TrustworthinessManifold).
pub struct ShepardGoodnessManifold;
impl QualityMetric for ShepardGoodnessManifold {
    fn name(&self) -> &'static str {
        "shepard_goodness_manifold"
    }
    fn base(&self) -> &'static str {
        "shepard_goodness"
    }
    fn space(&self) -> Space {
        Space::Manifold
    }
    fn family(&self) -> Family {
        Family::Distance
    }
    fn direction(&self) -> Direction {
        Direction::Maximize
    }
    fn is_objective(&self) -> bool {
        false
    }
    fn short(&self) -> &'static str {
        "shep_m"
    }
    fn label(&self) -> &'static str {
        "Shepard Goodness"
    }
    fn compute(&self, c: &EmbeddingContext<'_>) -> MetricValue {
        MetricValue::measured(shepard_goodness(c.high_dim_dist, c.manifold_dist(), c.n))
    }
}
