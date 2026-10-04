//! Embedding quality metrics.
//!
//! One metric per module: the function that computes it sits next to the
//! `impl QualityMetric` block that registers it, so everything the workspace
//! needs to know about a metric — its wire name, which distance matrix it
//! reads, which direction is better, whether it may serve as a qParEGO
//! objective, its labels, and how to compute it — is in one file. [`ALL`]
//! lists them, [`MetricValues`] holds one embedding's scores, and
//! [`EmbeddingContext`](crate::context::EmbeddingContext) carries the inputs
//! while deriving each distance matrix at most once.
//!
//! Metrics group into:
//! - **Local structure** — [`trustworthiness`], [`continuity`]
//! - **Class separation** — [`neighborhood_hit`] locally and
//!   [`distance_consistency`] globally; plus the unbounded ratios
//!   [`davies_bouldin_ratio`], [`dunn_index`], [`cluster_density_measure`]
//! - **Distance preservation** — [`normalized_stress`], [`shepard_goodness`]
//!
//! The spread diagnostics κ is gauged against — `r_max`, `r_rms`, `r_gyration`
//! — are deliberately *not* here: see [`crate::spread`].
//!
//! Most metrics have both a *projected* and a *manifold* reading. That
//! distinction is carried entirely by which distance matrix is passed in,
//! which is why the functions take a matrix and never a manifold — and why the
//! two registry entries of a twin pair share one module and one function.

mod cluster_density_measure;
mod continuity;
mod davies_bouldin_ratio;
mod distance_consistency;
mod dunn_index;
mod helpers;
mod neighborhood_hit;
mod normalized_stress;
mod quality;
mod shepard_goodness;
mod trustworthiness;
mod values;

pub use cluster_density_measure::cluster_density_measure;
pub use continuity::continuity;
pub use davies_bouldin_ratio::{davies_bouldin, davies_bouldin_ratio, davies_bouldin_ratio_from};
pub use distance_consistency::distance_consistency;
pub use dunn_index::dunn_index;
pub use helpers::euclidean_dist_2d;
pub use neighborhood_hit::neighborhood_hit;
pub use normalized_stress::normalized_stress;
pub use shepard_goodness::shepard_goodness;
pub use trustworthiness::trustworthiness;

pub use quality::{
    Direction, Family, Metric, QualityMetric, Space, ALL, CLUSTER_DENSITY_MEASURE, CONTINUITY,
    CONTINUITY_MANIFOLD, DAVIES_BOULDIN_RATIO, DISTANCE_CONSISTENCY, DUNN_INDEX, NEIGHBORHOOD_HIT,
    NEIGHBORHOOD_HIT_MANIFOLD, NORMALIZED_STRESS, NORMALIZED_STRESS_MANIFOLD, OBJECTIVES,
    SHEPARD_GOODNESS, SHEPARD_GOODNESS_MANIFOLD, TRUSTWORTHINESS, TRUSTWORTHINESS_MANIFOLD,
};
pub(crate) use values::mean_of as values_mean_of;
pub use values::{MetricValue, MetricValues};

use crate::cast::{count_to_f64, to_usize};

/// The neighbourhood size every metric is scored at: `k = min(30, 0.1n)`.
///
/// This is the one scoring convention shared by the optimizer sweeps, the
/// `results/` Pareto fronts and the interactive viewer, so a number read off
/// the viewer's panel is comparable with a number read out of a front. It is
/// deliberately *not* the perplexity: perplexity is a fitting knob, and tying
/// the score to it would let two configurations be scored at different `k`.
#[must_use]
pub fn scoring_k(n: usize) -> usize {
    to_usize((30_f64.min(count_to_f64(n) * 0.1)).round())
}
