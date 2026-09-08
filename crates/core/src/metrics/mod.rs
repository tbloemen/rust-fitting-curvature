//! Embedding quality metrics.
//!
//! Everything the workspace needs to know about a metric lives in that
//! metric's `impl QualityMetric` block in [`quality`]: its wire name, which
//! distance matrix it reads, which direction is better, whether it may serve as
//! a qParEGO objective, its labels, and how to compute it. [`ALL`] lists them,
//! [`MetricValues`] holds one embedding's scores, and
//! [`EmbeddingContext`](crate::context::EmbeddingContext) carries the inputs
//! while deriving each distance matrix at most once.
//!
//! Metrics group into:
//! - **Local structure** — trustworthiness, continuity
//! - **Class separation** — neighborhood_hit; plus the unbounded ratios
//!   davies_bouldin_ratio, dunn_index, cluster_density_measure
//! - **Distance preservation** — normalized_stress, shepard_goodness
//!
//! The spread diagnostics κ is gauged against — `r_max`, `r_rms`, `r_gyration`
//! — are deliberately *not* here: see [`crate::spread`].
//!
//! Most have both a *projected* and a *manifold* reading. That distinction is
//! carried entirely by which distance matrix is passed in, which is why the
//! functions in [`functions`] take a matrix and never a manifold.

mod functions;
mod quality;
mod values;

pub use functions::*;
pub use quality::{
    Direction, Family, Metric, QualityMetric, Space, ALL, CLUSTER_DENSITY_MEASURE, CONTINUITY,
    CONTINUITY_MANIFOLD, DAVIES_BOULDIN_RATIO, DUNN_INDEX, NEIGHBORHOOD_HIT,
    NEIGHBORHOOD_HIT_MANIFOLD, NORMALIZED_STRESS, NORMALIZED_STRESS_MANIFOLD, OBJECTIVES,
    SHEPARD_GOODNESS, SHEPARD_GOODNESS_MANIFOLD, TRUSTWORTHINESS, TRUSTWORTHINESS_MANIFOLD,
};
pub use values::MetricValues;
