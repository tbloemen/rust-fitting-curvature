//! One embedding, with every matrix derivable from it computed at most once.
//!
//! Read by both [`crate::metrics`] and [`crate::spread`], which is why it sits
//! above them rather than inside either.
//!
//! Before this existed, three callers derived the same matrices three ways:
//! `evaluate::metrics_from_embedding` went through
//! `create_manifold(curvature).pairwise_distances(..)`,
//! `evaluate::evaluate_with_metric` through `EmbeddingState::embedded_distances`,
//! and `metrics::compute_snapshot` took them pre-computed from its caller. The
//! doc comment on `metrics_from_embedding` said keeping those from drifting was
//! the point of that seam; a single context makes them the same by
//! construction, which is stronger than a comment.

use std::cell::OnceCell;

use crate::manifolds::create_manifold;
use crate::matrices::compute_euclidean_distance_matrix;
use crate::metrics::Space;
use crate::visualisation::{project_to_2d, SphericalProjection};

/// Everything a metric or diagnostic may read about one embedding.
///
/// `k` and `projection` are **inputs, not constants**, because the two callers
/// genuinely differ and that difference is load-bearing: the optimizer scores
/// with `k = min(30, 0.1n)` under `AzimuthalEquidistant`, while the interactive
/// `EmbeddingState` uses `k = perplexity` under the projection the user picked.
/// Folding either into the registry would silently move published numbers.
pub struct EmbeddingContext<'a> {
    /// Pairwise distances in the input space, flat row-major `n × n`.
    pub high_dim_dist: &'a [f64],
    /// Row-major `n × ambient_dim`, on the manifold of `curvature`.
    pub points: &'a [f64],
    /// Class labels, or `None` for unlabelled data — every label-aware metric
    /// then reports `f64::NAN`.
    pub labels: Option<&'a [u32]>,
    pub n: usize,
    pub ambient_dim: usize,
    pub curvature: f64,
    /// Neighbourhood size for the k-NN metrics.
    pub k: usize,
    pub projection: SphericalProjection,

    manifold_dist: OnceCell<Vec<f64>>,
    coords_2d: OnceCell<Vec<f64>>,
    dist_2d: OnceCell<Vec<f64>>,
    origin_dist: OnceCell<Vec<f64>>,
}

/// Whether every entry of a derived matrix is finite.
///
/// One `O(n²)` pass against metrics that are already `O(n² log n)` for their
/// rank sorts, and it only runs for a matrix something actually reads.
fn all_finite(xs: &[f64]) -> bool {
    xs.iter().all(|x| x.is_finite())
}

impl<'a> EmbeddingContext<'a> {
    /// Eight arguments is what scoring an embedding genuinely takes, and a
    /// builder would make `k` and `projection` skippable — the two inputs that
    /// silently move published numbers when a caller forgets them.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        high_dim_dist: &'a [f64],
        points: &'a [f64],
        labels: Option<&'a [u32]>,
        n: usize,
        ambient_dim: usize,
        curvature: f64,
        k: usize,
        projection: SphericalProjection,
    ) -> Self {
        Self {
            high_dim_dist,
            points,
            labels,
            n,
            ambient_dim,
            curvature,
            k,
            projection,
            manifold_dist: OnceCell::new(),
            coords_2d: OnceCell::new(),
            dist_2d: OnceCell::new(),
            origin_dist: OnceCell::new(),
        }
    }

    /// Seed the manifold geodesics from a caller that already has them, rather
    /// than paying for a second `pairwise_distances` over the same points.
    ///
    /// # Panics
    ///
    /// Panics if they have already been derived — silently ignoring the
    /// argument would be the drift this type exists to prevent.
    #[must_use]
    pub fn with_manifold_dist(self, dist: Vec<f64>) -> Self {
        self.manifold_dist
            .set(dist)
            .expect("manifold distances already derived");
        self
    }

    /// Geodesic distances on the manifold: the *before projection* reading.
    pub fn manifold_dist(&self) -> &[f64] {
        self.manifold_dist.get_or_init(|| {
            create_manifold(self.curvature).pairwise_distances(
                self.points,
                self.n,
                self.ambient_dim,
            )
        })
    }

    /// Flat `(x, y)` pairs as the viewer sees them.
    pub fn coords_2d(&self) -> &[f64] {
        self.coords_2d.get_or_init(|| {
            project_to_2d(
                self.points,
                self.n,
                self.ambient_dim,
                self.curvature,
                self.projection,
            )
            .coords
        })
    }

    /// Euclidean distances in the projected plane: the *after projection*
    /// reading.
    pub fn dist_2d(&self) -> &[f64] {
        self.dist_2d
            .get_or_init(|| compute_euclidean_distance_matrix(self.coords_2d(), self.n, 2))
    }

    /// Whether the distances a metric of this `space` reads are all finite.
    ///
    /// A diverged embedding produces `inf`/`NaN` distances, and only the
    /// metrics that *sum* them notice: the ones that compare fall into their
    /// degenerate branches, and the ones that rank sort NaN to a defined
    /// position and return a confident, meaningless score. `MetricValues::compute`
    /// asks this first so such a reading is recorded as
    /// [`crate::metrics::MetricValue::Diverged`] rather than as a number.
    ///
    /// Per space, because an embedding can be sound on the manifold and blow up
    /// only through the projection; failing both would discard a real
    /// measurement. `high_dim_dist` is deliberately not covered — non-finite
    /// *input* distances are a data problem, and reporting every metric as
    /// diverged would hide it.
    pub fn distances_are_finite(&self, space: Space) -> bool {
        match space {
            Space::Projected => all_finite(self.dist_2d()),
            Space::Manifold => all_finite(self.manifold_dist()),
        }
    }

    /// Whether the configuration's own extent is measurable — the gate
    /// [`crate::spread::SpreadDiagnostics`] uses, over the two matrices it
    /// reads.
    pub fn spread_is_finite(&self) -> bool {
        all_finite(self.origin_dist()) && all_finite(self.manifold_dist())
    }

    /// Geodesic distance from the manifold origin, per point. Only
    /// [`crate::spread::SpreadDiagnostics`] reads this.
    pub fn origin_dist(&self) -> &[f64] {
        self.origin_dist.get_or_init(|| {
            create_manifold(self.curvature).distances_from_origin(
                self.points,
                self.n,
                self.ambient_dim,
            )
        })
    }
}
