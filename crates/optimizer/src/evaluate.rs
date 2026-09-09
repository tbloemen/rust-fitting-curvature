use fitting_core::cast::{count_to_f64, to_usize};
use fitting_core::context::EmbeddingContext;
use fitting_core::curvature_detection::{detect_geometry, GeometryVerdict};
use fitting_core::embedding::EmbeddingState;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::{Metric, MetricValue, MetricValues};
use fitting_core::spread::SpreadDiagnostics;
use fitting_core::visualisation::SphericalProjection;
use indicatif::ProgressBar;

use crate::data::Dataset;
use crate::search_space::TrialConfig;

pub struct Evaluator {
    dataset: Dataset,
    high_dim_dist: Vec<f64>,
    n_samples: usize,
}

impl Evaluator {
    pub fn new(dataset: Dataset) -> Self {
        let n = dataset.n_points;
        let high_dim_dist = if dataset.precomputed_distances.is_empty() {
            compute_euclidean_distance_matrix(&dataset.x, n, dataset.n_features)
        } else {
            dataset.precomputed_distances.clone()
        };
        Self {
            n_samples: n,
            dataset,
            high_dim_dist,
        }
    }

    pub fn n_points(&self) -> usize {
        self.n_samples
    }

    /// The precomputed high-dimensional pairwise distance matrix (flat,
    /// row-major, `n_points × n_points`). Used by `--mode detect` to run the
    /// curvature-detection fits directly on the data distances.
    pub fn distances(&self) -> &[f64] {
        &self.high_dim_dist
    }

    /// Detect the best-fitting geometry for this dataset.  Returns only
    /// the [`GeometryVerdict`] (geometry label + curvature) — the caller
    /// acts on the decision, not the detector's diagnostic internals.
    pub fn infer_geometry(&self) -> GeometryVerdict {
        // Fit the curvature models at the embedding target dimension (2-D).
        detect_geometry(&self.high_dim_dist, self.n_samples, 2)
    }

    pub fn compute_all_metrics(
        &self,
        config: &TrialConfig,
        curvature_sign: f64,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> (MetricValues, SpreadDiagnostics) {
        let (state, curvature) = self.run_embedding(config, curvature_sign, seed, pb_iters);
        let ctx = self.context(&state, curvature);
        (
            MetricValues::compute(&ctx),
            SpreadDiagnostics::compute(&ctx),
        )
    }

    /// Score one configuration on a single named metric, for `--mode bayes`
    /// and `--mode scan`.
    ///
    /// This used to be a thirteen-arm `match` on the metric name, computing the
    /// same things `metrics_from_embedding` did a few lines below, behind its
    /// own pair of lazy closures and closing on a `panic!` whose message listed
    /// the valid names a fourth time. All of that is the registry's job now,
    /// and `EmbeddingContext` keeps the compute-only-what-is-asked-for property
    /// the closures were there for.
    pub fn evaluate_with_metric(
        &self,
        config: &TrialConfig,
        curvature_sign: f64,
        metric: &str,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> MetricValue {
        let metric = Metric::by_name(metric).unwrap_or_else(|| {
            panic!(
                "Unknown metric: {metric}. Options: {}",
                Metric::valid_names()
            )
        });
        let (state, curvature) = self.run_embedding(config, curvature_sign, seed, pb_iters);
        metric.compute(&self.context(&state, curvature))
    }

    /// Fit one embedding under `config`, driving the iteration progress bar.
    /// Returns the fitted state and the curvature it was fitted at.
    fn run_embedding(
        &self,
        config: &TrialConfig,
        curvature_sign: f64,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> (EmbeddingState, f64) {
        let n = self.n_samples;
        let training_config = config.to_training_config(n, curvature_sign, seed);

        pb_iters.reset();
        pb_iters.set_length(training_config.n_iterations as u64);

        let mut state = if self.dataset.precomputed_distances.is_empty() {
            EmbeddingState::new(&self.dataset.x, self.dataset.n_features, &training_config)
        } else {
            EmbeddingState::from_distances(&self.dataset.precomputed_distances, n, &training_config)
        }
        .with_loss_tracking(false);
        while !state.is_done() {
            state.step();
            pb_iters.inc(1);
        }
        (state, training_config.curvature)
    }

    /// The scoring context for an arbitrary configuration on the manifold of
    /// `curvature`.
    ///
    /// `k = min(30, 0.1n)` and `AzimuthalEquidistant` are this crate's scoring
    /// convention, and differ from the interactive viewer's. Every caller goes
    /// through this one function so they cannot drift apart, which is what the
    /// seam `metrics_from_embedding` documents was always for.
    fn context_for<'a>(
        &'a self,
        points: &'a [f64],
        ambient_dim: usize,
        curvature: f64,
    ) -> EmbeddingContext<'a> {
        EmbeddingContext::new(
            &self.high_dim_dist,
            points,
            Some(&self.dataset.labels),
            self.n_samples,
            ambient_dim,
            curvature,
            scoring_k(self.n_samples),
            SphericalProjection::AzimuthalEquidistant,
        )
    }

    /// The scoring context for a fitted state — [`Self::context_for`] plus the
    /// manifold distances the state already computed.
    fn context<'a>(&'a self, state: &'a EmbeddingState, curvature: f64) -> EmbeddingContext<'a> {
        self.context_for(&state.points, state.ambient_dim, curvature)
            .with_manifold_dist(state.embedded_distances())
    }

    /// Score a configuration this evaluator did not fit — for `--mode
    /// reference`, which scores a dataset's own ground-truth coordinates.
    ///
    /// `points` must be flat `n × ambient_dim` and lie on the manifold of
    /// `curvature`. Unlike [`Self::context`] this does not pre-set the manifold
    /// distances: the context then derives them with
    /// `create_manifold(curvature).pairwise_distances`, which is exactly what
    /// `EmbeddingState::embedded_distances` computes, so the two paths cannot
    /// drift. (`with_manifold_dist` also panics if set twice.)
    pub fn score_points(
        &self,
        points: &[f64],
        ambient_dim: usize,
        curvature: f64,
    ) -> (MetricValues, SpreadDiagnostics) {
        let ctx = self.context_for(points, ambient_dim, curvature);
        (
            MetricValues::compute(&ctx),
            SpreadDiagnostics::compute(&ctx),
        )
    }

    /// The dataset's own coordinates and their ambient dimension. Empty for
    /// graph datasets, which have no feature representation.
    pub fn source_points(&self) -> (&[f64], usize) {
        (&self.dataset.x, self.dataset.n_features)
    }
}

/// The neighbourhood size every metric in this crate is scored at.
pub fn scoring_k(n: usize) -> usize {
    to_usize((30_f64.min(count_to_f64(n) * 0.1)).round())
}
