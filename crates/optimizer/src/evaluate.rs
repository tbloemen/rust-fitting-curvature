use fitting_core::context::EmbeddingContext;
use fitting_core::embedding::EmbeddingState;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::scoring_k;
use fitting_core::metrics::MetricValues;
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
    /// `k` is the workspace-wide [`scoring_k`], shared with the viewer;
    /// `AzimuthalEquidistant` is this crate's fixed projection where the viewer
    /// uses whatever the user picked. Every caller goes through this one
    /// function so they cannot drift apart, which is what the seam
    /// `metrics_from_embedding` documents was always for.
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
}
