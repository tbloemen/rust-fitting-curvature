use fitting_core::curvature_detection::{detect_geometry, GeometryVerdict};
use fitting_core::embedding::EmbeddingState;
use fitting_core::manifolds::create_manifold;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::{
    class_density_measure, cluster_density_measure, continuity, davies_bouldin_ratio, dunn_index,
    knn_overlap, neighborhood_hit, normalized_stress, shepard_goodness, trustworthiness,
};
use fitting_core::visualisation::{project_to_2d, SphericalProjection};
use indicatif::ProgressBar;

use crate::data::Dataset;
use crate::metrics::AllMetrics;
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

    /// Class labels, as the label-aware metrics (`neighborhood_hit`, the
    /// density measures) need them. Used by `--mode wilson-mds`, which scores
    /// a reconstruction through [`metrics_from_embedding`] instead of running
    /// an embedding through [`Evaluator::compute_all_metrics`].
    pub fn labels(&self) -> &[u32] {
        &self.dataset.labels
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
    ) -> AllMetrics {
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

        metrics_from_embedding(
            &self.high_dim_dist,
            &self.dataset.labels,
            &state.points,
            n,
            state.ambient_dim,
            training_config.curvature,
        )
    }

    pub fn evaluate_with_metric(
        &self,
        config: &TrialConfig,
        curvature_sign: f64,
        metric: &str,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> f64 {
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

        let projected = project_to_2d(
            &state.points,
            n,
            state.ambient_dim,
            training_config.curvature,
            SphericalProjection::AzimuthalEquidistant,
        );

        let k = (30_f64.min(n as f64 * 0.1)).round() as usize;

        // Lazily compute distance matrices only when needed.
        let dist_2d = || compute_euclidean_distance_matrix(&projected.coords, n, 2);
        let manifold_dist = || state.embedded_distances();

        match metric {
            "trustworthiness" => trustworthiness(&self.high_dim_dist, &dist_2d(), n, k),
            "trustworthiness_manifold" => {
                trustworthiness(&self.high_dim_dist, &manifold_dist(), n, k)
            }
            "continuity" => continuity(&self.high_dim_dist, &dist_2d(), n, k),
            "continuity_manifold" => continuity(&self.high_dim_dist, &manifold_dist(), n, k),
            "knn_overlap" => knn_overlap(&self.high_dim_dist, &dist_2d(), n, k),
            "knn_overlap_manifold" => knn_overlap(&self.high_dim_dist, &manifold_dist(), n, k),
            "neighborhood_hit" => neighborhood_hit(&dist_2d(), &self.dataset.labels, n, k),
            "neighborhood_hit_manifold" => {
                neighborhood_hit(&manifold_dist(), &self.dataset.labels, n, k)
            }
            "normalized_stress" => normalized_stress(&self.high_dim_dist, &dist_2d(), n),
            "normalized_stress_manifold" => {
                normalized_stress(&self.high_dim_dist, &manifold_dist(), n)
            }
            "shepard_goodness" => shepard_goodness(&self.high_dim_dist, &dist_2d(), n),
            "shepard_goodness_manifold" => {
                shepard_goodness(&self.high_dim_dist, &manifold_dist(), n)
            }
            "dunn_index" => dunn_index(&dist_2d(), &self.dataset.labels, n),
            "davies_bouldin_ratio" => davies_bouldin_ratio(
                &self.high_dim_dist,
                &projected.coords,
                &self.dataset.labels,
                n,
            ),
            "class_density_measure" => {
                class_density_measure(&projected.coords, &self.dataset.labels, n)
            }
            "cluster_density_measure" => {
                cluster_density_measure(&projected.coords, &self.dataset.labels, n)
            }
            _ => panic!(
                "Unknown metric: {metric}. Options: trustworthiness[_manifold], \
                 continuity[_manifold], knn_overlap[_manifold], neighborhood_hit[_manifold], \
                 normalized_stress[_manifold], shepard_goodness[_manifold], \
                 davies_bouldin_ratio, dunn_index, class_density_measure, cluster_density_measure"
            ),
        }
    }
}

/// Score a configuration on every metric, given only its coordinates.
///
/// Split out of [`Evaluator::compute_all_metrics`], which is now this function
/// plus the t-SNE run that produces `points`. The split exists so that a
/// configuration which did *not* come from t-SNE — the Wilson reconstruction
/// of `--mode wilson-mds` — is measured by exactly this code rather than by a
/// parallel copy of it: the same neighbourhood size `k`, the same azimuthal
/// projection, the same metric implementations. Two embeddings scored by two
/// pieces of code produce numbers that only look comparable, and the whole
/// point of the Wilson-versus-front comparison is that they genuinely are.
///
/// `points` is row-major `n × ambient_dim` on the manifold of the given
/// `curvature`, matching `EmbeddingState::points` / `Reconstruction::points`.
/// Both distance matrices are derived here rather than passed in, because
/// `EmbeddingState` derives them the same way — from
/// `create_manifold(curvature)` — so deriving them once here keeps the two
/// callers from drifting.
pub fn metrics_from_embedding(
    high_dim_dist: &[f64],
    labels: &[u32],
    points: &[f64],
    n: usize,
    ambient_dim: usize,
    curvature: f64,
) -> AllMetrics {
    let manifold = create_manifold(curvature);

    let projected = project_to_2d(
        points,
        n,
        ambient_dim,
        curvature,
        SphericalProjection::AzimuthalEquidistant,
    );

    let k = (30_f64.min(n as f64 * 0.1)).round() as usize;

    // Before-projection distances: manifold geodesic.
    let manifold_dist = manifold.pairwise_distances(points, n, ambient_dim);
    // After-projection distances: Euclidean in 2D projected space.
    let dist_2d = compute_euclidean_distance_matrix(&projected.coords, n, 2);

    let origin_dist = manifold.distances_from_origin(points, n, ambient_dim);
    let r_max = origin_dist.iter().cloned().fold(0.0_f64, f64::max);
    let r_rms = {
        let sum_sq: f64 = origin_dist.iter().map(|d| d * d).sum();
        (sum_sq / origin_dist.len() as f64).sqrt()
    };

    AllMetrics {
        trustworthiness: trustworthiness(high_dim_dist, &dist_2d, n, k),
        trustworthiness_manifold: trustworthiness(high_dim_dist, &manifold_dist, n, k),
        continuity: continuity(high_dim_dist, &dist_2d, n, k),
        continuity_manifold: continuity(high_dim_dist, &manifold_dist, n, k),
        knn_overlap: knn_overlap(high_dim_dist, &dist_2d, n, k),
        knn_overlap_manifold: knn_overlap(high_dim_dist, &manifold_dist, n, k),
        neighborhood_hit: neighborhood_hit(&dist_2d, labels, n, k),
        neighborhood_hit_manifold: neighborhood_hit(&manifold_dist, labels, n, k),
        normalized_stress: normalized_stress(high_dim_dist, &dist_2d, n),
        normalized_stress_manifold: normalized_stress(high_dim_dist, &manifold_dist, n),
        shepard_goodness: shepard_goodness(high_dim_dist, &dist_2d, n),
        shepard_goodness_manifold: shepard_goodness(high_dim_dist, &manifold_dist, n),
        davies_bouldin_ratio: davies_bouldin_ratio(high_dim_dist, &projected.coords, labels, n),
        dunn_index: dunn_index(&dist_2d, labels, n),
        class_density_measure: class_density_measure(&projected.coords, labels, n),
        cluster_density_measure: cluster_density_measure(&projected.coords, labels, n),
        r_max,
        r_rms,
    }
}
