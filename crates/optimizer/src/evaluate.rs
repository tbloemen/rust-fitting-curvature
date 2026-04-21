use fitting_core::curvature_detection::{GeometryDetection, detect_geometry};
use fitting_core::embedding::EmbeddingState;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::{
    class_density_measure, cluster_density_measure, continuity, davies_bouldin_ratio, dunn_index,
    knn_overlap, neighborhood_hit, normalized_stress, shepard_goodness, trustworthiness,
};
use fitting_core::visualisation::{SphericalProjection, project_to_2d};
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

    /// Detect the best-fitting geometry for this dataset using shell density profiles.
    pub fn infer_geometry(&self) -> GeometryDetection {
        detect_geometry(&self.high_dim_dist, self.n_samples, 40, 0)
    }

    pub fn compute_all_metrics(
        &self,
        config: &TrialConfig,
        curvature: f64,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> AllMetrics {
        let n = self.n_samples;
        let training_config = config.to_training_config(n, curvature, seed);

        pb_iters.reset();
        pb_iters.set_length(training_config.n_iterations as u64);

        let mut state = if self.dataset.precomputed_distances.is_empty() {
            EmbeddingState::new(&self.dataset.x, self.dataset.n_features, &training_config)
        } else {
            EmbeddingState::from_distances(&self.dataset.precomputed_distances, n, &training_config)
        };
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

        // Before-projection distances: manifold geodesic.
        let manifold_dist = state.embedded_distances();
        // After-projection distances: Euclidean in 2D projected space.
        let dist_2d = compute_euclidean_distance_matrix(&projected.coords, n, 2);

        AllMetrics {
            trustworthiness: trustworthiness(&self.high_dim_dist, &dist_2d, n, k),
            trustworthiness_manifold: trustworthiness(&self.high_dim_dist, &manifold_dist, n, k),
            continuity: continuity(&self.high_dim_dist, &dist_2d, n, k),
            continuity_manifold: continuity(&self.high_dim_dist, &manifold_dist, n, k),
            knn_overlap: knn_overlap(&self.high_dim_dist, &dist_2d, n, k),
            knn_overlap_manifold: knn_overlap(&self.high_dim_dist, &manifold_dist, n, k),
            neighborhood_hit: neighborhood_hit(&dist_2d, &self.dataset.labels, n, k),
            neighborhood_hit_manifold: neighborhood_hit(&manifold_dist, &self.dataset.labels, n, k),
            normalized_stress: normalized_stress(&self.high_dim_dist, &dist_2d, n),
            normalized_stress_manifold: normalized_stress(&self.high_dim_dist, &manifold_dist, n),
            shepard_goodness: shepard_goodness(&self.high_dim_dist, &dist_2d, n),
            shepard_goodness_manifold: shepard_goodness(&self.high_dim_dist, &manifold_dist, n),
            davies_bouldin_ratio: davies_bouldin_ratio(
                &self.high_dim_dist,
                &projected.coords,
                &self.dataset.labels,
                n,
            ),
            dunn_index: dunn_index(&dist_2d, &self.dataset.labels, n),
            class_density_measure: class_density_measure(
                &projected.coords,
                &self.dataset.labels,
                n,
            ),
            cluster_density_measure: cluster_density_measure(
                &projected.coords,
                &self.dataset.labels,
                n,
            ),
        }
    }

    pub fn evaluate_with_metric(
        &self,
        config: &TrialConfig,
        curvature: f64,
        metric: &str,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> f64 {
        let n = self.n_samples;
        let training_config = config.to_training_config(n, curvature, seed);

        pb_iters.reset();
        pb_iters.set_length(training_config.n_iterations as u64);

        let mut state = if self.dataset.precomputed_distances.is_empty() {
            EmbeddingState::new(&self.dataset.x, self.dataset.n_features, &training_config)
        } else {
            EmbeddingState::from_distances(&self.dataset.precomputed_distances, n, &training_config)
        };
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
