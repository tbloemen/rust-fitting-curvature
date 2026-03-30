use fitting_core::embedding::EmbeddingState;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::{
    class_density_measure, cluster_density_measure, continuity, davies_bouldin_ratio, dunn_index,
    geodesic_distortion_gu2019, geodesic_distortion_mse, knn_overlap, trustworthiness,
};
use fitting_core::visualisation::{SphericalProjection, project_to_2d};
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
        let high_dim_dist = compute_euclidean_distance_matrix(&dataset.x, n, dataset.n_features);
        Self {
            n_samples: n,
            dataset,
            high_dim_dist,
        }
    }

    pub fn n_points(&self) -> usize {
        self.n_samples
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

        let mut state =
            EmbeddingState::new(&self.dataset.x, self.dataset.n_features, &training_config);
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

        // Only compute what the requested metric actually needs.
        // Metrics marked (*) need embedded_dist (O(n²)); others only need projected.coords.
        match metric {
            "trustworthiness" => {
                // *
                let d = compute_euclidean_distance_matrix(&projected.coords, n, 2);
                trustworthiness(&self.high_dim_dist, &d, n, k)
            }
            "continuity" => {
                // *
                let d = compute_euclidean_distance_matrix(&projected.coords, n, 2);
                continuity(&self.high_dim_dist, &d, n, k)
            }
            "knn_overlap" => {
                // *
                let d = compute_euclidean_distance_matrix(&projected.coords, n, 2);
                knn_overlap(&self.high_dim_dist, &d, n, k)
            }
            "geodesic_distortion_gu2019" => {
                // *
                let d = compute_euclidean_distance_matrix(&projected.coords, n, 2);
                geodesic_distortion_gu2019(&self.high_dim_dist, &d, n)
            }
            "geodesic_distortion_mse" => {
                // *
                let d = compute_euclidean_distance_matrix(&projected.coords, n, 2);
                geodesic_distortion_mse(&self.high_dim_dist, &d, n)
            }
            "dunn_index" => {
                // *
                let d = compute_euclidean_distance_matrix(&projected.coords, n, 2);
                dunn_index(&d, &self.dataset.labels, n)
            }
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
                "Unknown metric: {metric}. Options: trustworthiness, continuity, knn_overlap, \
                 geodesic_distortion_gu2019, geodesic_distortion_mse, davies_bouldin_ratio, \
                 dunn_index, class_density_measure, cluster_density_measure"
            ),
        }
    }
}
