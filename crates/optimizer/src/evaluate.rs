use fitting_core::curvature_detection::{GeometryDetection, detect_geometry};
use fitting_core::embedding::EmbeddingState;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::{
    class_density_measure, cluster_density_measure, continuity, davies_bouldin_ratio, dunn_index,
    geodesic_distortion_gu2019, geodesic_distortion_mse, knn_overlap, trustworthiness,
};
use fitting_core::visualisation::{SphericalProjection, project_to_2d};
use indicatif::ProgressBar;
use serde::Serialize;

use crate::data::Dataset;
use crate::search_space::TrialConfig;

#[derive(Debug, Clone, Serialize)]
pub struct AllMetrics {
    pub trustworthiness: f64,
    pub continuity: f64,
    pub knn_overlap: f64,
    pub geodesic_distortion_gu2019: f64,
    pub geodesic_distortion_mse: f64,
    pub davies_bouldin_ratio: f64,
    pub dunn_index: f64,
    pub class_density_measure: f64,
    pub cluster_density_measure: f64,
}

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

        // Compute all metrics once using precomputed data.
        let embedded_dist = compute_euclidean_distance_matrix(&projected.coords, n, 2);

        AllMetrics {
            trustworthiness: trustworthiness(&self.high_dim_dist, &embedded_dist, n, k),
            continuity: continuity(&self.high_dim_dist, &embedded_dist, n, k),
            knn_overlap: knn_overlap(&self.high_dim_dist, &embedded_dist, n, k),
            geodesic_distortion_gu2019: geodesic_distortion_gu2019(
                &self.high_dim_dist,
                &embedded_dist,
                n,
            ),
            geodesic_distortion_mse: geodesic_distortion_mse(
                &self.high_dim_dist,
                &embedded_dist,
                n,
            ),
            davies_bouldin_ratio: davies_bouldin_ratio(
                &self.high_dim_dist,
                &projected.coords,
                &self.dataset.labels,
                n,
            ),
            dunn_index: dunn_index(&embedded_dist, &self.dataset.labels, n),
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
