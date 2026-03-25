use fitting_core::embedding::EmbeddingState;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::{
    class_density_measure, cluster_density_measure, continuity, davies_bouldin_ratio, dunn_index,
    geodesic_distortion_gu2019, geodesic_distortion_mse, knn_overlap, trustworthiness,
};
use fitting_core::visualisation::{project_to_2d, SphericalProjection};
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
        Self { n_samples: n, dataset, high_dim_dist }
    }

    pub fn evaluate(
        &self,
        config: &TrialConfig,
        curvature: f64,
        seed: u64,
        pb_iters: &ProgressBar,
    ) -> EvaluationResult {
        let n = self.n_samples;
        let n_features = self.dataset.n_features;
        let training_config = config.to_training_config(n, curvature, seed);

        pb_iters.reset();
        pb_iters.set_length(training_config.n_iterations as u64);

        let mut state = EmbeddingState::new(&self.dataset.x, n_features, &training_config);
        while !state.is_done() {
            state.step();
            pb_iters.inc(1);
        }

        // Project to 2D for evaluation, matching what the visualisation shows.
        // Spherical uses azimuthal equidistant; hyperbolic uses Poincaré disk;
        // Euclidean uses the first two coordinates directly.
        let projected = project_to_2d(
            &state.points,
            n,
            state.ambient_dim,
            training_config.curvature,
            SphericalProjection::AzimuthalEquidistant,
        );
        let embedded_dist = compute_euclidean_distance_matrix(&projected.coords, n, 2);

        let k = (30_f64.min(n as f64 * 0.1)).round() as usize;

        let trust = trustworthiness(&self.high_dim_dist, &embedded_dist, n, k);
        let cont = continuity(&self.high_dim_dist, &embedded_dist, n, k);
        let knn = knn_overlap(&self.high_dim_dist, &embedded_dist, n, k);
        let geo_gu = geodesic_distortion_gu2019(&self.high_dim_dist, &embedded_dist, n);
        let geo_mse = geodesic_distortion_mse(&self.high_dim_dist, &embedded_dist, n);
        let db_ratio = davies_bouldin_ratio(&self.high_dim_dist, &projected.coords, &self.dataset.labels, n);
        let dunn = dunn_index(&embedded_dist, &self.dataset.labels, n);
        let cdm = class_density_measure(&projected.coords, &self.dataset.labels, n);
        let cldm = cluster_density_measure(&projected.coords, &self.dataset.labels, n);

        EvaluationResult {
            trustworthiness: trust,
            continuity: cont,
            knn_overlap: knn,
            geodesic_distortion_gu2019: geo_gu,
            geodesic_distortion_mse: geo_mse,
            davies_bouldin_ratio: db_ratio,
            dunn_index: dunn,
            class_density_measure: cdm,
            cluster_density_measure: cldm,
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
        self.evaluate(config, curvature, seed, pb_iters).get(metric)
    }
}

#[derive(Debug, Clone)]
pub struct EvaluationResult {
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

impl EvaluationResult {
    pub fn get(&self, metric: &str) -> f64 {
        match metric {
            "trustworthiness" => self.trustworthiness,
            "continuity" => self.continuity,
            "knn_overlap" => self.knn_overlap,
            "geodesic_distortion_gu2019" => self.geodesic_distortion_gu2019,
            "geodesic_distortion_mse" => self.geodesic_distortion_mse,
            "davies_bouldin_ratio" => self.davies_bouldin_ratio,
            "dunn_index" => self.dunn_index,
            "class_density_measure" => self.class_density_measure,
            "cluster_density_measure" => self.cluster_density_measure,
            _ => panic!(
                "Unknown metric: {}. Options: trustworthiness, continuity, knn_overlap, geodesic_distortion_gu2019, geodesic_distortion_mse, davies_bouldin_ratio, dunn_index, class_density_measure, cluster_density_measure",
                metric
            ),
        }
    }
}
