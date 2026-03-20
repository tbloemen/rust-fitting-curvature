use fitting_core::embedding::EmbeddingState;
use fitting_core::matrices::compute_euclidean_distance_matrix;
use fitting_core::metrics::{
    class_density_measure, cluster_density_measure, continuity, davies_bouldin_ratio, dunn_index,
    geodesic_distortion_gu2019, geodesic_distortion_mse, knn_overlap, trustworthiness,
};

use crate::data::Dataset;
use crate::search_space::TrialConfig;

pub struct Evaluator {
    dataset: Dataset,
    n_samples: usize,
}

impl Evaluator {
    pub fn new(dataset: Dataset) -> Self {
        let n_samples = dataset.n_points;
        Self { dataset, n_samples }
    }

    pub fn evaluate(&self, config: &TrialConfig, seed: u64) -> EvaluationResult {
        let n = self.n_samples;
        let n_features = self.dataset.n_features;

        let mut state = EmbeddingState::new(
            &self.dataset.x,
            n_features,
            &config.to_training_config(n, seed),
        );
        while !state.is_done() {
            state.step();
        }

        let high_dim_dist = compute_euclidean_distance_matrix(&self.dataset.x, n, n_features);
        let embedded_dist = state.embedded_distances();

        let k = (30_f64.min(n as f64 * 0.1)).round() as usize;

        let trust = trustworthiness(&high_dim_dist, &embedded_dist, n, k);
        let cont = continuity(&high_dim_dist, &embedded_dist, n, k);
        let knn = knn_overlap(&high_dim_dist, &embedded_dist, n, k);
        let geo_gu = geodesic_distortion_gu2019(&high_dim_dist, &embedded_dist, n);
        let geo_mse = geodesic_distortion_mse(&high_dim_dist, &embedded_dist, n);
        let db_ratio = davies_bouldin_ratio(&high_dim_dist, &state.points, &self.dataset.labels, n);
        let dunn = dunn_index(&embedded_dist, &self.dataset.labels, n);
        let cdm = class_density_measure(&state.points, &self.dataset.labels, n);
        let cldm = cluster_density_measure(&state.points, &self.dataset.labels, n);

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

    pub fn evaluate_with_metric(&self, config: &TrialConfig, metric: &str, seed: u64) -> f64 {
        let result = self.evaluate(config, seed);
        result.get(metric)
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
            _ => panic!("Unknown metric: {}. Options: trustworthiness, continuity, knn_overlap, geodesic_distortion_gu2019, geodesic_distortion_mse, davies_bouldin_ratio, dunn_index, class_density_measure, cluster_density_measure", metric),
        }
    }
}
