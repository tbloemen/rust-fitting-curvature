use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;

use crate::metrics::MetricValues;
use fitting_core::metrics::{
    CLUSTER_DENSITY_MEASURE, CONTINUITY, CONTINUITY_MANIFOLD, DAVIES_BOULDIN_RATIO, DUNN_INDEX,
    NEIGHBORHOOD_HIT, NEIGHBORHOOD_HIT_MANIFOLD, NORMALIZED_STRESS, NORMALIZED_STRESS_MANIFOLD,
    R_GYRATION, R_MAX, R_RMS, SHEPARD_GOODNESS, SHEPARD_GOODNESS_MANIFOLD, TRUSTWORTHINESS,
    TRUSTWORTHINESS_MANIFOLD,
};
use crate::search_space::TrialConfig;

#[derive(Debug, Serialize)]
pub(crate) struct TrialResult {
    pub(crate) dataset_name: String,
    pub(crate) n_samples: usize,
    pub(crate) n_seeds: usize,
    pub(crate) curvature: f64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) geometry: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) curvature_magnitude: Option<f64>,

    pub(crate) learning_rate: f64,
    pub(crate) perplexity_ratio: f64,
    pub(crate) momentum_main: f64,
    pub(crate) centering_weight: f64,
    pub(crate) global_loss_weight: f64,
    pub(crate) norm_loss_weight: f64,
    pub(crate) early_exaggeration_factor: f64,

    pub(crate) trustworthiness: Option<f64>,
    pub(crate) trustworthiness_manifold: Option<f64>,
    pub(crate) continuity: Option<f64>,
    pub(crate) continuity_manifold: Option<f64>,
    pub(crate) neighborhood_hit: Option<f64>,
    pub(crate) neighborhood_hit_manifold: Option<f64>,
    pub(crate) normalized_stress: Option<f64>,
    pub(crate) normalized_stress_manifold: Option<f64>,
    pub(crate) shepard_goodness: Option<f64>,
    pub(crate) shepard_goodness_manifold: Option<f64>,
    pub(crate) davies_bouldin_ratio: Option<f64>,
    pub(crate) dunn_index: Option<f64>,
    pub(crate) cluster_density_measure: Option<f64>,
    pub(crate) r_max: Option<f64>,
    pub(crate) r_rms: Option<f64>,
    pub(crate) r_gyration: Option<f64>,

    pub(crate) time_ms: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) scan_param: Option<String>,
}

impl TrialResult {
    pub(crate) fn new(
        config: &TrialConfig,
        dataset_name: &str,
        n_samples: usize,
        n_seeds: usize,
        curvature: f64,
        time_ms: u64,
    ) -> Self {
        Self {
            dataset_name: dataset_name.to_string(),
            n_samples,
            n_seeds,
            curvature,
            geometry: None,
            curvature_magnitude: None,
            learning_rate: config.learning_rate.value(),
            perplexity_ratio: config.perplexity_ratio.value(),
            momentum_main: config.momentum_main.value(),
            centering_weight: config.centering_weight.value(),
            global_loss_weight: config.global_loss_weight.value(),
            norm_loss_weight: config.norm_loss_weight.value(),
            early_exaggeration_factor: config.early_exaggeration_factor.value(),
            trustworthiness: None,
            trustworthiness_manifold: None,
            continuity: None,
            continuity_manifold: None,
            neighborhood_hit: None,
            neighborhood_hit_manifold: None,
            normalized_stress: None,
            normalized_stress_manifold: None,
            shepard_goodness: None,
            shepard_goodness_manifold: None,
            davies_bouldin_ratio: None,
            dunn_index: None,
            cluster_density_measure: None,
            r_max: None,
            r_rms: None,
            r_gyration: None,
            time_ms,
            scan_param: None,
        }
    }

    pub(crate) fn with_all_metrics(mut self, m: &MetricValues) -> Self {
        self.trustworthiness = m.get(TRUSTWORTHINESS);
        self.trustworthiness_manifold = m.get(TRUSTWORTHINESS_MANIFOLD);
        self.continuity = m.get(CONTINUITY);
        self.continuity_manifold = m.get(CONTINUITY_MANIFOLD);
        self.neighborhood_hit = m.get(NEIGHBORHOOD_HIT);
        self.neighborhood_hit_manifold = m.get(NEIGHBORHOOD_HIT_MANIFOLD);
        self.normalized_stress = m.get(NORMALIZED_STRESS);
        self.normalized_stress_manifold = m.get(NORMALIZED_STRESS_MANIFOLD);
        self.shepard_goodness = m.get(SHEPARD_GOODNESS);
        self.shepard_goodness_manifold = m.get(SHEPARD_GOODNESS_MANIFOLD);
        self.davies_bouldin_ratio = m.get(DAVIES_BOULDIN_RATIO);
        self.dunn_index = m.get(DUNN_INDEX);
        self.cluster_density_measure = m.get(CLUSTER_DENSITY_MEASURE);
        self.r_max = m.get(R_MAX);
        self.r_rms = m.get(R_RMS);
        self.r_gyration = m.get(R_GYRATION);
        self
    }
}

pub(crate) fn write_result(result: &TrialResult, out_path: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .unwrap();
    let json = serde_json::to_string(result).unwrap();
    writeln!(file, "{}", json).ok();
}
