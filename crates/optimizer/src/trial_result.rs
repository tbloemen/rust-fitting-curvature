use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;

use crate::metrics::AllMetrics;
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
    pub(crate) knn_overlap: Option<f64>,
    pub(crate) knn_overlap_manifold: Option<f64>,
    pub(crate) neighborhood_hit: Option<f64>,
    pub(crate) neighborhood_hit_manifold: Option<f64>,
    pub(crate) normalized_stress: Option<f64>,
    pub(crate) normalized_stress_manifold: Option<f64>,
    pub(crate) shepard_goodness: Option<f64>,
    pub(crate) shepard_goodness_manifold: Option<f64>,
    pub(crate) davies_bouldin_ratio: Option<f64>,
    pub(crate) dunn_index: Option<f64>,
    pub(crate) class_density_measure: Option<f64>,
    pub(crate) cluster_density_measure: Option<f64>,
    pub(crate) r_max: Option<f64>,
    pub(crate) r_rms: Option<f64>,

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
            knn_overlap: None,
            knn_overlap_manifold: None,
            neighborhood_hit: None,
            neighborhood_hit_manifold: None,
            normalized_stress: None,
            normalized_stress_manifold: None,
            shepard_goodness: None,
            shepard_goodness_manifold: None,
            davies_bouldin_ratio: None,
            dunn_index: None,
            class_density_measure: None,
            cluster_density_measure: None,
            r_max: None,
            r_rms: None,
            time_ms,
            scan_param: None,
        }
    }

    pub(crate) fn with_all_metrics(mut self, m: &AllMetrics) -> Self {
        self.trustworthiness = Some(m.trustworthiness);
        self.trustworthiness_manifold = Some(m.trustworthiness_manifold);
        self.continuity = Some(m.continuity);
        self.continuity_manifold = Some(m.continuity_manifold);
        self.knn_overlap = Some(m.knn_overlap);
        self.knn_overlap_manifold = Some(m.knn_overlap_manifold);
        self.neighborhood_hit = Some(m.neighborhood_hit);
        self.neighborhood_hit_manifold = Some(m.neighborhood_hit_manifold);
        self.normalized_stress = Some(m.normalized_stress);
        self.normalized_stress_manifold = Some(m.normalized_stress_manifold);
        self.shepard_goodness = Some(m.shepard_goodness);
        self.shepard_goodness_manifold = Some(m.shepard_goodness_manifold);
        self.davies_bouldin_ratio = Some(m.davies_bouldin_ratio);
        self.dunn_index = Some(m.dunn_index);
        self.class_density_measure = Some(m.class_density_measure);
        self.cluster_density_measure = Some(m.cluster_density_measure);
        self.r_max = Some(m.r_max);
        self.r_rms = Some(m.r_rms);
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
