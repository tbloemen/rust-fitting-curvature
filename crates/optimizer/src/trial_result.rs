use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;

use crate::metrics::MetricValues;
use crate::search_space::TrialConfig;
use fitting_core::spread::SpreadDiagnostics;

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

    /// Every metric, flattened to the top level so each is its own JSONL
    /// column — the same schema the sixteen `Option<f64>` fields here used to
    /// produce, and the same key order, since `MetricValues` serialises in
    /// `metrics::ALL` order and that order is this struct's old field order.
    #[serde(flatten)]
    pub(crate) metrics: MetricValues,
    /// How far the embedding reaches. Flattened second so `r_max`, `r_rms` and
    /// `r_gyration` land after the metric block and before `time_ms`, which is
    /// where they have always been.
    #[serde(flatten)]
    pub(crate) spread: SpreadDiagnostics,

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
            metrics: MetricValues::MISSING,
            spread: SpreadDiagnostics::MISSING,
            time_ms,
            scan_param: None,
        }
    }

    pub(crate) fn with_all_metrics(mut self, m: &MetricValues, spread: &SpreadDiagnostics) -> Self {
        self.metrics = *m;
        self.spread = *spread;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::parse_experiment;

    fn unscored() -> TrialResult {
        let mut rng = fitting_core::synthetic_data::Rng::new(1);
        let config = parse_experiment("all_free").sample(&mut rng);
        TrialResult::new(&config, "tree", 100, 1, 0.0, 5)
    }

    /// A result that was never scored must still carry both blocks whole,
    /// every column `null`.
    ///
    /// `--mode scan` builds results this way — it calls `TrialResult::new` and
    /// never `with_all_metrics` — so this is that path's on-disk shape. It is a
    /// test rather than a run of the binary because `--mode scan` panics before
    /// it writes anything, for reasons that predate this work
    /// (`ParamSpec::value()` on an `Optimize` spec, `search_space.rs:106`).
    #[test]
    fn an_unscored_result_writes_the_metric_block_as_nulls() {
        let json = serde_json::to_value(unscored()).unwrap();
        let obj = json.as_object().unwrap();
        let columns = crate::metrics::ALL_METRICS.iter().map(|m| m.name()).chain([
            "r_max",
            "r_rms",
            "r_gyration",
        ]);
        for column in columns {
            assert_eq!(
                obj.get(column),
                Some(&serde_json::Value::Null),
                "{column} is missing or not null"
            );
        }
    }

    /// The two flattened blocks sit between the hyperparameters and `time_ms`,
    /// metrics first and spread second, which is where the individual
    /// `Option<f64>` columns used to be. Key order is what makes a byte-diff
    /// against an existing results file mean anything.
    #[test]
    fn the_flattened_blocks_keep_their_position_in_the_line() {
        let json = serde_json::to_string(
            &unscored().with_all_metrics(&MetricValues::MISSING, &SpreadDiagnostics::MISSING),
        )
        .unwrap();
        let keys: Vec<&str> = json
            .trim_matches(|c| c == '{' || c == '}')
            .split(',')
            .map(|kv| kv.split(':').next().unwrap().trim_matches('"'))
            .collect();

        let first = keys.iter().position(|k| *k == "trustworthiness").unwrap();
        assert_eq!(keys[first - 1], "early_exaggeration_factor");

        let mut want: Vec<&str> = crate::metrics::ALL_METRICS
            .iter()
            .map(|m| m.name())
            .collect();
        want.extend(["r_max", "r_rms", "r_gyration", "time_ms"]);
        assert_eq!(&keys[first..first + want.len()], &want[..]);
    }
}
