use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::evaluate::Evaluator;
use crate::metrics::{Metric, MetricValues};
use crate::search_space::TrialConfig;
use fitting_core::spread::SpreadDiagnostics;

// ─── Experiment variants ──────────────────────────────────────────────────────

pub(crate) fn parse_experiment(name: &str) -> TrialConfig {
    match name {
        "all_off" => TrialConfig::all_off(),
        "centering_only" => TrialConfig::centering_only(),
        "global_only" => TrialConfig::global_only(),
        "norm_only" => TrialConfig::norm_only(),
        "all_free" => TrialConfig::all_free(),
        "rms_anchored" => TrialConfig::rms_anchored(),
        other => {
            eprintln!(
                "Unknown --experiment '{other}'. Valid: all_off, centering_only, global_only, \
                 norm_only, all_free, rms_anchored."
            );
            std::process::exit(1);
        }
    }
}

// ─── Shared evaluation helpers ────────────────────────────────────────────────

pub(crate) fn trial_seed(trial_idx: usize, seed_idx: usize) -> u64 {
    42 + trial_idx as u64 * 100 + seed_idx as u64
}

pub(crate) fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

pub(crate) fn eval_single_metric(
    evaluator: &Evaluator,
    config: &TrialConfig,
    curvature: f64,
    metric: &str,
    n_seeds: usize,
    trial_idx: usize,
    pb_iters: &ProgressBar,
) -> (f64, f64) {
    // A seed whose reading is absent is dropped rather than substituted: this
    // feeds `--mode scan`'s mean/std, which describe the seeds that produced a
    // number. `mean_std` over an empty slice yields NaN, which is the honest
    // answer when no seed measured anything.
    let values: Vec<f64> = (0..n_seeds)
        .filter_map(|si| {
            evaluator
                .evaluate_with_metric(
                    config,
                    curvature,
                    metric,
                    trial_seed(trial_idx, si),
                    pb_iters,
                )
                .value()
        })
        .collect();
    mean_std(&values)
}

pub(crate) fn eval_all_metrics(
    evaluator: &Evaluator,
    config: &TrialConfig,
    curvature: f64,
    n_seeds: usize,
    trial_idx: usize,
    pb_iters: &ProgressBar,
) -> (MetricValues, SpreadDiagnostics) {
    let (metrics, spread): (Vec<MetricValues>, Vec<SpreadDiagnostics>) = (0..n_seeds)
        .map(|si| {
            evaluator.compute_all_metrics(config, curvature, trial_seed(trial_idx, si), pb_iters)
        })
        .unzip();
    (
        MetricValues::mean(&metrics),
        SpreadDiagnostics::mean(&spread),
    )
}

pub(crate) fn make_progress_bar(mp: &MultiProgress, total: u64, template: &str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(template)
            .unwrap()
            .progress_chars("=>-"),
    );
    pb
}

pub(crate) fn parse_metric(name: &str) -> Metric {
    Metric::by_name(name).unwrap_or_else(|| {
        eprintln!(
            "Unknown metric '{}'. Valid options: {}",
            name,
            Metric::valid_names()
        );
        std::process::exit(1);
    })
}
