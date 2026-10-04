use crate::cli::Args;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::evaluate::Evaluator;
use crate::metrics::MetricValues;
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

/// The target geometry (name + curvature sign) from `--geometry`.
pub(crate) fn resolve_geometry(args: &Args) -> (&'static str, f64) {
    match args.geometry.as_str() {
        "hyperbolic" => ("hyperbolic", -1.0),
        "spherical" => ("spherical", 1.0),
        _ => ("euclidean", 0.0),
    }
}
