use crate::bayes::resolve_geometry;
use crate::cli::Args;
use crate::common::{eval_single_metric, make_progress_bar, parse_experiment};
use crate::evaluate::Evaluator;
use crate::search_space::TrialConfig;
use crate::trial_result::{write_result, TrialResult};
use indicatif::{MultiProgress, ProgressBar};
use std::sync::Arc;

fn load_best_config_from_jsonl(
    path: &str,
    n_points: usize,
    dataset_name: &str,
    geometry: &str,
) -> Option<TrialConfig> {
    use crate::search_space::ParamSpec;
    let content = std::fs::read_to_string(path).ok()?;
    let mut best_val = f64::NEG_INFINITY;
    let mut best: Option<TrialConfig> = None;

    for line in content.lines() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if v["dataset_name"].as_str().unwrap_or("") != dataset_name {
            continue;
        }
        if v["geometry"].as_str().unwrap_or("") != geometry {
            continue;
        }
        let metric = v["metric_mean"].as_f64().unwrap_or(f64::NEG_INFINITY);
        if metric > best_val {
            best_val = metric;
            let perp_ratio = v["perplexity_ratio"]
                .as_f64()
                .unwrap_or_else(|| v["perplexity"].as_f64().unwrap_or(15.0) / n_points as f64);
            let mut hp = TrialConfig::all_free();
            hp.learning_rate = ParamSpec::Fixed(v["learning_rate"].as_f64().unwrap_or(10.0));
            hp.perplexity_ratio = ParamSpec::Fixed(perp_ratio);
            hp.momentum_main = ParamSpec::Fixed(v["momentum_main"].as_f64().unwrap_or(0.8));
            hp.centering_weight = ParamSpec::Fixed(v["centering_weight"].as_f64().unwrap_or(0.0));
            hp.global_loss_weight =
                ParamSpec::Fixed(v["global_loss_weight"].as_f64().unwrap_or(0.0));
            hp.norm_loss_weight = ParamSpec::Fixed(v["norm_loss_weight"].as_f64().unwrap_or(0.0));
            hp.early_exaggeration_factor =
                ParamSpec::Fixed(v["early_exaggeration_factor"].as_f64().unwrap_or(12.0));
            hp.curvature_magnitude =
                ParamSpec::Fixed(v["curvature_magnitude"].as_f64().unwrap_or(0.0));
            best = Some(hp);
        }
    }
    best
}

fn sweep_values(lo: f64, hi: f64, n: usize, log: bool) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            if log {
                (lo.ln() + t * (hi.ln() - lo.ln())).exp()
            } else {
                lo + t * (hi - lo)
            }
        })
        .collect()
}

fn apply_param(config: &mut TrialConfig, param: &str, val: f64) {
    use crate::search_space::ParamSpec;
    let fixed = ParamSpec::Fixed(val);
    match param {
        "learning_rate" => config.learning_rate = fixed,
        "perplexity_ratio" => config.perplexity_ratio = fixed,
        "momentum_main" => config.momentum_main = fixed,
        "centering_weight" => config.centering_weight = fixed,
        "global_loss_weight" => config.global_loss_weight = fixed,
        "norm_loss_weight" => config.norm_loss_weight = fixed,
        "early_exaggeration_factor" => config.early_exaggeration_factor = fixed,
        "curvature_magnitude" => config.curvature_magnitude = fixed,
        _ => unreachable!(),
    }
}

/// Parameter sweep scan: each hyperparameter is swept individually from a base config.
///
/// Geometry is resolved once (via `--geometry` or auto-detection).  When the geometry
/// is non-Euclidean, curvature magnitude is also swept as an additional parameter.
pub fn run_scan(dataset_name: &str, args: &Args, evaluator: Arc<Evaluator>, mp: &MultiProgress) {
    let metric = args.metric.as_deref().unwrap();
    let n_points = evaluator.n_points();

    let (geometry, curvature_sign) = resolve_geometry(args, &evaluator);
    let optimize_curvature = curvature_sign != 0.0;

    let hp = parse_experiment(&args.experiment);
    use crate::search_space::ParamSpec;
    let mut default_config = hp.clone();
    // Override with sensible scan baseline values for the free parameters.
    default_config.learning_rate = ParamSpec::Fixed(10.0);
    default_config.perplexity_ratio = ParamSpec::Fixed(0.003);
    default_config.momentum_main = ParamSpec::Fixed(0.8);
    default_config.early_exaggeration_factor = ParamSpec::Fixed(12.0);
    default_config.curvature_magnitude =
        ParamSpec::Fixed(if optimize_curvature { 1.0 } else { 0.0 });

    let base = if let Some(scan_file) = &args.scan_from {
        match load_best_config_from_jsonl(scan_file, n_points, dataset_name, geometry) {
            Some(c) => {
                eprintln!(
                    "scan '{}' ({}): loaded base config from {}",
                    dataset_name, geometry, scan_file
                );
                c
            }
            None => {
                eprintln!(
                    "scan '{}' ({}): could not load from {}, using default",
                    dataset_name, geometry, scan_file
                );
                default_config
            }
        }
    } else {
        eprintln!(
            "scan '{}' ({}): no --scan-from, using default config",
            dataset_name, geometry
        );
        default_config
    };

    let n = args.scan_steps;
    let curvature_mag_min = args
        .curvature_min
        .abs()
        .max(crate::search_space::param_bounds("curvature_magnitude").0);
    let curvature_mag_max = args.curvature_max.abs().max(curvature_mag_min);
    use crate::search_space::param_bounds;
    let mut params: Vec<(&str, Vec<f64>)> = Vec::new();
    if hp.learning_rate.is_optimized() {
        let (lo, hi, log) = param_bounds("learning_rate");
        params.push(("learning_rate", sweep_values(lo, hi, n, log)));
    }
    if hp.perplexity_ratio.is_optimized() {
        let (lo, hi, log) = param_bounds("perplexity_ratio");
        params.push(("perplexity_ratio", sweep_values(lo, hi, n, log)));
    }
    if hp.centering_weight.is_optimized() {
        let (lo, hi, log) = param_bounds("centering_weight");
        params.push(("centering_weight", sweep_values(lo, hi, n, log)));
    }
    if hp.global_loss_weight.is_optimized() {
        let (lo, hi, log) = param_bounds("global_loss_weight");
        params.push(("global_loss_weight", sweep_values(lo, hi, n, log)));
    }
    if hp.norm_loss_weight.is_optimized() {
        let (lo, hi, log) = param_bounds("norm_loss_weight");
        params.push(("norm_loss_weight", sweep_values(lo, hi, n, log)));
    }
    if hp.early_exaggeration_factor.is_optimized() {
        let (lo, hi, log) = param_bounds("early_exaggeration_factor");
        params.push(("early_exaggeration_factor", sweep_values(lo, hi, n, log)));
    }
    if optimize_curvature {
        params.push((
            "curvature_magnitude",
            sweep_values(curvature_mag_min, curvature_mag_max, n, true),
        ));
    }

    let total = params.iter().map(|(_, v)| v.len()).sum::<usize>() as u64;
    let pb = make_progress_bar(
        mp,
        total,
        "{spinner:.green} scan={msg} [{bar:35.cyan/blue}] {pos}/{len} {wide_msg}",
    );
    pb.set_message(format!("{} ({})", dataset_name, geometry));

    let out_path = &args.output;
    let pb_iters = ProgressBar::hidden();
    let mut trial_idx = 0usize;

    for (param_name, values) in &params {
        for &val in values {
            trial_idx += 1;
            let mut config = base.clone();
            apply_param(&mut config, param_name, val);

            let actual_curvature = curvature_sign * config.curvature_magnitude.value();

            let start = std::time::Instant::now();
            let (mean, std) = eval_single_metric(
                &evaluator,
                &config,
                curvature_sign,
                metric,
                args.n_seeds,
                trial_idx,
                &pb_iters,
            );
            let elapsed = start.elapsed().as_millis() as u64;

            let mut result = TrialResult::new(
                &config,
                dataset_name,
                args.n_samples,
                args.n_seeds,
                actual_curvature,
                elapsed,
            );
            result.geometry = Some(geometry.to_string());
            if optimize_curvature {
                result.curvature_magnitude = Some(config.curvature_magnitude.value());
            }
            result.scan_param = Some(param_name.to_string());
            write_result(&result, out_path);

            pb.set_message(format!(
                "{} | {}={:.4} → {:.4} ± {:.4}",
                geometry, param_name, val, mean, std
            ));
            pb.inc(1);
        }
    }

    pb.finish_with_message(format!("{} ({}) scan done", dataset_name, geometry));
}
