use indicatif::{MultiProgress, ProgressBar};
use std::thread;

use crate::cli::Args;
use crate::common::{eval_all_metrics, make_progress_bar, parse_experiment, parse_metric};
use crate::evaluate::Evaluator;
use crate::gp::{GpOptimizer, GpState};
use crate::metrics::{Direction, Metric, MetricValues};
use crate::search_space::{param_bounds, ParamSpec, SearchSpace, TrialConfig};
use crate::trial_result::{write_result, TrialResult};
use fitting_core::spread::SpreadDiagnostics;

// ─── Bayesian optimisation (Algorithm 1, Frazier 2018) ───────────────────────

/// Resolve the target geometry (name + curvature sign) from CLI args or auto-detection.
///
/// `--geometry hyperbolic|spherical|euclidean` forces the choice; omitting it triggers
/// geometry detection via `evaluator.infer_geometry()`.
pub(crate) fn resolve_geometry(args: &Args, evaluator: &Evaluator) -> (&'static str, f64) {
    if let Some(geo) = &args.geometry {
        return match geo.as_str() {
            "hyperbolic" => ("hyperbolic", -1.0),
            "spherical" => ("spherical", 1.0),
            _ => ("euclidean", 0.0),
        };
    }
    let detection = evaluator.infer_geometry();
    let g = detection.best_geometry;
    eprintln!(
        "Geometry auto-detected: {} with curvature {}",
        g, detection.curvature
    );
    let sign: f64 = match g {
        "hyperbolic" => -1.0,
        "spherical" => 1.0,
        _ => 0.0,
    };
    (g, sign)
}

/// Load warm-start trials from a JSONL file, filtering by dataset + geometry field.
fn load_warm_start_trials(
    path: &str,
    metric: &str,
    dataset_name: &str,
    geometry: &str,
) -> Vec<(TrialConfig, f64)> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return vec![];
    };
    content
        .lines()
        .filter_map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).ok()?;
            if v["dataset_name"].as_str()? != dataset_name {
                return None;
            }
            if v["geometry"].as_str()? != geometry {
                return None;
            }
            let metric_val = v[metric].as_f64()?;
            if !metric_val.is_finite() {
                return None;
            }
            let mut config = TrialConfig::all_free();
            config.learning_rate = ParamSpec::Fixed(v["learning_rate"].as_f64()?);
            config.perplexity_ratio = ParamSpec::Fixed(v["perplexity_ratio"].as_f64()?);
            config.momentum_main = ParamSpec::Fixed(v["momentum_main"].as_f64()?);
            config.centering_weight =
                ParamSpec::Fixed(v["centering_weight"].as_f64().unwrap_or(0.0));
            config.global_loss_weight =
                ParamSpec::Fixed(v["global_loss_weight"].as_f64().unwrap_or(0.0));
            config.norm_loss_weight =
                ParamSpec::Fixed(v["norm_loss_weight"].as_f64().unwrap_or(0.0));
            config.early_exaggeration_factor =
                ParamSpec::Fixed(v["early_exaggeration_factor"].as_f64().unwrap_or(12.0));
            config.curvature_magnitude =
                ParamSpec::Fixed(v["curvature_magnitude"].as_f64().unwrap_or(0.0));
            Some((config, metric_val))
        })
        .collect()
}

/// Bayesian optimisation over 6 (or 7 with curvature magnitude) hyperparameters.
///
/// Geometry is resolved once via `--geometry` or auto-detection.  For non-Euclidean
/// geometries the curvature magnitude is included as a 7th BO dimension.
///
/// Parallel evaluation uses a **round-based batch** strategy: each round the GP
/// scores `n_ei_candidates` candidates and returns the top-`batch_size` by Expected
/// Improvement, which are then evaluated in parallel via `thread::scope`.  After
/// every round the GP is updated with all real results before the next suggest.
pub(crate) fn run_bayes(
    dataset_name: &str,
    args: &Args,
    evaluator: &Evaluator,
    mp: &MultiProgress,
    batch_size: usize,
) {
    let metric = parse_metric(args.metric.as_deref().unwrap());
    let direction = metric.direction();

    let (geometry, curvature_sign) = resolve_geometry(args, evaluator);
    let optimize_curvature = curvature_sign != 0.0;

    // Curvature magnitude bounds: take abs() of the signed range limits so that
    // e.g. --curvature-min -5 --curvature-max 5 → magnitude [0.001, 5.0].
    let curvature_mag_min = param_bounds("curvature_magnitude").0;
    let curvature_mag_max = args
        .curvature_max
        .abs()
        .max(args.curvature_min.abs())
        .max(curvature_mag_min);
    let mut hp = parse_experiment(&args.experiment);
    if optimize_curvature {
        hp.curvature_magnitude = ParamSpec::Optimize {
            lo: curvature_mag_min,
            hi: curvature_mag_max,
            log_scale: true,
        };
    }
    let mut optimizer = GpOptimizer::new(SearchSpace {
        direction: direction.into(),
        hyper_params: hp,
    });
    let mut rng = fitting_core::synthetic_data::Rng::new(0xdead_beef_cafe_0000);

    // Warm-start from prior results matching this dataset + geometry.
    let n_warm = if let Some(warm_file) = &args.warm_start {
        let trials = load_warm_start_trials(warm_file, metric.name(), dataset_name, geometry);
        let n = trials.len();
        for (config, metric_val) in trials {
            optimizer.observe(config, metric_val);
        }
        n
    } else {
        0
    };

    let out_path = &args.output;
    let pb = make_progress_bar(
        mp,
        args.n_trials as u64,
        "{spinner:.green} bayes={msg} [{bar:35.cyan/blue}] {pos}/{len} | best: {prefix}",
    );
    pb.set_message(format!("{geometry} (sign={curvature_sign:+.0})"));
    pb.set_prefix("n/a");
    if n_warm > 0 {
        pb.println(format!(
            "bayes '{dataset_name}' ({geometry}) warm-started from {n_warm} prior trials"
        ));
    }
    pb.println(format!(
        "bayes '{dataset_name}' ({geometry}) running with batch_size={batch_size}"
    ));

    let mut completed = 0usize;
    let mut remaining = args.n_trials;

    while remaining > 0 {
        let this_batch = batch_size.min(remaining);

        // Ask the GP for the top-`this_batch` promising configs in one shot.
        let configs = optimizer.suggest_batch(this_batch, &mut rng);

        // Evaluate all configs in this batch in parallel, then collect results.
        let results = evaluate_batch(evaluator, &configs, curvature_sign, args, completed);

        // Observe all results and update the GP before the next round.
        for (config, outcome) in configs.iter().zip(results.iter()) {
            completed = observe_batch_result(
                config,
                outcome,
                &mut optimizer,
                &metric,
                completed,
                dataset_name,
                args,
                geometry,
                optimize_curvature,
                out_path,
                &pb,
            );
        }

        remaining -= this_batch;
    }

    pb.finish_with_message(format!("{dataset_name} ({geometry}) done"));

    report_best(
        &optimizer,
        dataset_name,
        geometry,
        &metric,
        curvature_sign,
        &pb,
    );
    write_gp_state_file(&optimizer, dataset_name, geometry, out_path, &pb);
}

/// Evaluate one batch of configs in parallel, returning the per-trial result:
/// actual curvature, all metrics, spread diagnostics and elapsed milliseconds.
fn evaluate_batch(
    evaluator: &Evaluator,
    configs: &[TrialConfig],
    curvature_sign: f64,
    args: &Args,
    completed: usize,
) -> Vec<(f64, MetricValues, SpreadDiagnostics, u64)> {
    thread::scope(|s| {
        configs
            .iter()
            .enumerate()
            .map(|(i, config)| {
                let actual_curvature = curvature_sign * config.curvature_magnitude.value();
                let trial_idx = completed + i + 1;
                s.spawn(move || {
                    let pb_iters = ProgressBar::hidden();
                    let start = std::time::Instant::now();
                    let (all, spread) = eval_all_metrics(
                        evaluator,
                        config,
                        curvature_sign,
                        args.n_seeds,
                        trial_idx,
                        &pb_iters,
                    );
                    let elapsed = u64::try_from(start.elapsed().as_millis())
                        .expect("elapsed millis fit in u64");
                    (actual_curvature, all, spread, elapsed)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    })
}

/// Fold one completed trial into the GP, write its JSONL record and update the
/// progress bar, returning the updated completion count.
#[expect(
    clippy::too_many_arguments,
    reason = "bundles the per-trial logging state"
)]
fn observe_batch_result(
    config: &TrialConfig,
    outcome: &(f64, MetricValues, SpreadDiagnostics, u64),
    optimizer: &mut GpOptimizer,
    metric: &Metric,
    completed: usize,
    dataset_name: &str,
    args: &Args,
    geometry: &str,
    optimize_curvature: bool,
    out_path: &str,
    pb: &ProgressBar,
) -> usize {
    let (actual_curvature, all, spread, elapsed) = outcome;
    // An unmeasured reading — a diverged embedding, chiefly — scores as
    // the worst value in the metric's direction, matching what
    // `metrics_to_vec` does for the pareto path.
    let mean = all.get(*metric).unwrap_or(match metric.direction() {
        Direction::Maximize => 0.0,
        Direction::Minimize => 1.0,
    });
    optimizer.observe(config.clone(), mean);
    let completed = completed + 1;

    let mut result = TrialResult::new(
        config,
        dataset_name,
        args.n_samples,
        args.n_seeds,
        *actual_curvature,
        *elapsed,
    )
    .with_all_metrics(all, spread);
    result.geometry = Some(geometry.to_string());
    if optimize_curvature {
        result.curvature_magnitude = Some(config.curvature_magnitude.value());
    }
    write_result(&result, out_path);

    let best = optimizer.best_trial();
    pb.set_prefix(format!("{best:.4}"));
    pb.println(format!(
        "bayes '{}' trial {:3}/{} | {}={:.4} | best={:.4} | {}ms \
         | k={:.3} lr={:.4} perp={:.4}",
        dataset_name,
        completed,
        args.n_trials,
        metric.name(),
        mean,
        best,
        *elapsed,
        actual_curvature,
        config.learning_rate.value(),
        config.perplexity_ratio.value(),
    ));
    pb.inc(1);
    completed
}

fn report_best(
    optimizer: &GpOptimizer,
    dataset_name: &str,
    geometry: &str,
    metric: &Metric,
    curvature_sign: f64,
    pb: &ProgressBar,
) {
    if let Some(best) = optimizer.best_config() {
        pb.println(format!(
            "\n=== Best for '{}' ({}) | {}={:.4} ===\n  \
             k={:.3}  lr={:.4}  perp_ratio={:.4}  momentum={:.4}\n  \
             centering={:.3}  global_loss={:.3}  norm={:.4}",
            dataset_name,
            geometry,
            metric.name(),
            optimizer.best_trial(),
            curvature_sign * best.curvature_magnitude.value(),
            best.learning_rate.value(),
            best.perplexity_ratio.value(),
            best.momentum_main.value(),
            best.centering_weight.value(),
            best.global_loss_weight.value(),
            best.norm_loss_weight.value(),
        ));
    }
}

// Write GP state for external plotting (analyze_hyperparams.py --mode gp).
fn write_gp_state_file(
    optimizer: &GpOptimizer,
    dataset_name: &str,
    geometry: &str,
    out_path: &str,
    pb: &ProgressBar,
) {
    if let Some(state) = optimizer.export_state() {
        let stem = out_path
            .trim_end_matches(".jsonl")
            .trim_end_matches(".json");
        let state_path = format!("{stem}_gp_{dataset_name}_{geometry}.json");
        write_gp_state(&state, &state_path);
        pb.println(format!("GP state written to {state_path}"));
    }
}

fn write_gp_state(state: &GpState, path: &str) {
    match serde_json::to_string_pretty(state) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                eprintln!("Failed to write GP state to {path}: {e}");
            }
        }
        Err(e) => eprintln!("Failed to serialise GP state: {e}"),
    }
}
