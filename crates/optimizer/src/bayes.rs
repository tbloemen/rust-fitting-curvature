use indicatif::{MultiProgress, ProgressBar};
use std::sync::Arc;
use std::thread;

use crate::cli::Args;
use crate::common::{eval_all_metrics, make_progress_bar, parse_experiment, parse_metric};
use crate::evaluate::Evaluator;
use crate::gp::{GpOptimizer, GpState};
use crate::metrics::AllMetrics;
use crate::search_space::{param_bounds, ParamSpec, SearchSpace, TrialConfig};
use crate::trial_result::{write_result, TrialResult};

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
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
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
    evaluator: Arc<Evaluator>,
    mp: &MultiProgress,
    batch_size: usize,
) {
    let metric = parse_metric(args.metric.as_deref().unwrap());
    let direction = metric.direction();

    let (geometry, curvature_sign) = resolve_geometry(args, &evaluator);
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
        direction,
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
    pb.set_message(format!("{} (sign={:+.0})", geometry, curvature_sign));
    pb.set_prefix("n/a");
    if n_warm > 0 {
        pb.println(format!(
            "bayes '{}' ({}) warm-started from {} prior trials",
            dataset_name, geometry, n_warm
        ));
    }
    pb.println(format!(
        "bayes '{}' ({}) running with batch_size={}",
        dataset_name, geometry, batch_size
    ));

    let mut completed = 0usize;
    let mut remaining = args.n_trials;

    while remaining > 0 {
        let this_batch = batch_size.min(remaining);

        // Ask the GP for the top-`this_batch` promising configs in one shot.
        let configs = optimizer.suggest_batch(this_batch, &mut rng);

        // Evaluate all configs in this batch in parallel, then collect results.
        let results: Vec<(f64, AllMetrics, u64)> = thread::scope(|s| {
            configs
                .iter()
                .enumerate()
                .map(|(i, config)| {
                    let evaluator = &*evaluator;
                    let actual_curvature = curvature_sign * config.curvature_magnitude.value();
                    let trial_idx = completed + i + 1;
                    s.spawn(move || {
                        let pb_iters = ProgressBar::hidden();
                        let start = std::time::Instant::now();
                        let all = eval_all_metrics(
                            evaluator,
                            config,
                            curvature_sign,
                            args.n_seeds,
                            trial_idx,
                            &pb_iters,
                        );
                        let elapsed = start.elapsed().as_millis() as u64;
                        (actual_curvature, all, elapsed)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });

        // Observe all results and update the GP before the next round.
        for (config, (actual_curvature, all, elapsed)) in configs.iter().zip(results.iter()) {
            let mean = metric.value(all);
            optimizer.observe(config.clone(), mean);
            completed += 1;

            let mut result = TrialResult::new(
                config,
                dataset_name,
                args.n_samples,
                args.n_seeds,
                *actual_curvature,
                *elapsed,
            )
            .with_all_metrics(all);
            result.geometry = Some(geometry.to_string());
            if optimize_curvature {
                result.curvature_magnitude = Some(config.curvature_magnitude.value());
            }
            write_result(&result, out_path);

            let best = optimizer.best_trial();
            pb.set_prefix(format!("{:.4}", best));
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
        }

        remaining -= this_batch;
    }

    pb.finish_with_message(format!("{} ({}) done", dataset_name, geometry));

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

    // Write GP state for external plotting (analyze_hyperparams.py --mode gp).
    if let Some(state) = optimizer.export_state() {
        let stem = out_path
            .trim_end_matches(".jsonl")
            .trim_end_matches(".json");
        let state_path = format!("{}_gp_{}_{}.json", stem, dataset_name, geometry);
        write_gp_state(&state, &state_path);
        pb.println(format!("GP state written to {}", state_path));
    }
}

fn write_gp_state(state: &GpState, path: &str) {
    match serde_json::to_string_pretty(state) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                eprintln!("Failed to write GP state to {}: {}", path, e);
            }
        }
        Err(e) => eprintln!("Failed to serialise GP state: {}", e),
    }
}
