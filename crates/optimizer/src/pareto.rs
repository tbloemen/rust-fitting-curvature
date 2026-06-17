use crate::bayes::resolve_geometry;
use crate::cli::Args;
use crate::common::{make_progress_bar, parse_experiment};
use crate::evaluate::Evaluator;
use crate::gp::{MultiTrial, ParEgoOptimizer};
use crate::metrics::{AllMetrics, Metric};
use crate::resume::{eval_or_reuse_batch, load_prior_evals, BatchOutcome};
use crate::trial_result::{write_result, TrialResult};
use indicatif::MultiProgress;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub fn run_pareto(
    dataset_name: &str,
    args: &Args,
    evaluator: Arc<Evaluator>,
    mp: &MultiProgress,
    batch_size: usize,
) {
    let (geometry, curvature_sign) = resolve_geometry(args, &evaluator);
    let optimize_curvature = curvature_sign != 0.0;

    let curvature_mag_min = crate::search_space::param_bounds("curvature_magnitude").0;
    let curvature_mag_max = args
        .curvature_max
        .abs()
        .max(args.curvature_min.abs())
        .max(curvature_mag_min);

    let mut hp = parse_experiment(&args.experiment);
    if optimize_curvature {
        hp.curvature_magnitude = crate::search_space::ParamSpec::Optimize {
            lo: curvature_mag_min,
            hi: curvature_mag_max,
            log_scale: true,
        };
    }
    let metrics = default_pareto_metrics();
    let n_objectives = metrics.len();
    let mut optimizer = ParEgoOptimizer::new(metrics, hp);
    let mut rng = fitting_core::synthetic_data::Rng::new(0xdead_beef_cafe_2222);

    let out_path = &args.output;
    let lhs_total = optimizer.lhs_total();

    let stem = out_path
        .trim_end_matches(".jsonl")
        .trim_end_matches(".json");
    let front_path = format!("{}_pareto_{}_{}.json", stem, dataset_name, geometry);

    // ── Resume: replay already-recorded trials instead of re-evaluating them ──
    // `prior` is the ordered list of completed evaluations from a previous,
    // interrupted run of this exact (dataset, experiment, geometry) job. Their
    // suggestions are re-derived deterministically below; only the embedding
    // evaluation is skipped. An absent/empty file means a fresh start.
    let prior = if args.resume {
        load_prior_evals(out_path, optimizer.metrics.as_slice())
    } else {
        Vec::new()
    };
    let total_trials = lhs_total + args.n_trials;
    if !prior.is_empty() {
        if prior.len() >= total_trials && Path::new(&front_path).exists() {
            println!(
                "pareto '{}' ({}) already complete — {} trials recorded, front present; skipping.",
                dataset_name,
                geometry,
                prior.len()
            );
            return;
        }
        println!(
            "pareto '{}' ({}) resuming — {}/{} trials recorded; replaying then continuing.",
            dataset_name,
            geometry,
            prior.len().min(total_trials),
            total_trials
        );
    }

    // ── Phase 1: LHS init ────────────────────────────────────────────────────
    let pb = make_progress_bar(
        mp,
        lhs_total as u64,
        "{spinner:.cyan} [LHS] {msg} [{bar:35.cyan/blue}] {pos}/{len} ({eta})",
    );
    pb.set_message(format!("{} (sign={:+.0})", geometry, curvature_sign));
    pb.println(format!(
        "pareto '{}' ({}) — LHS init phase: {} points, {} objectives",
        dataset_name, geometry, lhs_total, n_objectives
    ));

    let mut lhs_completed = 0usize;
    while !optimizer.lhs_drained() {
        let remaining_lhs = lhs_total.saturating_sub(lhs_completed);
        let this_batch = batch_size.min(remaining_lhs.max(1));
        let configs = optimizer.suggest_batch(this_batch, &mut rng);

        let outcomes = eval_or_reuse_batch(
            &configs,
            lhs_completed,
            &prior,
            &evaluator,
            curvature_sign,
            args.n_seeds,
        );

        for (config, outcome) in configs.iter().zip(outcomes) {
            match outcome {
                BatchOutcome::Reused {
                    metric_vec,
                    r_max,
                    r_rms,
                } => {
                    optimizer.observe(config.clone(), metric_vec, r_max, r_rms);
                }
                BatchOutcome::Fresh {
                    all,
                    actual_curvature,
                    elapsed_ms,
                } => {
                    let metric_vec = metrics_to_vec(&all, optimizer.metrics.as_slice());
                    optimizer.observe(config.clone(), metric_vec, all.r_max, all.r_rms);

                    let mut result = TrialResult::new(
                        config,
                        dataset_name,
                        args.n_samples,
                        args.n_seeds,
                        actual_curvature,
                        elapsed_ms,
                    )
                    .with_all_metrics(&all);
                    result.geometry = Some(geometry.to_string());
                    if optimize_curvature {
                        result.curvature_magnitude = Some(config.curvature_magnitude.value());
                    }
                    write_result(&result, out_path);
                }
            }
            lhs_completed += 1;
            pb.inc(1);
        }
    }
    pb.finish_with_message(format!("{} LHS done ({} points)", geometry, lhs_completed));

    // ── Phase 2: GP optimisation ─────────────────────────────────────────────
    let pb = make_progress_bar(
        mp,
        args.n_trials as u64,
        "{spinner:.green} [GP]  {msg} [{bar:35.cyan/blue}] {pos}/{len} | front: {prefix} ({eta})",
    );
    pb.set_message(format!("{} (sign={:+.0})", geometry, curvature_sign));
    pb.set_prefix("0");
    pb.println(format!(
        "pareto '{}' ({}) — GP phase: {} trials, batch_size={}",
        dataset_name, geometry, args.n_trials, batch_size
    ));

    let mut completed = 0usize;
    let mut remaining = args.n_trials;

    while remaining > 0 {
        let this_batch = batch_size.min(remaining);
        let configs = optimizer.suggest_batch(this_batch, &mut rng);
        let base_global = lhs_total + completed;

        let outcomes = eval_or_reuse_batch(
            &configs,
            base_global,
            &prior,
            &evaluator,
            curvature_sign,
            args.n_seeds,
        );

        for (config, outcome) in configs.iter().zip(outcomes) {
            match outcome {
                BatchOutcome::Reused {
                    metric_vec,
                    r_max,
                    r_rms,
                } => {
                    // Replayed from the checkpoint — already in the JSONL, so
                    // don't rewrite it; just rebuild the optimizer's state.
                    optimizer.observe(config.clone(), metric_vec, r_max, r_rms);
                }
                BatchOutcome::Fresh {
                    all,
                    actual_curvature,
                    elapsed_ms,
                } => {
                    let metric_vec = metrics_to_vec(&all, optimizer.metrics.as_slice());
                    optimizer.observe(config.clone(), metric_vec, all.r_max, all.r_rms);

                    let mut result = TrialResult::new(
                        config,
                        dataset_name,
                        args.n_samples,
                        args.n_seeds,
                        actual_curvature,
                        elapsed_ms,
                    )
                    .with_all_metrics(&all);
                    result.geometry = Some(geometry.to_string());
                    if optimize_curvature {
                        result.curvature_magnitude = Some(config.curvature_magnitude.value());
                    }
                    write_result(&result, out_path);

                    let front_size = optimizer.pareto_front_indices().len();
                    pb.set_prefix(format!("{}", front_size));
                    pb.println(format!(
                        "pareto '{}' GP {:3}/{} | front={} | {}ms | k={:.3} lr={:.4} perp={:.4}",
                        dataset_name,
                        completed + 1,
                        args.n_trials,
                        front_size,
                        elapsed_ms,
                        actual_curvature,
                        config.learning_rate.value(),
                        config.perplexity_ratio.value(),
                    ));
                }
            }
            completed += 1;
            pb.inc(1);
        }

        remaining -= this_batch;
    }

    pb.finish_with_message(format!("{} ({}) done", dataset_name, geometry));

    let front = optimizer.pareto_trials();
    write_pareto_front(&front, &optimizer.metrics, args.n_samples, &front_path);
    pb.println(format!("Pareto front written to {}", front_path));
}

/// Default set of objectives for --mode pareto.
///
/// Includes both the 2D (post-projection) and manifold (pre-projection) variants
/// of the five core DR quality metrics, giving 10 objectives total.
fn default_pareto_metrics() -> Vec<Metric> {
    vec![
        Metric::Trustworthiness,
        Metric::TrustworthinessManifold,
        Metric::Continuity,
        Metric::ContinuityManifold,
        Metric::NormalizedStress,
        Metric::NormalizedStressManifold,
        Metric::ShepardGoodness,
        Metric::ShepardGoodnessManifold,
        Metric::NeighborhoodHit,
        Metric::NeighborhoodHitManifold,
    ]
}

/// Build the objective vector fed to the optimizer. A diverged embedding (e.g. an
/// unbounded Euclidean run that blew up to inf/NaN) yields non-finite metric
/// values; substitute each metric's worst finite value so the trial is scored as
/// bad rather than poisoning the GP normalisation or panicking the Pareto sorts.
/// The raw (possibly non-finite) values are still recorded in the JSONL via
/// `with_all_metrics`, so diverged trials remain visible in the results.
pub(crate) fn metrics_to_vec(m: &AllMetrics, metrics: &[Metric]) -> Vec<f64> {
    use crate::search_space::OptimizeDirection;
    metrics
        .iter()
        .map(|metric| {
            let v = metric.value(m);
            if v.is_finite() {
                v
            } else {
                match metric.direction() {
                    // The maximised metrics here are bounded below by 0 (0 = degenerate);
                    // normalized_stress is minimised and bounded above by 1 (1 = worst).
                    OptimizeDirection::Maximize => 0.0,
                    OptimizeDirection::Minimize => 1.0,
                }
            }
        })
        .collect()
}

fn write_pareto_front(front: &[&MultiTrial], metrics: &[Metric], n_samples: usize, path: &str) {
    #[derive(Serialize)]
    struct ParetoEntry<'a> {
        n_samples: usize,
        learning_rate: f64,
        perplexity_ratio: f64,
        momentum_main: f64,
        centering_weight: f64,
        global_loss_weight: f64,
        norm_loss_weight: f64,
        early_exaggeration_factor: f64,
        curvature_magnitude: f64,
        r_max: f64,
        r_rms: f64,
        metrics: HashMap<&'a str, f64>,
    }

    let entries: Vec<ParetoEntry> = front
        .iter()
        .map(|t| {
            let mut metric_map = HashMap::new();
            for (metric, &v) in metrics.iter().zip(&t.metrics) {
                metric_map.insert(metric.name(), v);
            }
            ParetoEntry {
                n_samples,
                learning_rate: t.config.learning_rate.value(),
                perplexity_ratio: t.config.perplexity_ratio.value(),
                momentum_main: t.config.momentum_main.value(),
                centering_weight: t.config.centering_weight.value(),
                global_loss_weight: t.config.global_loss_weight.value(),
                norm_loss_weight: t.config.norm_loss_weight.value(),
                early_exaggeration_factor: t.config.early_exaggeration_factor.value(),
                curvature_magnitude: t.config.curvature_magnitude.value(),
                r_max: t.r_max,
                r_rms: t.r_rms,
                metrics: metric_map,
            }
        })
        .collect();

    match serde_json::to_string_pretty(&entries) {
        Ok(json) => {
            std::fs::write(path, json).ok();
        }
        Err(e) => eprintln!("Failed to write Pareto front: {e}"),
    }
}
