use crate::bayes::resolve_geometry;
use crate::cli::Args;
use crate::common::{make_progress_bar, parse_experiment};
use crate::evaluate::Evaluator;
use crate::gp::{MultiTrial, ParEgoOptimizer};
use crate::metrics::{Direction, Metric, MetricValues, OBJECTIVES};
use crate::resume::{eval_or_reuse_batch, load_prior_evals, BatchOutcome, FreshEval};
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
    let front_path = format!("{stem}_pareto_{dataset_name}_{geometry}.json");

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
    pb.set_message(format!("{geometry} (sign={curvature_sign:+.0})"));
    pb.println(format!(
        "pareto '{dataset_name}' ({geometry}) — LHS init phase: {lhs_total} points, {n_objectives} objectives"
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
                BatchOutcome::Fresh(fresh) => {
                    let FreshEval {
                        all,
                        spread,
                        actual_curvature,
                        elapsed_ms,
                    } = *fresh;
                    let metric_vec = metrics_to_vec(&all, optimizer.metrics.as_slice());
                    optimizer.observe(
                        config.clone(),
                        metric_vec,
                        spread.r_max().unwrap_or(f64::NAN),
                        spread.r_rms().unwrap_or(f64::NAN),
                    );

                    let mut result = TrialResult::new(
                        config,
                        dataset_name,
                        args.n_samples,
                        args.n_seeds,
                        actual_curvature,
                        elapsed_ms,
                    )
                    .with_all_metrics(&all, &spread);
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
    pb.finish_with_message(format!("{geometry} LHS done ({lhs_completed} points)"));

    // ── Phase 2: GP optimisation ─────────────────────────────────────────────
    let pb = make_progress_bar(
        mp,
        args.n_trials as u64,
        "{spinner:.green} [GP]  {msg} [{bar:35.cyan/blue}] {pos}/{len} | front: {prefix} ({eta})",
    );
    pb.set_message(format!("{geometry} (sign={curvature_sign:+.0})"));
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
                BatchOutcome::Fresh(fresh) => {
                    let FreshEval {
                        all,
                        spread,
                        actual_curvature,
                        elapsed_ms,
                    } = *fresh;
                    let metric_vec = metrics_to_vec(&all, optimizer.metrics.as_slice());
                    optimizer.observe(
                        config.clone(),
                        metric_vec,
                        spread.r_max().unwrap_or(f64::NAN),
                        spread.r_rms().unwrap_or(f64::NAN),
                    );

                    let mut result = TrialResult::new(
                        config,
                        dataset_name,
                        args.n_samples,
                        args.n_seeds,
                        actual_curvature,
                        elapsed_ms,
                    )
                    .with_all_metrics(&all, &spread);
                    result.geometry = Some(geometry.to_string());
                    if optimize_curvature {
                        result.curvature_magnitude = Some(config.curvature_magnitude.value());
                    }
                    write_result(&result, out_path);

                    let front_size = optimizer.pareto_front_indices().len();
                    pb.set_prefix(format!("{front_size}"));
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

    pb.finish_with_message(format!("{dataset_name} ({geometry}) done"));

    let front = optimizer.pareto_trials();
    write_pareto_front(&front, &optimizer.metrics, args.n_samples, &front_path);
    pb.println(format!("Pareto front written to {front_path}"));
}

/// The objectives for --mode pareto: six metrics, all measured on the 2D
/// projection.
///
/// The list itself is `fitting_core::metrics::OBJECTIVES`, which is also what
/// `fitting_analysis::objectives::OBJECTIVES` reads — an alignment that used to
/// be maintained by hand across the two crates. Two rules fix its membership,
/// and `test_registry.rs` checks both rather than leaving them to this comment.
///
/// **Projected only.** The manifold (pre-projection, geodesic) variants used to
/// take half the objective budget. What the thesis judges is the 2D
/// visualisation, so the manifold half optimised a surface no reader looks at.
/// Those metrics are still measured and written to the JSONL — nothing about
/// `MetricValues` changed — they just no longer steer the search. `figures/exp4.rs`
/// reads those columns and is what shows whether dropping them was justified.
///
/// **Bounded in `[0, 1]` only.** Of the label-aware, projection-only metrics
/// `neighborhood_hit` and `distance_consistency` qualify — the first is a
/// fraction of neighbours, the second a fraction of points — and they are kept
/// as a pair because one is local and one is global: neighbourhood hit asks
/// only about a point's immediate neighbours, so it cannot separate cleanly
/// separated classes from classes that merely fail to interleave, which is
/// exactly what a comparison against every class centroid does see.
/// `dunn_index`, `davies_bouldin_ratio` and
/// `cluster_density_measure` are ratios, unbounded above, and measured over
/// `results/` their upper tails reach 3.0e10, 2.9e11 and 2e24 respectively —
/// the last mostly from collapsed clusters hitting the `1e-12` radius floor in
/// the formula. Admitting one would break both consumers:
/// `scalarize_subset` min-max normalises per batch, so a single outlier flattens
/// that axis to ~0 for every real trial, and `fitting_analysis::oriented_value`
/// clamps to `[0, 1]`, which would peg 74% of trials at 1.0 on that axis.
///
/// Keeping every objective naturally bounded is what lets both of those stay as
/// they are, with no transform and no estimated bounds (Karl et al., *MOHPO — An
/// Overview*, §3.3.3 and §4: normalise to `[0, 1]`, which is "fairly simple"
/// for metrics with known limits and needs estimation otherwise). It is also
/// what the DR-quality literature does — Espadoto et al. and Telea et al. use
/// metrics that all range in `[0, 1]`, and reach for bounded class-separation
/// measures rather than repairing unbounded ones.
pub(crate) fn default_pareto_metrics() -> Vec<Metric> {
    OBJECTIVES.to_vec()
}

/// Build the objective vector fed to the optimizer. A diverged embedding (e.g. an
/// unbounded Euclidean run that blew up to inf/NaN) yields non-finite metric
/// values; substitute each metric's worst finite value so the trial is scored as
/// bad rather than poisoning the GP normalisation or panicking the Pareto sorts.
/// The raw (possibly non-finite) values are still recorded in the JSONL via
/// `with_all_metrics`, so diverged trials remain visible in the results.
pub(crate) fn metrics_to_vec(m: &MetricValues, metrics: &[Metric]) -> Vec<f64> {
    metrics
        .iter()
        .map(|metric| {
            m.get(*metric).unwrap_or(match metric.direction() {
                // The maximised metrics here are bounded below by 0 (0 = degenerate);
                // normalized_stress is minimised and bounded above by 1 (1 = worst).
                Direction::Maximize => 0.0,
                Direction::Minimize => 1.0,
            })
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
