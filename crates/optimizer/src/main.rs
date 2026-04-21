use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Serialize;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::data::Dataset;
use crate::evaluate::Evaluator;
use crate::gp::{GpOptimizer, GpState, MultiTrial, ParEgoOptimizer};
use crate::metrics::{AllMetrics, Metric};
use crate::search_space::{SearchSpace, TrialConfig};

mod data;
mod evaluate;
mod gp;
mod metrics;
mod search_space;

#[derive(Parser, Debug, Clone)]
#[command(name = "fitting-optimizer")]
#[command(about = "Hyperparameter search for fitting-curvature")]
struct Args {
    #[arg(long, default_value = "./www/public/data")]
    data_path: String,

    #[arg(long, default_value = "250")]
    n_trials: usize,

    #[arg(long, default_value = "3")]
    n_seeds: usize,

    #[arg(long, default_value = "1000")]
    n_samples: usize,

    /// Output file. All results (all datasets, all curvatures) are appended here.
    #[arg(long, default_value = "results/results.jsonl")]
    output: String,

    #[arg(long)]
    dataset: Option<String>,

    /// Run mode: "random" (default), "bayes", "scan", or "pareto".
    /// random: sample random configs with continuous curvature, compute all metrics.
    /// bayes:  Bayesian optimisation over all 7 hyperparameters (requires --metric).
    ///         Geometry sign is detected automatically unless --geometry is given.
    /// scan:   sweep each parameter individually from a base config (requires --metric).
    /// pareto: qParEGO multi-objective optimisation over 10 objectives (no --metric needed).
    #[arg(long, default_value = "random")]
    mode: String,

    /// Force a specific geometry for --mode bayes and --mode scan.
    /// Values: "hyperbolic" (k<0), "euclidean" (k=0), "spherical" (k>0).
    /// If omitted, the geometry is inferred automatically via curvature detection.
    #[arg(long)]
    geometry: Option<String>,

    /// For --mode random: lower bound of the continuous curvature range.
    #[arg(long, default_value = "-5.0")]
    curvature_min: f64,

    /// For --mode random: upper bound of the continuous curvature range.
    #[arg(long, default_value = "5.0")]
    curvature_max: f64,

    /// For --mode bayes or scan: metric to optimise (e.g. trustworthiness).
    #[arg(long)]
    metric: Option<String>,

    /// For --mode scan: results file to load the best prior config from as a base.
    #[arg(long)]
    scan_from: Option<String>,

    /// For --mode scan: number of evenly-spaced values to sweep per parameter.
    #[arg(long, default_value = "12")]
    scan_steps: usize,

    /// For --mode bayes: results file to warm-start the GP from.
    #[arg(long, default_value = "results/results.jsonl")]
    warm_start: Option<String>,

    /// Number of worker threads. Defaults to the number of logical CPUs.
    #[arg(long)]
    threads: Option<usize>,

    /// Experiment variant controlling which loss weights are optimized vs fixed to 0.
    /// Values: all_off, centering_only, global_only, norm_only, all_free (default).
    /// In all variants: lr, perplexity, early_exaggeration_factor are always optimized;
    /// momentum_main is always fixed at 0.8; scaling_loss_type is always MeanDistance.
    #[arg(long, default_value = "all_free")]
    experiment: String,
}

// ─── Trial result ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct TrialResult {
    dataset_name: String,
    n_samples: usize,
    n_seeds: usize,
    curvature: f64,

    #[serde(skip_serializing_if = "Option::is_none")]
    geometry: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    curvature_magnitude: Option<f64>,

    learning_rate: f64,
    perplexity_ratio: f64,
    momentum_main: f64,
    centering_weight: f64,
    global_loss_weight: f64,
    norm_loss_weight: f64,
    early_exaggeration_factor: f64,

    trustworthiness: Option<f64>,
    trustworthiness_manifold: Option<f64>,
    continuity: Option<f64>,
    continuity_manifold: Option<f64>,
    knn_overlap: Option<f64>,
    knn_overlap_manifold: Option<f64>,
    neighborhood_hit: Option<f64>,
    neighborhood_hit_manifold: Option<f64>,
    normalized_stress: Option<f64>,
    normalized_stress_manifold: Option<f64>,
    shepard_goodness: Option<f64>,
    shepard_goodness_manifold: Option<f64>,
    davies_bouldin_ratio: Option<f64>,
    dunn_index: Option<f64>,
    class_density_measure: Option<f64>,
    cluster_density_measure: Option<f64>,

    time_ms: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    scan_param: Option<String>,
}

impl TrialResult {
    fn new(
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
            time_ms,
            scan_param: None,
        }
    }

    fn with_all_metrics(mut self, m: &AllMetrics) -> Self {
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
        self
    }
}

fn write_result(result: &TrialResult, out_path: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .unwrap();
    let json = serde_json::to_string(result).unwrap();
    writeln!(file, "{}", json).ok();
}

// ─── Experiment variants ──────────────────────────────────────────────────────

fn parse_experiment(name: &str) -> TrialConfig {
    match name {
        "all_off" => TrialConfig::all_off(),
        "centering_only" => TrialConfig::centering_only(),
        "global_only" => TrialConfig::global_only(),
        "norm_only" => TrialConfig::norm_only(),
        "all_free" => TrialConfig::all_free(),
        other => {
            eprintln!(
                "Unknown --experiment '{}'. Valid: all_off, centering_only, global_only, \
                 norm_only, all_free.",
                other
            );
            std::process::exit(1);
        }
    }
}

// ─── Shared evaluation helpers ────────────────────────────────────────────────

fn trial_seed(trial_idx: usize, seed_idx: usize) -> u64 {
    42 + trial_idx as u64 * 100 + seed_idx as u64
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

fn eval_single_metric(
    evaluator: &Evaluator,
    config: &TrialConfig,
    curvature: f64,
    metric: &str,
    n_seeds: usize,
    trial_idx: usize,
    pb_iters: &ProgressBar,
) -> (f64, f64) {
    let values: Vec<f64> = (0..n_seeds)
        .map(|si| {
            evaluator.evaluate_with_metric(
                config,
                curvature,
                metric,
                trial_seed(trial_idx, si),
                pb_iters,
            )
        })
        .collect();
    mean_std(&values)
}

fn eval_all_metrics(
    evaluator: &Evaluator,
    config: &TrialConfig,
    curvature: f64,
    n_seeds: usize,
    trial_idx: usize,
    pb_iters: &ProgressBar,
) -> AllMetrics {
    let samples: Vec<AllMetrics> = (0..n_seeds)
        .map(|si| {
            evaluator.compute_all_metrics(config, curvature, trial_seed(trial_idx, si), pb_iters)
        })
        .collect();
    AllMetrics::mean(&samples)
}

fn make_progress_bar(mp: &MultiProgress, total: u64, template: &str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(template)
            .unwrap()
            .progress_chars("=>-"),
    );
    pb
}

fn parse_metric(name: &str) -> Metric {
    Metric::from_str(name).unwrap_or_else(|| {
        eprintln!(
            "Unknown metric '{}'. Valid options: {}",
            name,
            Metric::valid_names()
        );
        std::process::exit(1);
    })
}

// ─── qParEGO: multi-objective optimisation ────────────────────────────────────

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

fn metrics_to_vec(m: &AllMetrics, metrics: &[Metric]) -> Vec<f64> {
    metrics.iter().map(|metric| metric.value(m)).collect()
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

fn run_pareto(
    dataset_name: &str,
    args: &Args,
    evaluator: Arc<Evaluator>,
    mp: &MultiProgress,
    batch_size: usize,
) {
    let (geometry, curvature_sign) = resolve_geometry(args, &evaluator);
    let optimize_curvature = curvature_sign != 0.0;

    let curvature_mag_min = crate::search_space::DEFAULT_CURVATURE_MAG_MIN;
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

        let results: Vec<(f64, AllMetrics, u64)> = thread::scope(|s| {
            configs
                .iter()
                .enumerate()
                .map(|(i, config)| {
                    let evaluator = &*evaluator;
                    let actual_curvature = curvature_sign * config.curvature_magnitude.value();
                    let trial_idx = lhs_completed + i + 1;
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

        for (config, (actual_curvature, all, elapsed)) in configs.iter().zip(results.iter()) {
            let metric_vec = metrics_to_vec(all, optimizer.metrics.as_slice());
            optimizer.observe(config.clone(), metric_vec);
            lhs_completed += 1;

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

        let results: Vec<(f64, AllMetrics, u64)> = thread::scope(|s| {
            configs
                .iter()
                .enumerate()
                .map(|(i, config)| {
                    let evaluator = &*evaluator;
                    let actual_curvature = curvature_sign * config.curvature_magnitude.value();
                    let trial_idx = lhs_completed + completed + i + 1;
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

        for (config, (actual_curvature, all, elapsed)) in configs.iter().zip(results.iter()) {
            let metric_vec = metrics_to_vec(all, optimizer.metrics.as_slice());
            optimizer.observe(config.clone(), metric_vec);
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

            let front_size = optimizer.pareto_front_indices().len();
            pb.set_prefix(format!("{}", front_size));
            pb.println(format!(
                "pareto '{}' GP {:3}/{} | front={} | {}ms | k={:.3} lr={:.4} perp={:.4}",
                dataset_name,
                completed,
                args.n_trials,
                front_size,
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

    let front = optimizer.pareto_trials();

    let stem = out_path
        .trim_end_matches(".jsonl")
        .trim_end_matches(".json");
    let front_path = format!("{}_pareto_{}_{}.json", stem, dataset_name, geometry);
    write_pareto_front(&front, &optimizer.metrics, args.n_samples, &front_path);
    pb.println(format!("Pareto front written to {}", front_path));
}

// ─── Random search ────────────────────────────────────────────────────────────

fn run_random(dataset_name: &str, args: &Args, evaluator: Arc<Evaluator>, mp: &MultiProgress) {
    let mut rng = fitting_core::synthetic_data::Rng::new(0xdead_beef_cafe_1111);
    let out_path = &args.output;
    let k_lo = args.curvature_min;
    let k_hi = args.curvature_max;
    let sample_space = SearchSpace {
        direction: crate::search_space::OptimizeDirection::Maximize,
        hyper_params: TrialConfig::all_free(),
    };

    let pb = make_progress_bar(
        mp,
        args.n_trials as u64,
        "{spinner:.green} {msg} [{bar:35.cyan/blue}] {pos}/{len} | {wide_msg}",
    );
    pb.set_message(format!("dataset={}", dataset_name));
    let pb_iters = ProgressBar::hidden();

    for trial_idx in 1..=args.n_trials {
        let start = std::time::Instant::now();
        let curvature = rng.uniform() * (k_hi - k_lo) + k_lo;
        let curvature_sign = if curvature < 0.0 {
            -1.0
        } else if curvature > 0.0 {
            1.0
        } else {
            0.0
        };
        let mut config = sample_space.sample(&mut rng);
        config.curvature_magnitude = crate::search_space::ParamSpec::Fixed(curvature.abs());
        let agg = eval_all_metrics(
            &evaluator,
            &config,
            curvature_sign,
            args.n_seeds,
            trial_idx,
            &pb_iters,
        );
        let elapsed = start.elapsed().as_millis() as u64;

        let result = TrialResult::new(
            &config,
            dataset_name,
            args.n_samples,
            args.n_seeds,
            curvature,
            elapsed,
        )
        .with_all_metrics(&agg);
        write_result(&result, out_path);

        pb.set_message(format!(
            "trial {:4} k={:+.2} | db={:.4} trust={:.4} | {}ms",
            trial_idx, curvature, agg.davies_bouldin_ratio, agg.trustworthiness, elapsed
        ));
        pb.inc(1);
    }

    pb.finish_with_message(format!("dataset={} done", dataset_name));
}

// ─── Bayesian optimisation (Algorithm 1, Frazier 2018) ───────────────────────

/// Resolve the target geometry (name + curvature sign) from CLI args or auto-detection.
///
/// `--geometry hyperbolic|spherical|euclidean` forces the choice; omitting it triggers
/// shell-density-profile detection via `evaluator.infer_geometry()`.
fn resolve_geometry(args: &Args, evaluator: &Evaluator) -> (&'static str, f64) {
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
        "Geometry auto-detected: {} (euclidean R²={:.3}, spherical R²={:.3}, hyperbolic R²={:.3})",
        g,
        detection.euclidean.r_squared,
        detection.spherical.r_squared,
        detection.hyperbolic.r_squared,
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
    use crate::search_space::ParamSpec;
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
fn run_bayes(
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
    let curvature_mag_min = crate::search_space::DEFAULT_CURVATURE_MAG_MIN;
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

// ─── Scan ─────────────────────────────────────────────────────────────────────

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
fn run_scan(dataset_name: &str, args: &Args, evaluator: Arc<Evaluator>, mp: &MultiProgress) {
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
        .max(crate::search_space::DEFAULT_CURVATURE_MAG_MIN);
    let curvature_mag_max = args.curvature_max.abs().max(curvature_mag_min);
    use crate::search_space::{
        CEN_MAX, CEN_MIN, EEF_MAX, EEF_MIN, GLW_MAX, GLW_MIN, LR_MAX, LR_MIN, NLW_MAX, NLW_MIN,
        PERP_MAX, PERP_MIN,
    };
    let mut params: Vec<(&str, Vec<f64>)> = Vec::new();
    if hp.learning_rate.is_optimized() {
        params.push(("learning_rate", sweep_values(LR_MIN, LR_MAX, n, true)));
    }
    if hp.perplexity_ratio.is_optimized() {
        params.push((
            "perplexity_ratio",
            sweep_values(PERP_MIN, PERP_MAX, n, true),
        ));
    }
    if hp.centering_weight.is_optimized() {
        params.push(("centering_weight", sweep_values(CEN_MIN, CEN_MAX, n, false)));
    }
    if hp.global_loss_weight.is_optimized() {
        params.push((
            "global_loss_weight",
            sweep_values(GLW_MIN, GLW_MAX, n, false),
        ));
    }
    if hp.norm_loss_weight.is_optimized() {
        params.push(("norm_loss_weight", sweep_values(NLW_MIN, NLW_MAX, n, false)));
    }
    if hp.early_exaggeration_factor.is_optimized() {
        params.push((
            "early_exaggeration_factor",
            sweep_values(EEF_MIN, EEF_MAX, n, false),
        ));
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

// ─── Main ─────────────────────────────────────────────────────────────────────

fn get_dataset_names(dataset_arg: &Option<String>) -> Vec<String> {
    match dataset_arg {
        Some(name) if name == "all" => vec![
            "mnist".to_string(),
            "fashion_mnist".to_string(),
            "pbmc".to_string(),
            "wordnet_mammals".to_string(),
            "sphere".to_string(),
            "antipodal_clusters".to_string(),
            "tree".to_string(),
            "hyperbolic_shells".to_string(),
        ],
        Some(name) => vec![name.clone()],
        None => vec!["mnist".to_string()],
    }
}

fn main() {
    let args = Args::parse();

    if (args.mode == "scan" || args.mode == "bayes") && args.metric.is_none() {
        eprintln!("Error: --metric is required for --mode {}", args.mode);
        std::process::exit(1);
    }
    if args.mode == "pareto" && args.metric.is_some() {
        eprintln!("Note: --metric is ignored for --mode pareto (optimises all objectives).");
    }

    if let Some(parent) = Path::new(&args.output).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }

    let dataset_names = get_dataset_names(&args.dataset);

    match args.mode.as_str() {
        "random" => println!(
            "Starting random search: {} datasets × {} trials, curvature=[{},{}], all metrics, seeds={}",
            dataset_names.len(),
            args.n_trials,
            args.curvature_min,
            args.curvature_max,
            args.n_seeds
        ),
        "scan" => println!(
            "Starting scan: {} datasets × ~{} sweep points, metric={}, geometry={}, seeds={}",
            dataset_names.len(),
            args.scan_steps * 7,
            args.metric.as_deref().unwrap(),
            args.geometry.as_deref().unwrap_or("auto-detect"),
            args.n_seeds
        ),
        "bayes" => println!(
            "Starting Bayesian optimisation: {} datasets × {} trials, metric={}, geometry={}, seeds={}",
            dataset_names.len(),
            args.n_trials,
            args.metric.as_deref().unwrap(),
            args.geometry.as_deref().unwrap_or("auto-detect"),
            args.n_seeds
        ),
        "pareto" => println!(
            "Starting qParEGO multi-objective optimisation: {} datasets × {} trials, 10 objectives, geometry={}, seeds={}",
            dataset_names.len(),
            args.n_trials,
            args.geometry.as_deref().unwrap_or("auto-detect"),
            args.n_seeds
        ),
        other => {
            eprintln!(
                "Unknown --mode '{}'. Use 'random', 'scan', 'bayes', or 'pareto'.",
                other
            );
            std::process::exit(1);
        }
    }
    println!("Output file: {}", args.output);

    let mp = Arc::new(MultiProgress::new());

    // Build unified per-dataset work queue.
    let mut work: VecDeque<(String, Arc<Evaluator>)> = VecDeque::new();
    for dataset_name in &dataset_names {
        println!("Loading dataset: {}...", dataset_name);
        let dp = &args.data_path;
        let n = args.n_samples;
        let result: Result<Dataset, String> = match dataset_name.as_str() {
            "mnist" => Dataset::load_mnist(&format!("{dp}/mnist"), n),
            "fashion_mnist" => Dataset::load_fashion_mnist(&format!("{dp}/fashion-mnist"), n),
            "wordnet_mammals" => Dataset::load_wordnet_mammals(&format!("{dp}/wordnet"), n),
            "pbmc" => Dataset::load_pbmc(&format!("{dp}/pbmc"), n),
            name => Dataset::load_synthetic(name, n, 42),
        };
        let dataset = match result {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error loading dataset '{dataset_name}': {e}");
                std::process::exit(1);
            }
        };
        println!(
            "Loaded {} samples with {} features",
            dataset.n_points, dataset.n_features
        );
        let evaluator = Arc::new(Evaluator::new(dataset));
        work.push_back((dataset_name.clone(), evaluator));
    }

    let n_threads = args
        .threads
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1);

    // The outer pool processes datasets in parallel (up to n_threads datasets at once).
    // For `bayes`, each dataset job additionally spawns n_threads parallel evaluators
    // per batch round — so with a single dataset all cores stay busy.
    // With multiple datasets the outer and inner parallelism combine; on a typical
    // single-dataset run this is always just n_threads total threads.
    let n_outer = n_threads.min(work.len().max(1));
    println!(
        "Using {} thread(s) ({} outer dataset worker(s), batch_size={} for bayes/pareto).",
        n_threads, n_outer, n_threads,
    );

    let queue = Arc::new(Mutex::new(work));
    let mut handles = Vec::new();

    for _ in 0..n_outer {
        let queue = Arc::clone(&queue);
        let args = args.clone();
        let mp = Arc::clone(&mp);
        let h = thread::spawn(move || {
            loop {
                let item = queue.lock().unwrap().pop_front();
                match item {
                    None => break,
                    Some((dataset_name, evaluator)) => match args.mode.as_str() {
                        "scan" => run_scan(&dataset_name, &args, evaluator, &mp),
                        "bayes" => run_bayes(&dataset_name, &args, evaluator, &mp, n_threads),
                        "pareto" => run_pareto(&dataset_name, &args, evaluator, &mp, n_threads),
                        _ => run_random(&dataset_name, &args, evaluator, &mp),
                    },
                }
            }
        });
        handles.push(h);
    }

    for h in handles {
        h.join().expect("optimizer thread panicked");
    }

    println!("\nAll sessions complete.");
}
