use indicatif::ProgressBar;
use serde::Deserialize;
use std::thread;

use crate::common::eval_all_metrics;
use crate::evaluate::Evaluator;
use crate::metrics::{AllMetrics, Metric};
use crate::pareto::metrics_to_vec;
use crate::search_space::TrialConfig;

// ─── Resume support ───────────────────────────────────────────────────────────

/// A previously-recorded trial result, reduced to what the optimizer needs to
/// re-`observe` it: the objective vector (in `default_pareto_metrics()` order)
/// plus the diagnostic radii. Configs are *not* stored — on resume they are
/// re-derived deterministically by replaying `suggest_batch`, so only the
/// evaluation outcome has to come from disk.
#[derive(Clone)]
pub(crate) struct PriorEval {
    pub(crate) metric_vec: Vec<f64>,
    pub(crate) r_max: f64,
    pub(crate) r_rms: f64,
}

/// Subset of a `TrialResult` JSONL line needed to reconstruct the objective
/// vector. Each field is `Option` because a diverged trial serialises its
/// non-finite metrics as JSON `null`; those are mapped back to the same
/// worst-case substitute `metrics_to_vec` applies, so the replayed observation
/// is identical to the original.
#[derive(Deserialize)]
struct PriorTrialRecord {
    trustworthiness: Option<f64>,
    trustworthiness_manifold: Option<f64>,
    continuity: Option<f64>,
    continuity_manifold: Option<f64>,
    normalized_stress: Option<f64>,
    normalized_stress_manifold: Option<f64>,
    shepard_goodness: Option<f64>,
    shepard_goodness_manifold: Option<f64>,
    neighborhood_hit: Option<f64>,
    neighborhood_hit_manifold: Option<f64>,
    r_max: Option<f64>,
    r_rms: Option<f64>,
}

/// Load prior pareto trials from `path`, in file order (which is exactly the
/// order they were proposed and observed). The returned vec lets a resumed run
/// skip re-evaluating them. A missing/empty file yields an empty vec (fresh
/// start). A torn final line (process killed mid-write) is dropped; a corrupt
/// interior line aborts, since silently skipping it would desync the positional
/// replay.
pub(crate) fn load_prior_evals(path: &str, metrics: &[Metric]) -> Vec<PriorEval> {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let lines: Vec<&str> = contents
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let n = lines.len();
    let mut out = Vec::with_capacity(n);
    for (i, line) in lines.iter().enumerate() {
        match serde_json::from_str::<PriorTrialRecord>(line) {
            Ok(rec) => {
                let nan = f64::NAN;
                // Only the 10 pareto objectives + r_max/r_rms matter to the
                // optimizer; the other AllMetrics fields are unused here.
                let all = AllMetrics {
                    trustworthiness: rec.trustworthiness.unwrap_or(nan),
                    trustworthiness_manifold: rec.trustworthiness_manifold.unwrap_or(nan),
                    continuity: rec.continuity.unwrap_or(nan),
                    continuity_manifold: rec.continuity_manifold.unwrap_or(nan),
                    knn_overlap: 0.0,
                    knn_overlap_manifold: 0.0,
                    neighborhood_hit: rec.neighborhood_hit.unwrap_or(nan),
                    neighborhood_hit_manifold: rec.neighborhood_hit_manifold.unwrap_or(nan),
                    normalized_stress: rec.normalized_stress.unwrap_or(nan),
                    normalized_stress_manifold: rec.normalized_stress_manifold.unwrap_or(nan),
                    shepard_goodness: rec.shepard_goodness.unwrap_or(nan),
                    shepard_goodness_manifold: rec.shepard_goodness_manifold.unwrap_or(nan),
                    davies_bouldin_ratio: 0.0,
                    dunn_index: 0.0,
                    class_density_measure: 0.0,
                    cluster_density_measure: 0.0,
                    r_max: rec.r_max.unwrap_or(nan),
                    r_rms: rec.r_rms.unwrap_or(nan),
                };
                out.push(PriorEval {
                    metric_vec: metrics_to_vec(&all, metrics),
                    r_max: all.r_max,
                    r_rms: all.r_rms,
                });
            }
            Err(e) => {
                if i + 1 == n {
                    eprintln!("resume: dropping torn final line in {path}: {e}");
                } else {
                    eprintln!("resume: corrupt record at {path}:{}: {e}", i + 1);
                    std::process::exit(1);
                }
            }
        }
    }
    out
}

/// Outcome of evaluating (or reusing) one config within a batch.
pub(crate) enum BatchOutcome {
    /// Result replayed from a resume checkpoint — no embedding was run.
    Reused {
        metric_vec: Vec<f64>,
        r_max: f64,
        r_rms: f64,
    },
    /// Freshly evaluated — carries everything needed to log a JSONL line.
    Fresh {
        all: AllMetrics,
        actual_curvature: f64,
        elapsed_ms: u64,
    },
}

/// Evaluate a batch of configs in parallel, reusing recorded results for any
/// whose global trial index is still covered by `prior` (the resume set).
/// `base_global` is the number of trials already observed before this batch.
/// Because `prior` is a prefix of the full trial sequence, the reused configs
/// are exactly the prefix of this batch; the remainder are evaluated fresh
/// (one thread each), keeping `trial_idx` — and thus the per-trial seeds —
/// identical to a single uninterrupted run.
pub(crate) fn eval_or_reuse_batch(
    configs: &[TrialConfig],
    base_global: usize,
    prior: &[PriorEval],
    evaluator: &Evaluator,
    curvature_sign: f64,
    n_seeds: usize,
) -> Vec<BatchOutcome> {
    let reused = prior.len().saturating_sub(base_global).min(configs.len());

    let fresh_results: Vec<(f64, AllMetrics, u64)> = thread::scope(|s| {
        configs[reused..]
            .iter()
            .enumerate()
            .map(|(j, config)| {
                let actual_curvature = curvature_sign * config.curvature_magnitude.value();
                let trial_idx = base_global + reused + j + 1;
                s.spawn(move || {
                    let pb_iters = ProgressBar::hidden();
                    let start = std::time::Instant::now();
                    let all = eval_all_metrics(
                        evaluator,
                        config,
                        curvature_sign,
                        n_seeds,
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

    let mut outcomes = Vec::with_capacity(configs.len());
    for i in 0..reused {
        let p = &prior[base_global + i];
        outcomes.push(BatchOutcome::Reused {
            metric_vec: p.metric_vec.clone(),
            r_max: p.r_max,
            r_rms: p.r_rms,
        });
    }
    for (actual_curvature, all, elapsed_ms) in fresh_results {
        outcomes.push(BatchOutcome::Fresh {
            all,
            actual_curvature,
            elapsed_ms,
        });
    }
    outcomes
}
