use indicatif::ProgressBar;
use std::thread;

use crate::common::eval_all_metrics;
use crate::evaluate::Evaluator;
use crate::metrics::{Metric, MetricValues};
use crate::pareto::metrics_to_vec;
use crate::search_space::TrialConfig;
use fitting_core::spread::SpreadDiagnostics;

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

/// The two flattened blocks a results line carries, which is all `--resume`
/// reads off it. Everything else on the line — hyperparameters, dataset,
/// timings — is re-derived by replaying `suggest_batch`, so it is ignored here.
#[derive(serde::Deserialize)]
struct PriorLine {
    #[serde(flatten)]
    metrics: MetricValues,
    #[serde(flatten)]
    spread: SpreadDiagnostics,
}

/// Load prior pareto trials from `path`, in file order (which is exactly the
/// order they were proposed and observed). The returned vec lets a resumed run
/// skip re-evaluating them. A missing/empty file yields an empty vec (fresh
/// start). A torn final line (process killed mid-write) is dropped; a corrupt
/// interior line aborts, since silently skipping it would desync the positional
/// replay.
pub(crate) fn load_prior_evals(path: &str, metrics: &[Metric]) -> Vec<PriorEval> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let lines: Vec<&str> = contents
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let n = lines.len();
    let mut out = Vec::with_capacity(n);
    for (i, line) in lines.iter().enumerate() {
        // The whole metric block, read by name. This replaced a
        // `PriorTrialRecord` that listed the pareto objectives by hand and
        // warned in its doc comment that a field missing from it would
        // deserialise as `None`, replay as the worst-case substitute, and
        // silently give a resumed run a different GP — no error. There is no
        // longer a second list to fall out of step with the first.
        match serde_json::from_str::<PriorLine>(line) {
            Ok(prior) => {
                out.push(PriorEval {
                    metric_vec: metrics_to_vec(&prior.metrics, metrics),
                    r_max: prior.spread.r_max().unwrap_or(f64::NAN),
                    r_rms: prior.spread.r_rms().unwrap_or(f64::NAN),
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
    ///
    /// Boxed because the readings are 256 bytes against `Reused`'s 40, and this
    /// enum is built one per config per batch: unboxed, every replayed trial
    /// would carry the fresh variant's footprint for nothing.
    Fresh(Box<FreshEval>),
}

/// The outcome of actually running an embedding.
pub(crate) struct FreshEval {
    pub(crate) all: MetricValues,
    pub(crate) spread: SpreadDiagnostics,
    pub(crate) actual_curvature: f64,
    pub(crate) elapsed_ms: u64,
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

    let fresh_results: Vec<(f64, MetricValues, SpreadDiagnostics, u64)> = thread::scope(|s| {
        configs[reused..]
            .iter()
            .enumerate()
            .map(|(j, config)| {
                let actual_curvature = curvature_sign * config.curvature_magnitude.value();
                let trial_idx = base_global + reused + j + 1;
                s.spawn(move || {
                    let pb_iters = ProgressBar::hidden();
                    let start = std::time::Instant::now();
                    let (all, spread) = eval_all_metrics(
                        evaluator,
                        config,
                        curvature_sign,
                        n_seeds,
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
    for (actual_curvature, all, spread, elapsed_ms) in fresh_results {
        outcomes.push(BatchOutcome::Fresh(Box::new(FreshEval {
            all,
            spread,
            actual_curvature,
            elapsed_ms,
        })));
    }
    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Direction;
    use crate::pareto::default_pareto_metrics;
    use fitting_core::cast::count_to_f64;
    use std::cmp::Ordering;

    /// Every objective the optimizer searches must survive a round trip
    /// through the JSONL, or `--resume` replays it as the worst-case
    /// substitute instead of its recorded value — silently, with no error.
    ///
    /// This used to guard a hand-written `PriorTrialRecord` that listed the
    /// objectives a second time; it now passes because there is only one list,
    /// which is the point. It is kept because it is behavioural rather than
    /// structural — it writes a line, loads it back, and requires the replayed
    /// vector to be the values written — so it still catches a wire name that
    /// stops round-tripping for any other reason.
    #[test]
    fn prior_record_covers_every_pareto_objective() {
        let metrics = default_pareto_metrics();

        // Distinct, in-range, and never equal to a worst-case substitute
        // (0.0 for maximised objectives, 1.0 for minimised ones).
        let values: Vec<f64> = (0..metrics.len())
            .map(|i| 0.11 + 0.07 * count_to_f64(i))
            .collect();

        let fields: Vec<String> = metrics
            .iter()
            .zip(&values)
            .map(|(m, v)| format!("\"{}\":{}", m.name(), v))
            .collect();
        let line = format!("{{{},\"r_max\":1.0,\"r_rms\":2.0}}", fields.join(","));

        let dir = std::env::temp_dir().join(format!("resume_objectives_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prior.jsonl");
        std::fs::write(&path, format!("{line}\n")).unwrap();

        let prior = load_prior_evals(path.to_str().unwrap(), &metrics);
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(prior.len(), 1, "the recorded trial should replay");
        for ((got, want), metric) in prior[0].metric_vec.iter().zip(&values).zip(&metrics) {
            assert!(
                (got - want).abs() < 1e-12,
                "objective `{}` replayed as {got} instead of {want}",
                metric.name()
            );
        }
        assert_eq!(prior[0].r_max.partial_cmp(&1.0), Some(Ordering::Equal));
        assert_eq!(prior[0].r_rms.partial_cmp(&2.0), Some(Ordering::Equal));
    }

    /// A line from `results/` as it is actually shaped: retired metric columns
    /// that no longer parse, no `r_gyration`, a `null` from a diverged trial,
    /// and non-numeric columns alongside. All of it has to load, because
    /// `--resume` reads files written by builds that predate the registry.
    #[test]
    fn a_pre_registry_results_line_still_replays() {
        let metrics = default_pareto_metrics();
        let line = r#"{"dataset_name":"antipodal_clusters","geometry":"euclidean","n_samples":1000,
            "knn_overlap":0.4,"class_density_measure":12.5,
            "trustworthiness":0.9,"continuity":0.8,"normalized_stress":0.3,
            "shepard_goodness":null,"neighborhood_hit":0.7,
            "r_max":1.0,"r_rms":2.0,"time_ms":42}"#
            .replace('\n', "");

        let dir = std::env::temp_dir().join(format!("resume_legacy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prior.jsonl");
        std::fs::write(&path, format!("{line}\n")).unwrap();

        let prior = load_prior_evals(path.to_str().unwrap(), &metrics);
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(prior.len(), 1, "a pre-registry line must still load");
        // The columns the line actually carries a number for. Everything else —
        // `shepard_goodness`, explicitly `null` from a diverged trial, and any
        // objective added after these files were written — replays as the
        // metric's worst-case substitute, which is what keeps a partial line
        // from scoring well. Listed by name rather than positionally, so a new
        // objective does not have to be added here to keep the test honest.
        let carried = [
            ("trustworthiness", 0.9),
            ("continuity", 0.8),
            ("normalized_stress", 0.3),
            ("neighborhood_hit", 0.7),
        ];
        assert_eq!(prior[0].metric_vec.len(), metrics.len());
        for (metric, got) in metrics.iter().zip(prior[0].metric_vec.iter().copied()) {
            let want = carried
                .iter()
                .find(|(name, _)| *name == metric.name())
                .map_or(
                    match metric.direction() {
                        Direction::Maximize => 0.0,
                        Direction::Minimize => 1.0,
                    },
                    |&(_, v)| v,
                );
            assert_eq!(
                got.partial_cmp(&want),
                Some(Ordering::Equal),
                "{} replayed as {got}",
                metric.name()
            );
        }
        assert_eq!(prior[0].r_max.partial_cmp(&1.0), Some(Ordering::Equal));
    }
}
