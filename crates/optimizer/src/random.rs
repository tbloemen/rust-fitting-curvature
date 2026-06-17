use indicatif::{MultiProgress, ProgressBar};
use std::sync::Arc;

use crate::cli::Args;
use crate::common::{eval_all_metrics, make_progress_bar};
use crate::evaluate::Evaluator;
use crate::search_space::{OptimizeDirection, ParamSpec, SearchSpace, TrialConfig};
use crate::trial_result::{write_result, TrialResult};

// ─── Random search ────────────────────────────────────────────────────────────

pub(crate) fn run_random(
    dataset_name: &str,
    args: &Args,
    evaluator: Arc<Evaluator>,
    mp: &MultiProgress,
) {
    let mut rng = fitting_core::synthetic_data::Rng::new(0xdead_beef_cafe_1111);
    let out_path = &args.output;
    let k_lo = args.curvature_min;
    let k_hi = args.curvature_max;
    let sample_space = SearchSpace {
        direction: OptimizeDirection::Maximize,
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
        config.curvature_magnitude = ParamSpec::Fixed(curvature.abs());
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
