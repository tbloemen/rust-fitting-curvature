//! Experiment 1: does the embedding geometry matching the data's intrinsic
//! curvature beat the Euclidean baseline?
//!
//! Joins two things that were previously not comparable:
//!
//! - the **Pareto front** each `(dataset, N, geometry)` sweep produced, scored
//!   by the R2 indicator under every preference region, exactly as
//!   `r2 stats` does;
//! - the **Wilson fit** of the same dataset, reconstructed into coordinates and
//!   scored on the same ten objectives by the optimizer's `--mode wilson-mds`.
//!
//! Because the Wilson point lives in the same oriented objective space as every
//! trial, the whole existing toolkit applies to it unchanged, and the table can
//! answer two nested questions instead of one:
//!
//! 1. Does curvature-matched t-SNE beat Euclidean t-SNE? → `delta_r2`, formed
//!    as `R2(euclidean) − R2(geometry)` so positive favours the curved arm
//!    (the same direction as `aggregate.rs`, because R2 is a cost).
//! 2. Does *learned* curvature-aware t-SNE beat the *closed-form*
//!    constant-curvature MDS the detector hands you for free? →
//!    `wilson.r2_gain_over_front`.
//!
//! ## Three ways to score one point, and which to trust
//!
//! `wilson.r2` is the R2 indicator of the **singleton** `{w}`. It is a legal
//! argument to `r2()` and directly printable beside the front's, but R2 is
//! monotone under set inclusion — a one-point set can never beat a front that
//! contains a comparable point. It is an upper bound on what the closed-form
//! solution offers and must never be read as "Wilson lost".
//!
//! `wilson.r2_gain_over_front` is `R2(front) − R2(front ∪ {w})`, the fair
//! number: it asks whether the closed-form solution reaches anywhere the sweep
//! did not. Exactly `0` means the front already covers it, and it cannot be
//! negative — adding a point can only lower a cost. **This is the headline.**
//!
//! `wilson.eps_*` and `wilson.dominated_by_front` are the parameter-free
//! cross-check, carrying none of the preference model's choices.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};

use fitting_analysis::cell::{discover_cells, truth_of};
use fitting_analysis::objectives::{oriented_matrix, oriented_row, N_OBJECTIVES};
use fitting_analysis::r2::{cell_summary, front_utilities, r2, Weights};
use fitting_analysis::stats;
use fitting_analysis::{
    epsilon_pair, load_jsonl, pareto_front_mask, trial_records, write_jsonl, Error, Result,
    TrialRecord,
};

/// The loss-weight setting Experiment 1 reads. `all_off` zeroes every auxiliary
/// loss, recovering plain KL-divergence t-SNE, so geometry is the only thing
/// varying between the cells being compared — which is the whole question.
/// `5results.typ` also reuses these same fronts for Experiment 5.
const DEFAULT_SETTING: &str = "all_off";

#[derive(Parser, Debug)]
#[command(
    about = "Experiment 1: matched-geometry vs the Euclidean baseline, with the Wilson fit scored on the same objectives"
)]
struct Args {
    /// Directory of sweep results (`<setting>_<dataset>[_n5000]_<geometry>.jsonl`).
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// JSONL from `optimizer --mode wilson-mds`: the reconstructed Wilson fits
    /// scored on the ten Pareto objectives.
    #[arg(long, default_value = "wilson_mds.jsonl")]
    wilson_mds: PathBuf,

    /// Loss-weight setting to read.
    #[arg(long, default_value = DEFAULT_SETTING)]
    setting: String,

    /// Let a sample size with no Wilson run of its own reuse the fit from
    /// another N. The Wilson search is `O(n³)` per candidate radius, so it is
    /// run at N=1000 and the N=5000 rows reuse it; `wilson_n` records which N
    /// the fit actually came from. Without this flag those rows get `null`.
    ///
    /// Only `rho` and `kappa` carry across — both are dimensionless by
    /// construction and describe the generator. The fields that compare the
    /// Wilson point *against this row's front* stay `null`, because that front
    /// comes from a different sample than a carried-over reconstruction.
    #[arg(long)]
    wilson_fallback: bool,

    #[arg(long, default_value = "exp1_geometry_match.jsonl")]
    out: PathBuf,
}

// ─── Input: the wilson-mds table ─────────────────────────────────────────────

/// One arm of `optimizer --mode wilson-mds`.
///
/// `metrics` deserialises straight into a [`TrialRecord`]: the mode writes the
/// same metric names a trial does, and every `TrialRecord` field is `Option`
/// with a serde default, so the Wilson point reaches objective space through
/// `oriented_row` — the same map every trial goes through, by construction
/// rather than by a parallel implementation.
#[derive(Debug, Clone, Deserialize)]
struct WilsonMds {
    dataset: String,
    n_samples: usize,
    geometry: String,
    wilson_residual_normalised: f64,
    wilson_at_upper_bound: bool,
    kappa: f64,
    metrics: TrialRecord,
}

// ─── Output ──────────────────────────────────────────────────────────────────

/// What the Wilson fit contributes to one row.
///
/// The fields split in two by what they are a property *of*, which decides
/// whether `--wilson-fallback` may carry them across sample sizes:
///
/// - `rho` and `kappa` describe **the generator**. Both are
///   dimensionless by construction — `ρ` is gauged by `n · d_max²` precisely so
///   it compares across sample sizes, and `κ = |K|·R_rms²` likewise — and each
///   is a statistic of the fit or the reconstruction alone, converging as the
///   sample grows. Reusing them from another `N` is defensible, and `wilson_n`
///   records that it happened.
/// - everything below them is a **comparison against this row's front**, and a
///   carried-over reconstruction was built from a different sample than that
///   front. Scoring a 1000-point reconstruction against a 5000-point front
///   compares nothing, so these are `None` whenever `wilson_n != n`.
#[derive(Debug, Clone, Serialize)]
struct WilsonSummary {
    /// Which sample size the fit was run at. Differs from the row's `n` when
    /// `--wilson-fallback` is in play, and the caption has to say so.
    wilson_n: usize,
    /// `WilsonFit::residual_normalised` — the `ρ` of the thesis table, gauged
    /// by `n · d_max²`. Lower is better. Also the eigenvalue mass the
    /// reconstruction scored here discarded.
    rho: f64,
    /// Whether `r*` pinned at the flat-ward edge of its search window, making
    /// the radius a bound rather than a measurement.
    pinned: bool,
    /// `|K| · R_rms²` of the reconstruction — the same gauge
    /// `TrialRecord::kappa()` uses, so this is directly comparable to
    /// `kappa_median` and the two may be differenced.
    kappa: f64,
    /// R2 of the singleton `{w}`. See the module docs: monotone under set
    /// inclusion, so not a fair comparison against `r2`.
    r2: Option<BTreeMap<String, f64>>,
    /// `R2(front) − R2(front ∪ {w})`, per region. Zero means the front already
    /// covers the closed-form solution. Never negative.
    r2_gain_over_front: Option<BTreeMap<String, f64>>,
    /// `I_ε+({w}, front)`; `≤ 0` iff the Wilson point covers the whole front.
    eps_wilson_vs_front: Option<f64>,
    /// `I_ε+(front, {w})`; `≤ 0` iff the front covers the Wilson point.
    eps_front_vs_wilson: Option<f64>,
    /// Whether some front point dominates the Wilson point outright.
    dominated_by_front: Option<bool>,
}

/// One `(dataset, N, geometry)` row.
#[derive(Debug, Clone, Serialize)]
struct Exp1Row {
    dataset: String,
    n: usize,
    setting: String,
    geometry: String,
    /// The geometry the dataset is built to have (`cell::SYNTH_TRUTH`).
    truth: &'static str,
    /// Whether this row's embedding geometry is the matching one.
    matched: bool,

    n_trials: usize,
    n_front: usize,

    /// R2 indicator per preference region. Smaller is better.
    r2: BTreeMap<String, f64>,
    /// The same for this dataset's Euclidean cell — the baseline this row is
    /// referenced to. Present on the Euclidean row too, where it equals `r2`.
    r2_euclidean: BTreeMap<String, f64>,
    /// `R2(euclidean) − R2(geometry)`, so positive favours this row's geometry.
    /// `null` on the Euclidean row, which is the baseline rather than a
    /// comparison against it.
    delta_r2: Option<BTreeMap<String, f64>>,

    /// Median `|K|·R_rms²` over the front. `0` for Euclidean by construction.
    kappa_median: Option<f64>,
    kappa_q25: Option<f64>,
    kappa_q75: Option<f64>,

    /// `null` when no Wilson run covers this dataset at this N.
    wilson: Option<WilsonSummary>,
}

// ─── The join ────────────────────────────────────────────────────────────────

/// Everything one cell contributes before the Euclidean baseline is known.
struct Scored {
    n_trials: usize,
    n_front: usize,
    r2: BTreeMap<String, f64>,
    /// The front, in oriented objective space — needed a second time to score
    /// the Wilson point against it.
    front: Vec<[f64; N_OBJECTIVES]>,
    kappa: Vec<f64>,
}

/// R2 of an arbitrary point set under every region, using one utilities pass.
fn r2_by_region(front: &[[f64; N_OBJECTIVES]], weights: &Weights) -> BTreeMap<String, f64> {
    let utilities = front_utilities(front, &weights.vectors);
    weights
        .regions
        .iter()
        .map(|region| (region.name.clone(), r2(&utilities, region)))
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    let wilson: Vec<WilsonMds> = load_jsonl(&args.wilson_mds)?;
    // (dataset, n, geometry) → the arm, so a row can find its own fit.
    let mut wilson_by_key: BTreeMap<(String, usize, String), WilsonMds> = BTreeMap::new();
    for w in wilson {
        wilson_by_key.insert((w.dataset.clone(), w.n_samples, w.geometry.clone()), w);
    }

    let weights = Weights::new();

    // Pass 1: score every cell of interest. Cells are walked in
    // `discover_cells` order (sorted by stem) so the output is byte-identical
    // across runs, as every other table in this crate is.
    let mut scored: BTreeMap<(String, usize, String), Scored> = BTreeMap::new();
    let mut order: Vec<(String, usize, String)> = Vec::new();
    for cf in &cells {
        if cf.cell.setting != args.setting || truth_of(&cf.cell.dataset).is_none() {
            continue;
        }
        let records = trial_records(&cf.path)?;
        let summary = cell_summary(&records, &weights);

        let all = oriented_matrix(&records);
        let front: Vec<[f64; N_OBJECTIVES]> = summary.front.iter().map(|&i| all[i]).collect();

        let kappa: Vec<f64> = summary
            .front
            .iter()
            .filter_map(|&i| records[i].kappa())
            .filter(|k| k.is_finite())
            .collect();

        let key = (cf.cell.dataset.clone(), cf.cell.n, cf.cell.geometry.clone());
        order.push(key.clone());
        scored.insert(
            key,
            Scored {
                n_trials: summary.n_trials,
                n_front: summary.n_front,
                r2: summary.r2.clone(),
                front,
                kappa,
            },
        );
    }

    // Pass 2: reference each row to its dataset's Euclidean cell, which pass 1
    // may not have reached yet.
    let mut rows = Vec::with_capacity(order.len());
    for key in &order {
        let (dataset, n, geometry) = key;
        let cell = &scored[key];
        let truth = truth_of(dataset).expect("filtered above");

        let euclidean_key = (dataset.clone(), *n, "euclidean".to_string());
        let Some(euclidean) = scored.get(&euclidean_key) else {
            // A dataset with no Euclidean cell has no baseline to compare
            // against, which makes the whole question unanswerable for it.
            return Err(Error::NoBaselineCell {
                baseline: "euclidean",
                n: *n,
                geometry: geometry.clone(),
                dataset: dataset.clone(),
            });
        };

        let delta_r2 = (geometry != "euclidean").then(|| {
            cell.r2
                .iter()
                .map(|(region, value)| {
                    let base = euclidean.r2.get(region).copied().unwrap_or(f64::NAN);
                    (region.clone(), base - value)
                })
                .collect()
        });

        let wilson = wilson_summary(
            &wilson_by_key,
            dataset,
            *n,
            geometry,
            args.wilson_fallback,
            cell,
            &weights,
        );

        rows.push(Exp1Row {
            dataset: dataset.clone(),
            n: *n,
            setting: args.setting.clone(),
            geometry: geometry.clone(),
            truth,
            matched: truth == geometry,
            n_trials: cell.n_trials,
            n_front: cell.n_front,
            r2: cell.r2.clone(),
            r2_euclidean: euclidean.r2.clone(),
            delta_r2,
            kappa_median: stats::quantile(&cell.kappa, 0.5),
            kappa_q25: stats::quantile(&cell.kappa, 0.25),
            kappa_q75: stats::quantile(&cell.kappa, 0.75),
            wilson,
        });
    }

    write_jsonl(&args.out, &rows)
}

/// Score this row's Wilson arm against this row's front.
#[allow(clippy::too_many_arguments)]
fn wilson_summary(
    wilson_by_key: &BTreeMap<(String, usize, String), WilsonMds>,
    dataset: &str,
    n: usize,
    geometry: &str,
    fallback: bool,
    cell: &Scored,
    weights: &Weights,
) -> Option<WilsonSummary> {
    let exact = wilson_by_key.get(&(dataset.to_string(), n, geometry.to_string()));
    let w = match (exact, fallback) {
        (Some(w), _) => w,
        // No run at this N: fall back to whichever N was run, if allowed. The
        // fit characterises the generator, which does not change with sample
        // size — only the quality of its estimate does. `wilson_n` keeps that
        // substitution visible rather than silent.
        (None, true) => wilson_by_key
            .iter()
            .find(|((d, _, g), _)| d == dataset && g == geometry)
            .map(|(_, w)| w)?,
        (None, false) => return None,
    };

    // Scoring a reconstruction of one sample against a front built from a
    // different one compares nothing, so the front-comparison half is only
    // filled in when the Wilson run and the sweep saw the same N. The fit's own
    // statistics carry regardless. See `WilsonSummary`.
    let comparable = w.n_samples == n;

    let mut summary = WilsonSummary {
        wilson_n: w.n_samples,
        rho: w.wilson_residual_normalised,
        pinned: w.wilson_at_upper_bound,
        kappa: w.kappa,
        r2: None,
        r2_gain_over_front: None,
        eps_wilson_vs_front: None,
        eps_front_vs_wilson: None,
        dominated_by_front: None,
    };
    if !comparable {
        return Some(summary);
    }

    let point = oriented_row(&w.metrics);
    let singleton = [point];

    // `front ∪ {w}` — how much the closed-form solution adds to a front that
    // took a thousand trials to build.
    let mut augmented = cell.front.clone();
    augmented.push(point);

    let base = r2_by_region(&cell.front, weights);
    let with_wilson = r2_by_region(&augmented, weights);

    let pair = epsilon_pair(&singleton, &cell.front);

    // The augmented front's mask says directly whether the front dominates the
    // newcomer: if the Wilson point survives the sort, nothing dominates it.
    let mask = pareto_front_mask(&augmented);

    summary.r2 = Some(r2_by_region(&singleton, weights));
    summary.r2_gain_over_front = Some(
        base.iter()
            .map(|(region, before)| {
                let after = with_wilson.get(region).copied().unwrap_or(f64::NAN);
                (region.clone(), before - after)
            })
            .collect(),
    );
    summary.eps_wilson_vs_front = pair.as_ref().map(|p| p.setting_vs_baseline);
    summary.eps_front_vs_wilson = pair.as_ref().map(|p| p.baseline_vs_setting);
    summary.dominated_by_front = Some(!mask[augmented.len() - 1]);
    Some(summary)
}
