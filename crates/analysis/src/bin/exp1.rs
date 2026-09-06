//! Experiment 1: does the embedding geometry matching the data's intrinsic
//! curvature beat the Euclidean baseline?
//!
//! Joins two things that were previously not comparable:
//!
//! - the **Pareto front** each `(dataset, N, geometry)` sweep produced, scored
//!   by the R2 indicator under every preference region, exactly as
//!   `r2 stats` does;
//! - the **Wilson fit** of the same dataset, as `optimizer --mode detect`
//!   reports it: the signature residual and the curvature each of the three
//!   constant-curvature arms infers.
//!
//! The table reports the two side by side:
//!
//! 1. Does curvature-matched t-SNE beat Euclidean t-SNE? → `delta_r2`, formed
//!    as `R2(euclidean) − R2(geometry)` so positive favours the curved arm
//!    (the same direction as `aggregate.rs`, because R2 is a cost).
//! 2. What the *closed-form* constant-curvature MDS the detector hands you for
//!    free inferred about the same dataset → the `wilson` block.
//!
//! The two are set beside each other, not differenced. Every number that scored
//! the Wilson point *against* the front — its singleton R2, the
//! `R2(front) − R2(front ∪ {w})` gain, the ε-indicator pair, the dominance flag
//! — has been removed, so the `wilson` block is now the fit's own statistics
//! and nothing else. The one column that still invites a comparison is `kappa`
//! against `kappa_median`: both are `|K|·R_rms²` on the same gauge, which is
//! what makes them differenceable.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};

use fitting_analysis::cell::{discover_cells, truth_of};
use fitting_analysis::r2::{cell_summary, Weights};
use fitting_analysis::stats;
use fitting_analysis::{load_jsonl, trial_records, write_jsonl, Error, Result};

/// The loss-weight setting Experiment 1 reads. `all_off` zeroes every auxiliary
/// loss, recovering plain KL-divergence t-SNE, so geometry is the only thing
/// varying between the cells being compared — which is the whole question.
/// `5results.typ` also reuses these same fronts for Experiment 5.
const DEFAULT_SETTING: &str = "all_off";

#[derive(Parser, Debug)]
#[command(
    about = "Experiment 1: matched-geometry vs the Euclidean baseline, alongside the Wilson fit of the same dataset"
)]
struct Args {
    /// Directory of sweep results (`<setting>_<dataset>[_n5000]_<geometry>.jsonl`).
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// JSONL from `optimizer --mode detect`: one line per dataset carrying all
    /// three Wilson arms. Only `rho`, the pin flag and `kappa` are read.
    #[arg(long, default_value = "results/kappa_data.jsonl")]
    kappa_data: PathBuf,

    /// Loss-weight setting to read.
    #[arg(long, default_value = DEFAULT_SETTING)]
    setting: String,

    /// Let a sample size with no detect run of its own reuse the fit from
    /// another N. The Wilson search is `O(n³)` per candidate radius, so it is
    /// run at N=1000 and the N=5000 rows reuse it; `wilson_n` records which N
    /// the fit actually came from. Without this flag those rows get `null`.
    ///
    /// Everything in the `wilson` block carries across: it is all dimensionless
    /// by construction and describes the generator, not the sample. No field is
    /// suppressed on a carried-over row — `wilson_n` is what marks it.
    #[arg(long)]
    wilson_fallback: bool,

    #[arg(long, default_value = "results/exp1_geometry_match.jsonl")]
    out: PathBuf,
}

// ─── Input: the detection table ──────────────────────────────────────────────

/// One dataset's curvature-detection record from `optimizer --mode detect`,
/// reduced to the fields this table reports.
///
/// The detect record is **wide**: one line per dataset carrying all three
/// Wilson arms side by side, where this table's rows are long — one per
/// (dataset, geometry). [`DetectRecord::arm`] is the transpose.
///
/// The euclidean arm has no radius to pin and no reconstruction to gauge (its
/// `K` is `0` exactly), which is why it has two fields where the curved arms
/// have three. serde ignores the many diagnostic fields the record also
/// carries — the δ(k) block, the radii, the `r_rms` gauges — so they stay
/// available in the JSONL for anything else that wants them.
#[derive(Debug, Clone, Deserialize)]
struct DetectRecord {
    dataset: String,
    n_samples: usize,

    sph_residual_normalised: f64,
    sph_at_upper_bound: bool,
    sph_kappa: f64,

    hyp_residual_normalised: f64,
    hyp_at_upper_bound: bool,
    hyp_kappa: f64,

    euc_residual_normalised: f64,
    euc_kappa: f64,
}

impl DetectRecord {
    /// This record's `(rho, pinned, kappa)` under one geometry, or `None` for a
    /// geometry the record does not carry.
    fn arm(&self, geometry: &str) -> Option<(f64, bool, f64)> {
        match geometry {
            "spherical" => Some((
                self.sph_residual_normalised,
                self.sph_at_upper_bound,
                self.sph_kappa,
            )),
            "hyperbolic" => Some((
                self.hyp_residual_normalised,
                self.hyp_at_upper_bound,
                self.hyp_kappa,
            )),
            // Nothing to pin: the flat model carries no free radius.
            "euclidean" => Some((self.euc_residual_normalised, false, self.euc_kappa)),
            _ => None,
        }
    }
}

// ─── Output ──────────────────────────────────────────────────────────────────

/// What the Wilson fit contributes to one row.
///
/// Every field here describes **the generator**, which is why
/// `--wilson-fallback` may carry the whole block across sample sizes. `ρ` is
/// gauged by `n · d_max²` precisely so it compares across sample sizes, and
/// `κ = |K|·R_rms²` likewise; each is a statistic of the fit or the
/// reconstruction alone, converging as the sample grows. Reusing them from
/// another `N` is defensible, and `wilson_n` records that it happened.
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
    kappa: Vec<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    let detected: Vec<DetectRecord> = load_jsonl(&args.kappa_data)?;
    // (dataset, n) → the record, whose three arms a row then selects from.
    let mut detect_by_key: BTreeMap<(String, usize), DetectRecord> = BTreeMap::new();
    for d in detected {
        detect_by_key.insert((d.dataset.clone(), d.n_samples), d);
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

        let wilson = wilson_summary(&detect_by_key, dataset, *n, geometry, args.wilson_fallback);

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

/// This row's Wilson arm, taken from the detection record for its dataset.
fn wilson_summary(
    detect_by_key: &BTreeMap<(String, usize), DetectRecord>,
    dataset: &str,
    n: usize,
    geometry: &str,
    fallback: bool,
) -> Option<WilsonSummary> {
    let exact = detect_by_key.get(&(dataset.to_string(), n));
    let d = match (exact, fallback) {
        (Some(d), _) => d,
        // No detect run at this N: fall back to whichever N was run, if allowed.
        // The fit characterises the generator, which does not change with sample
        // size — only the quality of its estimate does. `wilson_n` keeps that
        // substitution visible rather than silent.
        (None, true) => detect_by_key
            .iter()
            .find(|((ds, _), _)| ds == dataset)
            .map(|(_, d)| d)?,
        (None, false) => return None,
    };

    let (rho, pinned, kappa) = d.arm(geometry)?;
    Some(WilsonSummary {
        wilson_n: d.n_samples,
        rho,
        pinned,
        kappa,
    })
}
