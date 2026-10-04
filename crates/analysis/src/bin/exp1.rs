//! Experiment 1: does the embedding geometry matching the data's intrinsic
//! curvature beat the Euclidean baseline?
//!
//! Scores the **Pareto front** each `(dataset, N, geometry)` sweep produced by
//! the R2 indicator under every preference region, exactly as `r2 stats` does,
//! and answers: does curvature-matched t-SNE beat Euclidean t-SNE? That is
//! `delta_r2`, formed as `R2(euclidean) − R2(geometry)` so positive favours the
//! curved arm (the same direction as `aggregate.rs`, because R2 is a cost).
//!
//! Alongside `delta_r2`, and answering the same question without its preference
//! model, is the `epsilon` block: the binary additive ε-indicator between this
//! row's front and the *matched* geometry's front, in both directions. R2 is an
//! average over 252 weight vectors and depends on the region definitions; ε is
//! parameter-free, fully Pareto compliant and a worst case, so it is the
//! cross-check that the ΔR2 verdicts are not an artefact of that model. It is
//! referenced to the matched arm rather than to Euclidean because, unlike R2, ε
//! is not a number a third front can be scored against — a comparison is a
//! *pair* of fronts, and the pair the experiment asks about is
//! matched-against-mismatched.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;

use fitting_analysis::cell::{discover_cells, truth_of};
use fitting_analysis::indicators::epsilon_pair;
use fitting_analysis::objectives::{oriented_row, resolve_space, ObjectiveSpace, Row};
use fitting_analysis::r2::{cell_summary, Weights};
use fitting_analysis::{trial_records, write_jsonl, Error, Result};

/// The loss-weight setting Experiment 1 reads. `all_off` zeroes every auxiliary
/// loss, recovering plain KL-divergence t-SNE, so geometry is the only thing
/// varying between the cells being compared — which is the whole question.
const DEFAULT_SETTING: &str = "all_off";

#[derive(Parser, Debug)]
#[command(about = "Experiment 1: matched-geometry vs the Euclidean baseline")]
struct Args {
    /// Directory of sweep results (`<setting>_<dataset>[_n5000]_<geometry>.jsonl`).
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Loss-weight setting to read.
    #[arg(long, default_value = DEFAULT_SETTING)]
    setting: String,

    /// Output JSONL. Defaults to
    /// `results/exp1_geometry_match_<space>.jsonl`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Force the objective space instead of reading it off the sweeps.
    #[arg(long)]
    objectives: Option<ObjectiveSpace>,
}

// ─── Output ──────────────────────────────────────────────────────────────────

/// The ε-indicator between this row's front and the matched geometry's, in
/// both directions.
///
/// `I_ε+(A, B)` is the smallest amount by which every objective of *A* must be
/// shifted before *A* dominates *B*, so `I_ε+(A, B) ≤ 0` exactly when *A*
/// covers *B* (`crate::indicators`). The measure is **asymmetric**, and neither
/// direction alone settles the comparison when the two fronts cross, so both
/// are carried.
///
/// Absent on the matched row itself — a front is not compared with itself — and
/// on any row whose dataset has no matched cell, which is the same condition
/// that leaves the figure's group undrawn.
#[derive(Debug, Clone, Serialize)]
struct EpsilonSummary {
    /// The geometry the comparison is against: this dataset's `truth`. Written
    /// out so a row is readable without the ground-truth map to hand.
    matched_geometry: &'static str,
    /// Size of the matched front. Reported next to the indicator because ε says
    /// nothing about cardinality: a front of 12 points and one of 230 can score
    /// the same. This row's own front size is the sibling `n_front`.
    n_front_matched: usize,
    /// `I_ε+(matched, geometry)`: how far the matched front must be shifted to
    /// cover this row's. Smaller is better *for the matched arm*.
    eps_matched_vs_geometry: f64,
    /// `I_ε+(geometry, matched)`: the same in the other direction.
    eps_geometry_vs_matched: f64,
    /// `eps_geometry_vs_matched − eps_matched_vs_geometry`. Positive means the
    /// matched geometry came out ahead — the same reading direction as
    /// `delta_r2` and `@eq:r2-gain`, because ε is likewise a cost.
    delta_eps: f64,
    /// `eps_matched_vs_geometry ≤ 0`: the matched front covers this one
    /// outright.
    matched_covers_geometry: bool,
    /// `eps_geometry_vs_matched ≤ 0`: this front covers the matched one
    /// outright.
    geometry_covers_matched: bool,
}

/// One `(dataset, N, geometry)` row.
#[derive(Debug, Clone, Serialize)]
struct Exp1Row {
    dataset: String,
    n: usize,
    setting: String,
    /// The objective space the R2 values were computed in
    /// (`ObjectiveSpace::tag`).
    space: &'static str,
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

    /// The parameter-free cross-check on `delta_r2`, against the matched arm.
    /// `null` on the matched row itself and where the matched cell is missing.
    epsilon: Option<EpsilonSummary>,
}

// ─── The join ────────────────────────────────────────────────────────────────

/// Everything one cell contributes before the Euclidean baseline is known.
struct Scored {
    n_trials: usize,
    n_front: usize,
    r2: BTreeMap<String, f64>,
    /// The front itself, oriented (all objectives in `[0, 1]`, higher better).
    /// R2 reduces a front to one number per region, which is enough to
    /// difference; ε does not — it is computed *between* two fronts — so the
    /// points have to outlive the cell that produced them. A front is a few
    /// hundred rows of six floats, so holding every cell's is nothing next to
    /// the trial records they came from, which are dropped.
    front: Vec<Row>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    let space = resolve_space(&cells, args.objectives)?;
    let weights = Weights::new(space);

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

        // The front's points, kept for the ε comparison in pass 2. Re-oriented
        // from the records rather than returned by `cell_summary`, which hands
        // back indices; the rows it built are the same ones, since both go
        // through `oriented_row` in this same space.
        let front: Vec<Row> = summary
            .front
            .iter()
            .map(|&i| oriented_row(&records[i], space))
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

        let epsilon = epsilon_summary(&scored, dataset, *n, geometry, truth, cell);

        rows.push(Exp1Row {
            dataset: dataset.clone(),
            n: *n,
            setting: args.setting.clone(),
            space: space.tag(),
            geometry: geometry.clone(),
            truth,
            matched: truth == geometry,
            n_trials: cell.n_trials,
            n_front: cell.n_front,
            r2: cell.r2.clone(),
            r2_euclidean: euclidean.r2.clone(),
            delta_r2,
            epsilon,
        });
    }

    write_jsonl(
        args.out.unwrap_or_else(|| {
            PathBuf::from(format!("results/exp1_geometry_match_{}.jsonl", space.tag()))
        }),
        &rows,
    )
}

/// This row's ε comparison against the matched arm of the same dataset and *N*.
///
/// `None` on the matched row itself — `I_ε+(A, A) = 0` in both directions,
/// which is a tautology and not a comparison — and on a dataset whose matched
/// cell was not swept, where there is nothing to reference against. A missing
/// matched cell is *not* an error, unlike a missing Euclidean one: `delta_r2`
/// is the table's headline column and the whole table is unanswerable without
/// its baseline, where ε is the cross-check.
fn epsilon_summary(
    scored: &BTreeMap<(String, usize, String), Scored>,
    dataset: &str,
    n: usize,
    geometry: &str,
    truth: &'static str,
    arm: &Scored,
) -> Option<EpsilonSummary> {
    if geometry == truth {
        return None;
    }
    let matched = scored.get(&(dataset.to_string(), n, truth.to_string()))?;

    // `epsilon_pair(setting, baseline)` reads its first argument as the
    // treatment: here the matched geometry, whose case the experiment is
    // making, against the mismatched arm as control. That is what puts
    // `delta_eps` the same way round as `delta_r2`.
    let eps = epsilon_pair(&matched.front, &arm.front)?;
    Some(EpsilonSummary {
        matched_geometry: truth,
        n_front_matched: matched.n_front,
        eps_matched_vs_geometry: eps.setting_vs_baseline,
        eps_geometry_vs_matched: eps.baseline_vs_setting,
        delta_eps: eps.delta,
        matched_covers_geometry: eps.setting_covers_baseline(),
        geometry_covers_matched: eps.baseline_covers_setting(),
    })
}
