//! R2-indicator analysis of the qParEGO sweeps.
//!
//! Three subcommands, each writing a **file** — JSONL throughout, the same format
//! the sweeps themselves are written in. Nothing goes to stdout; the one thing
//! that reaches the terminal is a failure, rendered once by `main` returning
//! `Err`.
//!
//! * **`stats`** — stage 1. For every experiment cell (one results `.jsonl` file
//!   = one (setting, dataset, N, geometry) run) compute the R2 indicator of its
//!   Pareto front under each preference region. One JSON object per cell.
//!
//! * **`aggregate`** — stage 2. ΔR2 over the `all_off` baseline per region, per
//!   dataset.
//!
//! * **`compare`** — the parameter-free cross-check. The binary additive
//!   ε-indicator between each setting's front and the `all_off` baseline's,
//!   in both directions. No preference regions here: having no parameters is
//!   the point.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

use clap::{Parser, Subcommand};

use fitting_analysis::aggregate::{self, CellRecord};
use fitting_analysis::cell::{discover_cells, CellFile};
use fitting_analysis::indicators::{epsilon_pair, EpsilonRow};
use fitting_analysis::objectives::{oriented_matrix, resolve_space, ObjectiveSpace};
use fitting_analysis::r2::{cell_summary, Weights};
use fitting_analysis::{pareto_front_records, trial_records, write_jsonl, Error, Result};

#[derive(Parser, Debug)]
#[command(name = "r2", about = "R2-indicator analysis of the qParEGO sweeps")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Stage 1: per-cell R2 indicator under each preference region.
    Stats(StatsArgs),
    /// Stage 2: ΔR2 over the baseline.
    Aggregate(AggregateArgs),
    /// Parameter-free cross-check: the binary additive ε-indicator vs the baseline.
    Compare(CompareArgs),
}

#[derive(Parser, Debug)]
struct StatsArgs {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Output JSONL path (one line per cell). Defaults to
    /// `results/r2_local_<space>.jsonl`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Force the objective space instead of reading it off the sweeps.
    #[arg(long)]
    objectives: Option<ObjectiveSpace>,
}

#[derive(Parser, Debug)]
struct AggregateArgs {
    /// Stage-1 JSONL file(s), as written by `stats --out`.
    #[arg(required = true)]
    tables: Vec<PathBuf>,

    /// Optional path to write the per-dataset ΔR2 rows as JSONL. Absent means
    /// the table is not written; the pipeline's path is
    /// `results/r2_delta.jsonl`, which is where `figures --r2-delta` looks for
    /// it.
    #[arg(long)]
    deltas: Option<PathBuf>,

    /// Restrict every written table to one preference region.
    #[arg(long)]
    region: Option<String>,
}

#[derive(Parser, Debug)]
struct CompareArgs {
    /// Directory of `*.jsonl` result files.
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Output JSONL path (one line per (dataset, geometry, N, setting)).
    /// Force the objective space instead of reading it off the sweeps.
    #[arg(long)]
    objectives: Option<ObjectiveSpace>,

    #[arg(long, default_value = "results/r2_epsilon.jsonl")]
    out: PathBuf,

    /// Settings to compare against the baseline. The default is the four
    /// loss-weight settings of Experiment 4; `rms_anchored` is excluded because
    /// it fixes a different gauge and only exists for hyperbolic.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "centering_only,global_only,norm_only,all_free"
    )]
    settings: Vec<String>,
}

/// (N, geometry, dataset) — the block a set of comparable cells shares.
type BlockKey = (usize, String, String);

/// The default output path for *stem*, tagged with the objective space.
///
/// Every default carries the tag because a table scored in one space and a
/// table scored in the other are not comparable, and an untagged default would
/// let the second run silently overwrite the first. `--out` still overrides it.
fn tagged(stem: &str, space: ObjectiveSpace) -> PathBuf {
    PathBuf::from(format!("results/{stem}_{}.jsonl", space.tag()))
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Stats(a) => run_stats(a),
        Command::Aggregate(a) => run_aggregate(&a),
        Command::Compare(a) => run_compare(a),
    }
}

// ─── Stage 1: per-cell R2 ─────────────────────────────────────────────────────

fn run_stats(args: StatsArgs) -> Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    let space = resolve_space(&cells, args.objectives)?;
    let weights = Weights::new(space);

    // Cells are visited in `discover_cells` order
    let mut rows = Vec::with_capacity(cells.len());
    for cf in &cells {
        let records = trial_records(&cf.path)?;
        let summary = cell_summary(&records, &weights);
        rows.push(CellRecord {
            stem: cf.stem.clone(),
            space: space.tag().to_string(),
            setting: cf.cell.setting.clone(),
            dataset: cf.cell.dataset.clone(),
            n: cf.cell.n,
            geometry: cf.cell.geometry.clone(),
            n_trials: summary.n_trials,
            n_front: summary.n_front,
            r2: summary.r2.clone(),
        });
    }
    write_jsonl(args.out.unwrap_or_else(|| tagged("r2_local", space)), &rows)
}

// ─── Stage 2: ΔR2 + the rank test ─────────────────────────────────────────────

/// The one objective space a stage-1 table was written in.
///
/// # Errors
///
/// Returns [`Error::MixedObjectiveSpaces`] if the rows disagree.
fn space_of_table(table: &[CellRecord]) -> Result<ObjectiveSpace> {
    let mut found: Option<&CellRecord> = None;
    for row in table {
        match found {
            None => found = Some(row),
            Some(first) if first.space != row.space => {
                return Err(Error::MixedObjectiveSpaces {
                    first: first.stem.clone(),
                    first_space: ObjectiveSpace::from_str(&first.space)
                        .map_or("unknown", ObjectiveSpace::tag),
                    second: row.stem.clone(),
                    second_space: ObjectiveSpace::from_str(&row.space)
                        .map_or("unknown", ObjectiveSpace::tag),
                })
            }
            Some(_) => {}
        }
    }
    let tag = found.map_or(ObjectiveSpace::Current6.tag(), |r| r.space.as_str());
    ObjectiveSpace::from_str(tag).map_err(|_| Error::UnknownObjectiveSpace(tag.to_string()))
}

fn run_aggregate(args: &AggregateArgs) -> Result<()> {
    let table: Vec<CellRecord> = aggregate::load_table(&args.tables)?;
    if table.is_empty() {
        let first = args.tables.first().cloned().unwrap_or_default();
        return Err(Error::NoCells(first));
    }
    // Stage 2 differences R2 against a baseline cell, so every row it reads has
    // to be in one known space; a row tagged otherwise is caught here rather
    // than producing a ΔR2 column in two units.
    space_of_table(&table)?;
    if let Some(region) = &args.region {
        let available = aggregate::regions(&table);
        if !available.iter().any(|r| r == region) {
            return Err(Error::UnknownRegion {
                region: region.clone(),
                available,
            });
        }
    }

    let mut rows = aggregate::compute_deltas(&table);
    let keep_region = |region: &str| args.region.as_deref().is_none_or(|r| r == region);

    // The per-dataset ΔR2 rows the thesis tables read.
    if let Some(path) = &args.deltas {
        rows.sort_by(|a, b| {
            (a.n, &a.geometry, &a.setting, &a.region, &a.dataset).cmp(&(
                b.n,
                &b.geometry,
                &b.setting,
                &b.region,
                &b.dataset,
            ))
        });
        write_jsonl(path, rows.iter().filter(|r| keep_region(&r.region)))?;
    }
    Ok(())
}

// ─── The parameter-free cross-check ───────────────────────────────────────────

fn run_compare(args: CompareArgs) -> Result<()> {
    let cells = discover_cells(&args.results_dir)?;
    if cells.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }

    // (N, geometry, dataset) → setting → cell. Sorted keys throughout, so the
    // output is sorted by (n, geometry, dataset, setting) without a final sort
    // and is byte-identical across runs.
    let mut blocks: BTreeMap<BlockKey, BTreeMap<&str, &CellFile>> = BTreeMap::new();
    for cf in &cells {
        blocks
            .entry((cf.cell.n, cf.cell.geometry.clone(), cf.cell.dataset.clone()))
            .or_default()
            .insert(cf.cell.setting.as_str(), cf);
    }

    // Comparing the baseline with itself is not a comparison; drop it silently
    // so `--settings` can be pasted from the `aggregate` invocation.
    let wanted: Vec<&str> = args
        .settings
        .iter()
        .map(String::as_str)
        .filter(|s| *s != aggregate::BASELINE)
        .collect();

    let space = resolve_space(&cells, args.objectives)?;
    let mut rows: Vec<EpsilonRow> = Vec::new();
    for ((n, geometry, dataset), by_setting) in &blocks {
        let present: Vec<&str> = wanted
            .iter()
            .copied()
            .filter(|s| by_setting.contains_key(s))
            .collect();
        if present.is_empty() {
            continue;
        }
        let Some(base_cf) = by_setting.get(aggregate::BASELINE) else {
            return Err(Error::NoBaselineCell {
                baseline: aggregate::BASELINE,
                n: *n,
                geometry: geometry.clone(),
                dataset: dataset.clone(),
            });
        };
        // One cell of records at a time: the front is all that outlives the load.
        let baseline = oriented_matrix(
            &pareto_front_records(&trial_records(&base_cf.path)?, space),
            space,
        );

        for setting in present {
            let cf = by_setting[setting];
            let front = oriented_matrix(
                &pareto_front_records(&trial_records(&cf.path)?, space),
                space,
            );
            // An empty front means an empty cell; there is nothing to compare.
            let Some(eps) = epsilon_pair(&front, &baseline) else {
                continue;
            };
            rows.push(EpsilonRow {
                dataset: dataset.clone(),
                geometry: geometry.clone(),
                n: *n,
                setting: setting.to_string(),
                n_front_setting: front.len(),
                n_front_baseline: baseline.len(),
                eps_setting_vs_baseline: eps.setting_vs_baseline,
                eps_baseline_vs_setting: eps.baseline_vs_setting,
                delta_eps: eps.delta,
                setting_covers_baseline: eps.setting_covers_baseline(),
                baseline_covers_setting: eps.baseline_covers_setting(),
            });
        }
    }
    if rows.is_empty() {
        return Err(Error::NoCells(args.results_dir));
    }
    write_jsonl(&args.out, &rows)
}
