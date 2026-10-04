//! Stage 2: ΔR2 over the baseline.
//!
//! Reads the per-cell indicator table stage 1 produced and forms
//! `ΔR2_d = R2(all_off, d) − R2(setting, d)` per dataset and preference region.
//!
//! The subtraction runs baseline-minus-setting because the R2 indicator is a
//! distance to the ideal point: smaller is better, so a *reduction* is the gain,
//! and `ΔR2 > 0` reads the same direction as the quality metrics themselves.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::records::load_jsonl;

/// The setting every other setting is compared against.
pub const BASELINE: &str = "all_off";

/// One stage-1 per-cell record (one line of the `r2 stats` output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRecord {
    pub stem: String,
    /// The objective space this cell was scored in, as
    /// `ObjectiveSpace::tag`, so stage 2 can tag its outputs without
    /// re-reading the sweeps.
    pub space: String,
    pub setting: String,
    pub dataset: String,
    pub n: usize,
    pub geometry: String,
    pub n_trials: usize,
    pub n_front: usize,
    /// Preference region name → R2 indicator.
    pub r2: BTreeMap<String, f64>,
}

/// One (n, geometry, setting, dataset, region) row: its R2, the baseline's, and ΔR2.
///
/// `Deserialize` as well as `Serialize` because the figures read this table back
/// from its JSONL: the ΔR2 bar charts plot the rows this stage writes rather
/// than recomputing them, so the charts and the thesis table cannot disagree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaRow {
    /// The objective space these R2 values were computed in, carried through
    /// from [`CellRecord::space`] so the bar charts and the Typst table pick
    /// the right region columns without re-reading the sweeps.
    pub space: String,
    pub n: usize,
    pub geometry: String,
    pub setting: String,
    pub dataset: String,
    pub region: String,
    pub r2: f64,
    pub r2_baseline: Option<f64>,
    pub delta_r2: Option<f64>,
}

/// Concatenate per-cell records from one or more stage-1 JSONL files.
///
/// A later file wins on a duplicate `stem`, so re-running a few cells at a
/// different setting and concatenating keeps the newer values.
///
/// # Errors
///
/// Propagates errors from [`load_jsonl`] (file I/O or deserialization).
pub fn load_table(paths: &[impl AsRef<Path>]) -> Result<Vec<CellRecord>> {
    let mut by_stem: BTreeMap<String, CellRecord> = BTreeMap::new();
    for path in paths {
        for rec in load_jsonl::<CellRecord>(path)? {
            by_stem.insert(rec.stem.clone(), rec);
        }
    }
    Ok(by_stem.into_values().collect())
}

/// Every region name present in the table, in first-seen-sorted order.
#[must_use]
pub fn regions(table: &[CellRecord]) -> Vec<String> {
    let mut names: Vec<String> = table
        .iter()
        .flat_map(|r| r.r2.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    names.sort();
    names
}

/// ΔR2 rows for every (n, geometry, setting, dataset, region) against the baseline.
#[must_use]
pub fn compute_deltas(table: &[CellRecord]) -> Vec<DeltaRow> {
    // One table is one space — `run_aggregate` rejects a mixed one before
    // getting here — so the first row's tag names every output row's.
    let space = table.first().map_or_else(
        || {
            crate::objectives::ObjectiveSpace::Current6
                .tag()
                .to_string()
        },
        |r| r.space.clone(),
    );
    let mut values: BTreeMap<(usize, String, String, String, String), f64> = BTreeMap::new();
    for r in table {
        for (region, &v) in &r.r2 {
            values.insert(
                (
                    r.n,
                    r.geometry.clone(),
                    r.setting.clone(),
                    r.dataset.clone(),
                    region.clone(),
                ),
                v,
            );
        }
    }

    values
        .iter()
        .map(|((n, geom, setting, dataset, region), &value)| {
            let base = values
                .get(&(
                    *n,
                    geom.clone(),
                    BASELINE.to_string(),
                    dataset.clone(),
                    region.clone(),
                ))
                .copied();
            DeltaRow {
                space: space.clone(),
                n: *n,
                geometry: geom.clone(),
                setting: setting.clone(),
                dataset: dataset.clone(),
                region: region.clone(),
                r2: value,
                r2_baseline: base,
                // Baseline minus setting: the indicator is a cost, so a drop is a gain.
                delta_r2: base.map(|b| b - value),
            }
        })
        .collect()
}
