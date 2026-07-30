//! Stage 2 of the hypervolume analysis: ΔH over the baseline + cross-dataset test.
//!
//! Reads the per-cell HV table(s) stage 1 produced and, for each
//! (N, geometry, setting) group, forms
//! `ΔH_d = HV(setting, dataset d) − HV(all_off, dataset d)` over the datasets
//! that have both, then tests `ΔH ≠ 0` across datasets with a Wilcoxon
//! signed-rank test (Demšar 2006's recommended paired cross-dataset test).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::stats;

/// The setting every other setting is compared against.
pub const BASELINE: &str = "all_off";

/// One stage-1 per-cell record (one line of the `hv stats` output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HvRecord {
    pub stem: String,
    pub setting: String,
    pub dataset: String,
    pub n: usize,
    pub geometry: String,
    pub n_mc: u64,
    pub n_trials: usize,
    pub n_front: usize,
    pub hv: f64,
    pub hv_se: f64,
}

/// One (n, geometry, setting, dataset) row: its HV, the baseline's, and ΔH.
#[derive(Debug, Clone, Serialize)]
pub struct DeltaRow {
    pub n: usize,
    pub geometry: String,
    pub setting: String,
    pub dataset: String,
    pub hv: f64,
    pub hv_baseline: Option<f64>,
    pub delta_hv: Option<f64>,
}

/// Per-(n, geometry, setting) ΔH summary with the cross-dataset signed test.
#[derive(Debug, Clone, Serialize)]
pub struct GroupSummary {
    pub n: usize,
    pub geometry: String,
    pub setting: String,
    pub n_datasets: usize,
    pub mean_delta: Option<f64>,
    pub median_delta: Option<f64>,
    pub n_positive: usize,
    pub wilcoxon_p: Option<f64>,
}

/// Concatenate per-cell HV records from one or more stage-1 JSONL files.
///
/// A later file wins on a duplicate `stem` (e.g. a re-run shard), so mixing a
/// coarse local table with a few refined shards keeps the refined values.
pub fn load_hv_table(paths: &[impl AsRef<Path>]) -> std::io::Result<Vec<HvRecord>> {
    use std::io::BufRead;
    let mut by_stem: BTreeMap<String, HvRecord> = BTreeMap::new();
    for path in paths {
        let file = std::fs::File::open(path.as_ref())?;
        for line in std::io::BufReader::new(file).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let rec: HvRecord = serde_json::from_str(line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{}: {e}", path.as_ref().display()),
                )
            })?;
            by_stem.insert(rec.stem.clone(), rec);
        }
    }
    Ok(by_stem.into_values().collect())
}

/// ΔH rows for every (n, geometry, setting, dataset) against the baseline.
pub fn compute_deltas(table: &[HvRecord]) -> Vec<DeltaRow> {
    let mut hv: BTreeMap<(usize, String, String, String), f64> = BTreeMap::new();
    for r in table {
        hv.insert(
            (
                r.n,
                r.geometry.clone(),
                r.setting.clone(),
                r.dataset.clone(),
            ),
            r.hv,
        );
    }

    hv.iter()
        .map(|((n, geom, setting, dataset), &value)| {
            let base = hv
                .get(&(*n, geom.clone(), BASELINE.to_string(), dataset.clone()))
                .copied();
            DeltaRow {
                n: *n,
                geometry: geom.clone(),
                setting: setting.clone(),
                dataset: dataset.clone(),
                hv: value,
                hv_baseline: base,
                delta_hv: base.map(|b| value - b),
            }
        })
        .collect()
}

/// Per-(n, geometry, setting) ΔH summary with the cross-dataset signed test.
pub fn summarise(rows: &[DeltaRow]) -> Vec<GroupSummary> {
    let mut groups: BTreeMap<(usize, String, String), Vec<f64>> = BTreeMap::new();
    for r in rows {
        let entry = groups
            .entry((r.n, r.geometry.clone(), r.setting.clone()))
            .or_default();
        if let Some(d) = r.delta_hv {
            entry.push(d);
        }
    }

    groups
        .into_iter()
        .map(|((n, geometry, setting), deltas)| {
            // The baseline vs itself is all-zero; a signed-rank test is undefined
            // and meaningless there, so skip it.
            let wilcoxon_p = if setting != BASELINE && deltas.iter().any(|&d| d != 0.0) {
                stats::wilcoxon(&deltas).map(|(_, p, _)| p)
            } else {
                None
            };
            GroupSummary {
                n,
                geometry,
                setting,
                n_datasets: deltas.len(),
                mean_delta: stats::mean(&deltas),
                median_delta: stats::median(&deltas),
                n_positive: deltas.iter().filter(|&&d| d > 0.0).count(),
                wilcoxon_p,
            }
        })
        .collect()
}
