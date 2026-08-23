//! Stage 2: ΔR2 over the baseline + the cross-dataset rank test.
//!
//! Reads the per-cell indicator table stage 1 produced and forms
//! `ΔR2_d = R2(all_off, d) − R2(setting, d)` per dataset and preference region.
//!
//! The subtraction runs baseline-minus-setting because the R2 indicator is a
//! distance to the ideal point: smaller is better, so a *reduction* is the gain,
//! and `ΔR2 > 0` reads the same direction as the quality metrics themselves.
//!
//! Two views of the same rows:
//!
//! * [`summarise`] is descriptive: mean, median and sign count of ΔR2 per
//!   (N, geometry, setting, region).
//! * [`rank_tests`] is inferential and is what the thesis reports. Settings are
//!   ranked within each (dataset, geometry, N) block, a Friedman test is applied
//!   to the ranks, and post-hoc comparisons against the `all_off` control are
//!   Holm-corrected (Demšar 2006 §3.2.2). Blocks are the unit of replication.
//!
//! Friedman needs **complete** blocks, and the sweep grid is not rectangular:
//! `norm_only` has no spherical runs and `rms_anchored` only exists for
//! hyperbolic. [`rank_tests`] therefore fixes a setting list first and then
//! keeps only the blocks that carry every one of them, reporting how many it
//! dropped rather than silently imputing.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::records::load_jsonl;
use crate::stats;

/// The setting every other setting is compared against.
pub const BASELINE: &str = "all_off";

/// One stage-1 per-cell record (one line of the `r2 stats` output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellRecord {
    pub stem: String,
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
#[derive(Debug, Clone, Serialize)]
pub struct DeltaRow {
    pub n: usize,
    pub geometry: String,
    pub setting: String,
    pub dataset: String,
    pub region: String,
    pub r2: f64,
    pub r2_baseline: Option<f64>,
    pub delta_r2: Option<f64>,
}

/// Per-(n, geometry, setting, region) ΔR2 summary with the cross-dataset test.
#[derive(Debug, Clone, Serialize)]
pub struct GroupSummary {
    pub n: usize,
    pub geometry: String,
    pub setting: String,
    pub region: String,
    pub n_datasets: usize,
    pub mean_delta: Option<f64>,
    pub median_delta: Option<f64>,
    pub n_positive: usize,
}

/// Concatenate per-cell records from one or more stage-1 JSONL files.
///
/// A later file wins on a duplicate `stem`, so re-running a few cells at a
/// different setting and concatenating keeps the newer values.
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
pub fn compute_deltas(table: &[CellRecord]) -> Vec<DeltaRow> {
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

/// One Friedman test with its Holm post-hoc comparisons against the control.
#[derive(Debug, Clone, Serialize)]
pub struct RankTest {
    pub region: String,
    /// A single geometry, or `"*"` when blocks are pooled across geometries.
    pub geometry: String,
    /// Treatments, in the order `mean_ranks` and `holm_p` use.
    pub settings: Vec<String>,
    /// Complete blocks the test ran on.
    pub n_blocks: usize,
    /// Blocks discarded for not carrying every setting.
    pub dropped_blocks: usize,
    pub statistic: f64,
    /// Friedman omnibus p-value.
    pub p: f64,
    /// Mean rank per setting; 1 is best.
    pub mean_ranks: Vec<f64>,
    /// Holm-adjusted p vs the `all_off` control; `None` at the control itself.
    pub holm_p: Vec<Option<f64>>,
}

/// Friedman + Holm over the (dataset, geometry, N) blocks, per preference region.
///
/// Emits one test per region pooled over geometries (`geometry = "*"`) and one
/// per (region, geometry), because the second prediction of Experiment 2 is a
/// setting-by-geometry interaction and pooling would hide it.
///
/// *settings* fixes the treatment list. Only blocks carrying every one of them
/// contribute; the rest are counted in `dropped_blocks`. A group that cannot
/// support the test at all (fewer than two complete blocks, or fewer than three
/// settings) is omitted rather than reported with a missing p-value.
pub fn rank_tests(rows: &[DeltaRow], settings: &[String]) -> Vec<RankTest> {
    if settings.len() < 3 {
        return Vec::new();
    }
    let control = match settings.iter().position(|s| s == BASELINE) {
        Some(i) => i,
        None => return Vec::new(),
    };

    // (region, geometry, block) → setting → ΔR2. The geometry is carried in the
    // key as well as the block so the pooled and per-geometry views share it.
    type BlockKey = (usize, String, String); // (n, geometry, dataset)
    let mut by_region: BTreeMap<String, BTreeMap<BlockKey, BTreeMap<String, f64>>> =
        BTreeMap::new();
    for r in rows {
        let Some(d) = r.delta_r2 else { continue };
        if !d.is_finite() || !settings.contains(&r.setting) {
            continue;
        }
        by_region
            .entry(r.region.clone())
            .or_default()
            .entry((r.n, r.geometry.clone(), r.dataset.clone()))
            .or_default()
            .insert(r.setting.clone(), d);
    }

    let mut out = Vec::new();
    for (region, blocks) in &by_region {
        let mut geometries: Vec<String> = blocks.keys().map(|(_, g, _)| g.clone()).collect();
        geometries.sort();
        geometries.dedup();

        // "*" first, then one scope per geometry.
        let scopes = std::iter::once("*".to_string()).chain(geometries);
        for scope in scopes {
            let selected = blocks
                .iter()
                .filter(|((_, g, _), _)| scope == "*" || *g == scope);
            let mut complete: Vec<Vec<f64>> = Vec::new();
            let mut dropped = 0usize;
            for (_, values) in selected {
                match settings.iter().map(|s| values.get(s).copied()).collect() {
                    Some(block) => complete.push(block),
                    None => dropped += 1,
                }
            }
            let Some(f) = stats::friedman(&complete) else {
                continue;
            };
            out.push(RankTest {
                region: region.clone(),
                geometry: scope,
                settings: settings.to_vec(),
                n_blocks: f.n_blocks,
                dropped_blocks: dropped,
                statistic: f.statistic,
                p: f.p,
                mean_ranks: f.mean_ranks.clone(),
                holm_p: stats::holm_against_control(&f, control),
            });
        }
    }
    out
}

/// The settings present in *rows*, ordered with the baseline first.
pub fn settings(rows: &[DeltaRow]) -> Vec<String> {
    let mut names: Vec<String> = rows
        .iter()
        .map(|r| r.setting.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    names.sort_by_key(|s| (s != BASELINE, s.clone()));
    names
}

/// Per-(n, geometry, setting, region) ΔR2 summary with the cross-dataset test.
pub fn summarise(rows: &[DeltaRow]) -> Vec<GroupSummary> {
    let mut groups: BTreeMap<(usize, String, String, String), Vec<f64>> = BTreeMap::new();
    for r in rows {
        let entry = groups
            .entry((r.n, r.geometry.clone(), r.setting.clone(), r.region.clone()))
            .or_default();
        if let Some(d) = r.delta_r2 {
            entry.push(d);
        }
    }

    groups
        .into_iter()
        .map(|((n, geometry, setting, region), deltas)| GroupSummary {
            n,
            geometry,
            setting,
            region,
            n_datasets: deltas.len(),
            mean_delta: stats::mean(&deltas),
            median_delta: stats::median(&deltas),
            n_positive: deltas.iter().filter(|&&d| d > 0.0).count(),
        })
        .collect()
}
