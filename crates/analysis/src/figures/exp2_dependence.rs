//! Experiment 2 (`metric-dependence`) — do the metrics order the same
//! visualisations the same way?
//!
//! [`MetricDependence`] computes **the Spearman rank correlations between the
//! projected metrics, one matrix per embedding geometry**, which
//! [`super::exp2_dumbbell::DependenceDumbbell`] draws. It answers a different
//! question about the same corpus than [`super::exp2::MetricTrend`]: not how a
//! metric responds to curvature, but whether an "improvement on several
//! metrics" is several findings or one. Correlation is evidence of similar
//! rankings, not proof of redundancy, and the thesis text says so.
//!
//! ### Within each dataset, then across them
//!
//! ρ is computed **inside each `(dataset, geometry, N)` cell** and the figure
//! shows the **median of those per-dataset ρ over the datasets**, with the
//! min–max range across datasets printed under it. The trials are never pooled
//! across datasets: two datasets whose metric readings sit at different levels
//! would correlate perfectly when pooled, whatever happens inside either — the
//! spurious correlation the results chapter warns against. Combining
//! correlations rather than trials is how one figure honours both "within each
//! dataset" and "no pooling". The datasets combined are whichever ones have an
//! `all_off` cell at that geometry and N — synthetic and real alike, and the
//! panel says how many.
//!
//! ### Population
//!
//! The same as [`super::exp2::MetricTrend`]'s, on purpose: **the Pareto
//! front of every `all_off` cell** ([`super::exp2::SETTING`]) — front points,
//! not every trial, because the thesis compares corpora and a corpus is the
//! front, so the ranking asked about is the ranking of the corpus; and the
//! baseline setting only, so the auxiliary loss weights do not vary underneath
//! the ranking. The reduction happens in `bin/figures.rs`, in the scoring
//! space, before the `CellMap` reaches [`MetricDependence::panels`]. A front is
//! a few hundred points where a sweep is a thousand, so [`MIN_TRIALS`] now
//! bites: a dataset cell whose front has fewer complete points contributes
//! nothing, and the panel's `n = used / total` line shows it.
//!
//! Collapsed embeddings are **not** filtered out. They are finite readings of
//! a real trial — 14% of the hyperbolic corpus — and every metric reads them as bad at once, which pulls each
//! pairwise ρ toward +1 on the geometries where collapse happens. That is a
//! property of the population the figure states, not an artefact to remove,
//! and the text is expected to read the hyperbolic panel with it in mind.
//!
//! ### One surface
//!
//! **Projected metrics only** — every `Space::Projected` metric in the
//! registry: the six objectives plus the three unbounded diagnostics
//! (`dunn_index`, `davies_bouldin_ratio`, `cluster_density_measure`), which a
//! rank correlation handles as well as a bounded one. The axis runs in
//! `OBJECTIVES` order — grouped by family, so the structure pair, the distance
//! pair and the class-separation pair sit together — and then the diagnostics,
//! which are label-aware too, directly after `neighborhood_hit` and
//! `distance_consistency`. Registry order would put `neighborhood_hit` between
//! `continuity` and `normalized_stress`, where it is (JSONL column order, not a
//! grouping), and split the label-aware block in two. The `_manifold` twins are not drawn: the thesis judges the 2-D
//! visualisation, and a manifold reading on the same axis as its projected twin
//! is how an earlier figure came to difference an unoriented reading against
//! an oriented one. Keeping the
//! surfaces separate here means drawing exactly one.
//!
//! ### Orientation and missing readings
//!
//! Every metric is read through [`super::exp2::reading`], so `normalized_stress`
//! enters as `1 - stress` and is labelled that way; a positive ρ always means
//! "better on one, better on the other". (For a rank correlation `1 - v` and
//! `-v` are the same thing, so this is the same orientation `objectives` uses,
//! not a third one.)
//!
//! Missing readings are handled **listwise, per dataset cell**: a metric with
//! no finite reading on *any* trial of the cell is dropped from that cell's
//! matrix, and then every trial missing *any* remaining metric is dropped
//! whole. One population per matrix, which the panel reports as
//! `n = used / total` summed over its datasets. A diverged trial, the kind that
//! writes a confident `trustworthiness` beside a `null` stress in the older
//! sweeps, is therefore excluded outright rather than ranked on
//! the metrics it happened to write. A cell needs [`MIN_TRIALS`] complete
//! trials to contribute; below that its ρ would be noise, and the panel's
//! dataset count excludes it.

use plotters::coord::ranged1d::{DefaultFormatting, KeyPointHint, Ranged};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

use fitting_core::cast::{count_to_f64, to_i32};
use fitting_core::metrics::{Metric, Space, ALL, OBJECTIVES};

use super::exp2::{reading, short_label, SETTING};
use super::{CellMap, GEOMETRIES, OK_BLACK, OK_BLUE, OK_GREY, OK_VERMILLION};
use crate::records::TrialRecord;
use crate::stats::{median, spearman_matrix};

/// Complete trials a dataset cell needs before its ρ counts. A cell is ~1000
/// trials, so this only bites on a cell that is nearly all diverged, where a
/// correlation over the survivors would describe the survivors.
pub const MIN_TRIALS: usize = 30;

/// Height of the strip above a heatmap carrying its title, used by
/// [`super::exp2_region_gain::RegionGain`].
pub(super) const TITLE_STRIP: u32 = 40;

/// The metrics on the heatmap: the objectives in [`OBJECTIVES`] order, which
/// is grouped by family, then every other projected metric in registry order
/// — the label-aware diagnostics, landing beside the class-separation pair.
fn projected_metrics() -> Vec<Metric> {
    OBJECTIVES
        .iter()
        .copied()
        .chain(
            ALL.iter()
                .copied()
                .filter(|m| m.space() == Space::Projected && !OBJECTIVES.contains(m)),
        )
        .collect()
}

/// One dataset cell's correlation matrix. An intermediate: the figure is the
/// median over these, and no per-dataset file is written.
#[derive(Debug, Clone)]
struct CellDependence {
    /// The metrics on the axes, and their [`short_label`]s, in one order.
    metrics: Vec<Metric>,
    labels: Vec<String>,
    rho: Vec<Vec<Option<f64>>>,
    /// Trials with every metric in `labels` finite — the rows ρ was taken over.
    n_used: usize,
    /// Trials in the cell.
    n_total: usize,
}

impl CellDependence {
    /// The listwise-complete correlation matrix of *records*, or `None` when
    /// fewer than two metrics have any reading or fewer than [`MIN_TRIALS`]
    /// trials carry all of them.
    fn from_records(records: &[TrialRecord]) -> Option<Self> {
        // A metric no trial measured is not a column — it is absent from the
        // sweep, not failed on every row — so it is dropped before the row
        // filter, which would otherwise empty the cell.
        let metrics: Vec<Metric> = projected_metrics()
            .into_iter()
            .filter(|&m| records.iter().any(|r| reading(m, r).is_some()))
            .collect();
        if metrics.len() < 2 {
            return None;
        }
        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); metrics.len()];
        for r in records {
            let row: Option<Vec<f64>> = metrics.iter().map(|&m| reading(m, r)).collect();
            if let Some(row) = row {
                for (col, v) in columns.iter_mut().zip(row) {
                    col.push(v);
                }
            }
        }
        let n_used = columns[0].len();
        if n_used < MIN_TRIALS {
            return None;
        }
        Some(Self {
            labels: metrics.iter().map(|&m| short_label(m)).collect(),
            metrics,
            rho: spearman_matrix(&columns),
            n_used,
            n_total: records.len(),
        })
    }
}

/// The per-geometry Spearman matrix: the median over datasets of each
/// within-dataset ρ, and the range those datasets span.
pub struct MetricDependence {
    geometry: &'static str,
    /// The metrics on the axes, in draw order, with their labels alongside;
    /// the dumbbell rendering groups pairs by `Metric::family`.
    metrics: Vec<Metric>,
    labels: Vec<String>,
    /// `median[i][j]`, over the datasets whose ρ is defined; `None` if none.
    median: Vec<Vec<Option<f64>>>,
    /// `(min, max)` over the same datasets.
    range: Vec<Vec<Option<(f64, f64)>>>,
    /// The per-dataset ρ themselves, in cell order — what the median and
    /// range summarise, for the rendering that draws every one.
    samples: Vec<Vec<Vec<f64>>>,
    /// Dataset cells that cleared [`MIN_TRIALS`] and so contribute.
    n_datasets: usize,
    /// Complete trials and trials, summed over those cells.
    n_used: usize,
    n_total: usize,
}

impl MetricDependence {
    /// One panel per geometry, in [`GEOMETRIES`] order; a geometry with no
    /// contributing dataset cell is absent rather than returned empty, the
    /// same contract `exp2::MetricTrend::panels` has.
    #[must_use]
    pub fn panels(cells: &CellMap, n: usize) -> Vec<MetricDependence> {
        GEOMETRIES
            .iter()
            .filter_map(|&geometry| {
                let per_dataset: Vec<CellDependence> = cells
                    .iter()
                    .filter(|(cell, _)| {
                        cell.setting == SETTING && cell.n == n && cell.geometry == geometry
                    })
                    .filter_map(|(_, records)| CellDependence::from_records(records))
                    .collect();
                Self::combine(geometry, &per_dataset)
            })
            .collect()
    }

    /// Median and range of the per-dataset matrices, over the metrics every
    /// one of them carries. The label sets agree across the cells of one
    /// results directory (one objective space, one registry), so the
    /// intersection is a guard rather than a path anything takes today.
    fn combine(geometry: &'static str, per_dataset: &[CellDependence]) -> Option<Self> {
        let first = per_dataset.first()?;
        let metrics: Vec<Metric> = first
            .metrics
            .iter()
            .copied()
            .filter(|m| per_dataset.iter().all(|c| c.metrics.contains(m)))
            .collect();
        let labels: Vec<String> = metrics.iter().map(|&m| short_label(m)).collect();
        if labels.len() < 2 {
            return None;
        }
        // Each cell's ρ at the shared labels, by position in `labels`.
        let slots: Vec<Vec<usize>> = per_dataset
            .iter()
            .map(|c| {
                labels
                    .iter()
                    .map(|l| {
                        c.labels
                            .iter()
                            .position(|x| x == l)
                            .expect("label intersected")
                    })
                    .collect()
            })
            .collect();

        let k = labels.len();
        let mut median_m = vec![vec![None; k]; k];
        let mut range_m = vec![vec![None; k]; k];
        let mut samples = vec![vec![Vec::new(); k]; k];
        for i in 0..k {
            for j in 0..k {
                let rhos: Vec<f64> = per_dataset
                    .iter()
                    .zip(&slots)
                    .filter_map(|(c, slot)| c.rho[slot[i]][slot[j]])
                    .collect();
                median_m[i][j] = median(&rhos);
                range_m[i][j] = rhos
                    .iter()
                    .copied()
                    .fold(None, |acc: Option<(f64, f64)>, v| {
                        Some(acc.map_or((v, v), |(lo, hi)| (lo.min(v), hi.max(v))))
                    });
                samples[i][j] = rhos;
            }
        }
        Some(Self {
            geometry,
            metrics,
            labels,
            median: median_m,
            range: range_m,
            samples,
            n_datasets: per_dataset.len(),
            n_used: per_dataset.iter().map(|c| c.n_used).sum(),
            n_total: per_dataset.iter().map(|c| c.n_total).sum(),
        })
    }

    /// Always true for a panel [`MetricDependence::panels`] returned; kept so
    /// the driver reads the same as every other figure's.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.n_datasets > 0 && self.labels.len() >= 2
    }

    /// The metrics on the axes, in draw order.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The same metrics as [`labels`](Self::labels), as the registry knows them.
    #[must_use]
    pub fn metrics(&self) -> &[Metric] {
        &self.metrics
    }

    /// The embedding geometry this panel's ρ were computed on.
    #[must_use]
    pub fn geometry(&self) -> &'static str {
        self.geometry
    }

    /// The drawn value at `(row, col)`.
    #[must_use]
    pub fn median_at(&self, row: usize, col: usize) -> Option<f64> {
        self.median[row][col]
    }

    /// The per-dataset range at `(row, col)`.
    #[must_use]
    pub fn range_at(&self, row: usize, col: usize) -> Option<(f64, f64)> {
        self.range[row][col]
    }

    /// The per-dataset ρ at `(row, col)` that the median and range summarise.
    #[must_use]
    pub fn samples_at(&self, row: usize, col: usize) -> &[f64] {
        &self.samples[row][col]
    }

    /// `(datasets, complete trials, trials)` behind the panel.
    #[must_use]
    pub fn population(&self) -> (usize, usize, usize) {
        (self.n_datasets, self.n_used, self.n_total)
    }
}

/// The diverging ramp every Exp 2 heatmap is filled on: `OK_BLUE` at −1,
/// white at 0, `OK_VERMILLION` at +1, mixed linearly in RGB and clamped
/// outside; `OK_GREY` for an undefined value. Colourblind-safe ends, and the
/// same two hues the geometry palette uses for hyperbolic and spherical — a
/// reader who has seen those panels already knows them. A figure whose values
/// are not in `[-1, 1]` divides by its own scale first.
#[must_use]
pub fn diverging_color(t: Option<f64>) -> RGBColor {
    let Some(t) = t.filter(|t| t.is_finite()) else {
        return OK_GREY;
    };
    let t = t.clamp(-1.0, 1.0);
    let end = if t < 0.0 { OK_BLUE } else { OK_VERMILLION };
    let a = t.abs();
    let mix = |c: u8| {
        let v = 255.0 + (f64::from(c) - 255.0) * a;
        // `v` is inside [c, 255] by construction.
        u8::try_from(to_i32(v.round())).unwrap_or(u8::MAX)
    };
    RGBColor(mix(end.0), mix(end.1), mix(end.2))
}

/// The text colour on a [`diverging_color`] fill at *t*: white once the fill
/// is saturated enough that black would not read.
pub(super) fn text_on(t: Option<f64>) -> RGBColor {
    match t {
        Some(r) if r.abs() > 0.6 => WHITE,
        _ => OK_BLACK,
    }
}

/// A categorical axis over `k` slots, ticked at each slot's centre and
/// labelled with that slot's name. Its own [`Ranged`] for the same reason
/// [`LinearTicks`] is: plotters offers no formatter over an f64 range with
/// custom key points, and the mesh will not label a half-integer otherwise.
/// Reversed when `top_down`, so row 0 of the matrix is at the top.
#[derive(Clone)]
pub(super) struct CategoryAxis {
    labels: Vec<String>,
    range: std::ops::Range<f64>,
}

impl CategoryAxis {
    pub(super) fn new(labels: &[String], top_down: bool) -> Self {
        let k = count_to_f64(labels.len());
        Self {
            labels: labels.to_vec(),
            range: if top_down { k..0.0 } else { 0.0..k },
        }
    }

    /// The name of the slot *v* falls in; empty off the axis. By reference
    /// because that is the signature `x_label_formatter` hands over.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub(super) fn label(&self, v: &f64) -> String {
        // The floor of a tick at `i + 0.5`, which is where the mesh asks.
        let i = v.floor();
        if i < 0.0 {
            return String::new();
        }
        usize::try_from(to_i32(i))
            .ok()
            .and_then(|i| self.labels.get(i))
            .cloned()
            .unwrap_or_default()
    }
}

impl Ranged for CategoryAxis {
    type FormatOption = DefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        RangedCoordf64::from(self.range.clone()).map(value, limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, _hint: Hint) -> Vec<f64> {
        (0..self.labels.len())
            .map(|i| count_to_f64(i) + 0.5)
            .collect()
    }

    fn range(&self) -> std::ops::Range<f64> {
        self.range.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;

    /// One trial line with the given projected readings; anything not named
    /// is absent.
    fn trial(fields: &[(&str, Option<f64>)]) -> TrialRecord {
        let body: Vec<String> = fields
            .iter()
            .map(|(k, v)| match v {
                Some(v) => format!("\"{k}\":{v:?}"),
                None => format!("\"{k}\":null"),
            })
            .collect();
        serde_json::from_str(&format!("{{{}}}", body.join(","))).expect("trial fixture")
    }

    /// `MIN_TRIALS` complete trials where `trustworthiness` rises with the
    /// index and `normalized_stress` rises with it too (so `1-stress` falls),
    /// plus the readings the other columns need to be non-constant.
    fn monotone_cell(extra_rows: Vec<TrialRecord>) -> Vec<TrialRecord> {
        let mut rows: Vec<TrialRecord> = (0..MIN_TRIALS)
            .map(|i| {
                let t = count_to_f64(i) / count_to_f64(MIN_TRIALS);
                trial(&[
                    ("trustworthiness", Some(t)),
                    ("continuity", Some(t * t)),
                    ("normalized_stress", Some(t)),
                    ("shepard_goodness", Some(1.0 - t)),
                    ("neighborhood_hit", Some((t * 7.0).sin())),
                    ("dunn_index", Some(t * 3.0)),
                    ("davies_bouldin_ratio", Some(1.0 / (t + 0.1))),
                    ("cluster_density_measure", Some((t * 3.0).cos())),
                ])
            })
            .collect();
        rows.extend(extra_rows);
        rows
    }

    #[test]
    fn projected_metrics_carry_no_manifold_twin() {
        let metrics = projected_metrics();
        assert!(metrics.len() >= 6, "{metrics:?}");
        assert!(metrics.iter().all(|m| m.space() == Space::Projected));
        assert!(metrics.iter().all(|m| !m.name().ends_with("_manifold")));
        // Every objective is on the heatmap, first and in family order, so
        // the class-separation pair leads straight into the label-aware
        // diagnostics.
        assert_eq!(&metrics[..OBJECTIVES.len()], OBJECTIVES);
        let names: Vec<&str> = metrics.iter().map(|m| m.name()).collect();
        let nh = names.iter().position(|n| *n == "neighborhood_hit").unwrap();
        let db = names
            .iter()
            .position(|n| *n == "davies_bouldin_ratio")
            .unwrap();
        assert!(
            db > nh
                && names[nh + 1..db]
                    .iter()
                    .all(|n| *n == "distance_consistency")
        );
        // Nothing is listed twice.
        let mut dedup = names.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), names.len());
    }

    #[test]
    fn a_metric_nobody_measured_is_a_dropped_column_not_dropped_rows() {
        let cell = CellDependence::from_records(&monotone_cell(vec![])).expect("cell");
        assert_eq!(cell.n_used, MIN_TRIALS);
        assert_eq!(cell.n_total, MIN_TRIALS);
        // `distance_consistency` is absent from every fixture row: not a label.
        assert!(!cell.labels.iter().any(|l| l == "dsc"), "{:?}", cell.labels);
        assert_eq!(cell.labels.len(), 8);
        assert_eq!(cell.rho.len(), 8);
    }

    #[test]
    fn a_trial_missing_one_metric_is_dropped_whole() {
        // A diverged trial: confident trustworthiness, null stress.
        let diverged = trial(&[
            ("trustworthiness", Some(0.93)),
            ("continuity", Some(0.9)),
            ("normalized_stress", None),
            ("shepard_goodness", Some(0.5)),
            ("neighborhood_hit", Some(0.5)),
            ("dunn_index", Some(0.1)),
            ("davies_bouldin_ratio", Some(0.2)),
            ("cluster_density_measure", Some(0.3)),
        ]);
        let cell = CellDependence::from_records(&monotone_cell(vec![diverged])).expect("cell");
        assert_eq!(cell.n_total, MIN_TRIALS + 1);
        assert_eq!(cell.n_used, MIN_TRIALS);
    }

    #[test]
    fn a_cell_below_the_floor_does_not_count() {
        let rows: Vec<TrialRecord> = monotone_cell(vec![])
            .into_iter()
            .take(MIN_TRIALS - 1)
            .collect();
        assert!(CellDependence::from_records(&rows).is_none());
    }

    #[test]
    fn stress_is_oriented_so_a_positive_rho_means_agreement() {
        let cell = CellDependence::from_records(&monotone_cell(vec![])).expect("cell");
        let trust = cell.labels.iter().position(|l| l == "trust").unwrap();
        let stress = cell.labels.iter().position(|l| l == "1-stress").unwrap();
        let shep = cell.labels.iter().position(|l| l == "shep").unwrap();
        // Raw stress rises with trustworthiness, so oriented `1-stress` falls:
        // the two *disagree*, and the sign has to say so.
        assert!((cell.rho[trust][stress].unwrap() + 1.0).abs() < 1e-12);
        // Shepard falls with trustworthiness too, and is not flipped.
        assert!((cell.rho[trust][shep].unwrap() + 1.0).abs() < 1e-12);
        // ...so 1-stress and shepard agree perfectly.
        assert!((cell.rho[stress][shep].unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(cell.rho[trust][trust], Some(1.0));
    }

    #[test]
    fn panels_take_the_median_and_range_over_datasets_and_skip_other_settings() {
        // Dataset A: continuity = t², perfectly concordant with trust (ρ = 1).
        let a = monotone_cell(vec![]);
        // Dataset B: continuity reversed, ρ(trust, cont) = -1.
        let b: Vec<TrialRecord> = (0..MIN_TRIALS)
            .map(|i| {
                let t = count_to_f64(i) / count_to_f64(MIN_TRIALS);
                trial(&[
                    ("trustworthiness", Some(t)),
                    ("continuity", Some(1.0 - t)),
                    ("normalized_stress", Some(t)),
                    ("shepard_goodness", Some(1.0 - t)),
                    ("neighborhood_hit", Some((t * 7.0).sin())),
                    ("dunn_index", Some(t * 3.0)),
                    ("davies_bouldin_ratio", Some(1.0 / (t + 0.1))),
                    ("cluster_density_measure", Some((t * 3.0).cos())),
                ])
            })
            .collect();
        // Dataset C: a copy of A, so the median lands on a value and the
        // range still spans both signs.
        let c = a.clone();

        let mut cells = CellMap::new();
        cells.insert(Cell::new(SETTING, "a", 1000, "hyperbolic"), a.clone());
        cells.insert(Cell::new(SETTING, "b", 1000, "hyperbolic"), b);
        cells.insert(Cell::new(SETTING, "c", 1000, "hyperbolic"), c);
        // Wrong setting, wrong N: neither is on the panel.
        cells.insert(Cell::new("all_free", "a", 1000, "hyperbolic"), a.clone());
        cells.insert(Cell::new(SETTING, "a", 5000, "hyperbolic"), a.clone());
        // Too small to count.
        cells.insert(
            Cell::new(SETTING, "tiny", 1000, "hyperbolic"),
            a.iter().take(MIN_TRIALS - 1).cloned().collect(),
        );

        let panels = MetricDependence::panels(&cells, 1000);
        assert_eq!(panels.len(), 1, "only hyperbolic has cells at N=1000");
        let p = &panels[0];
        assert_eq!(p.geometry, "hyperbolic");
        assert_eq!(p.population(), (3, 3 * MIN_TRIALS, 3 * MIN_TRIALS));
        let trust = p.labels().iter().position(|l| l == "trust").unwrap();
        let cont = p.labels().iter().position(|l| l == "cont").unwrap();
        // Median of {1, -1, 1} is 1; the range spans both.
        assert!((p.median_at(trust, cont).unwrap() - 1.0).abs() < 1e-12);
        let (lo, hi) = p.range_at(trust, cont).unwrap();
        assert!((lo + 1.0).abs() < 1e-12 && (hi - 1.0).abs() < 1e-12);
        // Symmetric.
        assert_eq!(p.median_at(trust, cont), p.median_at(cont, trust));
    }

    #[test]
    fn diverging_color_runs_blue_white_vermillion() {
        assert_eq!(diverging_color(Some(0.0)), WHITE);
        assert_eq!(diverging_color(Some(-1.0)), OK_BLUE);
        assert_eq!(diverging_color(Some(1.0)), OK_VERMILLION);
        assert_eq!(diverging_color(None), OK_GREY);
        assert_eq!(diverging_color(Some(f64::NAN)), OK_GREY);
        // Half-way is half-way.
        let RGBColor(r, g, b) = diverging_color(Some(0.5));
        assert_eq!((r, g, b), (234, 175, 128));
    }

    #[test]
    fn category_axis_labels_each_slot_at_its_centre() {
        let labels = vec!["a".to_string(), "b".to_string()];
        let axis = CategoryAxis::new(&labels, false);
        assert_eq!(axis.key_points(2), vec![0.5, 1.5]);
        assert_eq!(axis.label(&0.5), "a");
        assert_eq!(axis.label(&1.5), "b");
        assert_eq!(axis.label(&2.5), "");
        assert_eq!(axis.label(&-0.5), "");
        let down = CategoryAxis::new(&labels, true);
        assert_eq!(down.range(), 2.0..0.0);
    }
}
