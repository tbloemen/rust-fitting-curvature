//! Experiment 2 (`metric-dependence`) — do the metrics order the same
//! visualisations the same way?
//!
//! [`MetricDependence`] is the figure this module draws: **a Spearman rank
//! correlation heatmap over the projected metrics, one figure per embedding
//! geometry.** It sits beside [`super::exp2::MetricTrend`] in the results
//! chapter and answers a different question about the same corpus: not how a
//! metric responds to curvature, but whether an "improvement on several
//! metrics" is several findings or one. Correlation is evidence of similar
//! rankings, not proof of redundancy, and the thesis text says so; the figure
//! only has to make the rankings comparable.
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
//! The same as [`super::exp2::MetricTrend`]'s, on purpose: **every trial of
//! the `all_off` cells** ([`super::exp2::SETTING`]) — all trials, not front
//! points, because the question is how the metrics rank the searched corpus,
//! not what the search kept; and the baseline setting only, so the auxiliary
//! loss weights do not vary underneath the ranking.
//!
//! Collapsed embeddings are **not** filtered out. They are finite readings of
//! a real trial — 14% of the hyperbolic corpus (`crates/analysis/CLAUDE.md`
//! § *κ floor*) — and every metric reads them as bad at once, which pulls each
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
//! is the bug `crates/analysis/CLAUDE.md` § *Orientation* records. Keeping the
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
//! matrix — today that is `distance_consistency`, which no `obj10` sweep
//! measured — and then every trial missing *any* remaining metric is dropped
//! whole. One population per matrix, which the panel reports as
//! `n = used / total` summed over its datasets. A diverged trial, the kind that
//! writes a confident `trustworthiness` beside a `null` stress (CLAUDE.md
//! § *Diverged trials*), is therefore excluded outright rather than ranked on
//! the metrics it happened to write. A cell needs [`MIN_TRIALS`] complete
//! trials to contribute; below that its ρ would be noise, and the panel's
//! dataset count excludes it.
//!
//! ### Layout
//!
//! Symmetric, full matrix, diagonal included — the conventional reading, and
//! the `1.00` on the diagonal is the scale's own reference. Cell fill is a
//! diverging blue–white–vermillion ramp ([`rho_color`]) and the number is
//! printed in every cell, so the figure reads without its colourbar; the
//! colourbar is still written, once per run, as its own file
//! ([`DependenceColorbar`]) so a caption can set it under the row. Axis labels
//! are the registry abbreviations ([`super::exp2::short_label`]) — nine wire
//! names do not fit along one axis. The geometry and the population line are
//! the only things drawn on the panel; N, the setting and the objective space
//! stay in the filename, as on every Exp 2 figure.

use plotters::coord::ranged1d::{DefaultFormatting, KeyPointHint, Ranged};
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use fitting_core::cast::{count_to_f64, to_i32};
use fitting_core::metrics::{Metric, Space, ALL, OBJECTIVES};

use super::exp2::{reading, short_label, SETTING};
use super::{
    CellMap, Figure, LinearTicks, Res, GEOMETRIES, OK_BLACK, OK_BLUE, OK_GREY, OK_VERMILLION,
};
use crate::records::TrialRecord;
use crate::stats::{median, spearman_matrix};
use crate::style_mesh;

/// Complete trials a dataset cell needs before its ρ counts. A cell is ~1000
/// trials, so this only bites on a cell that is nearly all diverged, where a
/// correlation over the survivors would describe the survivors.
pub const MIN_TRIALS: usize = 30;

/// Canvas of one heatmap: a `k × k` grid of cells wide enough for two lines of
/// text each, plus the label areas. Nine metrics at ~48 px per cell.
const PANEL: (u32, u32) = (560, 580);

/// Height of the strip above the chart carrying the geometry and the
/// population line.
const TITLE_STRIP: u32 = 40;

/// Canvas of the colourbar: as wide as a panel, one strip tall.
const COLORBAR: (u32, u32) = (PANEL.0, 64);

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
            rho: spearman_matrix(&columns),
            n_used,
            n_total: records.len(),
        })
    }
}

/// The per-geometry Spearman heatmap: the median over datasets of each
/// within-dataset ρ, and the range those datasets span.
pub struct MetricDependence {
    geometry: &'static str,
    n: usize,
    labels: Vec<String>,
    /// `median[i][j]`, over the datasets whose ρ is defined; `None` if none.
    median: Vec<Vec<Option<f64>>>,
    /// `(min, max)` over the same datasets.
    range: Vec<Vec<Option<(f64, f64)>>>,
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
                Self::combine(geometry, n, &per_dataset)
            })
            .collect()
    }

    /// Median and range of the per-dataset matrices, over the metrics every
    /// one of them carries. The label sets agree across the cells of one
    /// results directory (one objective space, one registry), so the
    /// intersection is a guard rather than a path anything takes today.
    fn combine(geometry: &'static str, n: usize, per_dataset: &[CellDependence]) -> Option<Self> {
        let first = per_dataset.first()?;
        let labels: Vec<String> = first
            .labels
            .iter()
            .filter(|l| per_dataset.iter().all(|c| c.labels.contains(l)))
            .cloned()
            .collect();
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
            }
        }
        Some(Self {
            geometry,
            n,
            labels,
            median: median_m,
            range: range_m,
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

    /// `(datasets, complete trials, trials)` behind the panel.
    #[must_use]
    pub fn population(&self) -> (usize, usize, usize) {
        (self.n_datasets, self.n_used, self.n_total)
    }
}

/// The fill for a correlation: `OK_BLUE` at −1, white at 0, `OK_VERMILLION`
/// at +1, mixed linearly in RGB; `OK_GREY` for an undefined ρ. Colourblind-safe
/// ends, and the same two hues the geometry palette uses for hyperbolic and
/// spherical — a reader who has seen those panels already knows them.
#[must_use]
pub fn rho_color(rho: Option<f64>) -> RGBColor {
    let Some(rho) = rho.filter(|r| r.is_finite()) else {
        return OK_GREY;
    };
    let t = rho.clamp(-1.0, 1.0);
    let end = if t < 0.0 { OK_BLUE } else { OK_VERMILLION };
    let a = t.abs();
    let mix = |c: u8| {
        let v = 255.0 + (f64::from(c) - 255.0) * a;
        // `v` is inside [c, 255] by construction.
        u8::try_from(to_i32(v.round())).unwrap_or(u8::MAX)
    };
    RGBColor(mix(end.0), mix(end.1), mix(end.2))
}

/// The text colour on a [`rho_color`] fill: white once the fill is saturated
/// enough that black would not read.
fn text_on(rho: Option<f64>) -> RGBColor {
    match rho {
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
struct CategoryAxis {
    labels: Vec<String>,
    range: std::ops::Range<f64>,
}

impl CategoryAxis {
    fn new(labels: &[String], top_down: bool) -> Self {
        let k = count_to_f64(labels.len());
        Self {
            labels: labels.to_vec(),
            range: if top_down { k..0.0 } else { 0.0..k },
        }
    }

    /// The name of the slot *v* falls in; empty off the axis. By reference
    /// because that is the signature `x_label_formatter` hands over.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn label(&self, v: &f64) -> String {
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

impl Figure for MetricDependence {
    fn name(&self) -> String {
        format!("exp2_dependence_{}_N{}", self.geometry, self.n)
    }

    fn size(&self) -> (u32, u32) {
        PANEL
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        // The title strip is drawn by hand rather than through `caption`,
        // which is one line: the geometry is what tells two panels apart, and
        // the population line is what the results chapter asks every panel to
        // state.
        let (title, plot) = root.split_vertically(TITLE_STRIP);
        let (width, _) = title.dim_in_pixel();
        let centre = to_i32(f64::from(width) / 2.0);
        title.draw(&Text::new(
            self.geometry,
            (centre, 12),
            ("sans-serif", 15)
                .into_font()
                .style(FontStyle::Bold)
                .color(&OK_BLACK)
                .pos(Pos::new(HPos::Center, VPos::Center)),
        ))?;

        let x_axis = CategoryAxis::new(&self.labels, false);
        let y_axis = CategoryAxis::new(&self.labels, true);
        let mut chart = ChartBuilder::on(&plot)
            .margin(6)
            .margin_right(14)
            .x_label_area_size(30)
            .y_label_area_size(64)
            .build_cartesian_2d(x_axis.clone(), y_axis.clone())?;
        style_mesh!(chart.configure_mesh())
            .disable_mesh()
            .x_label_formatter(&|v| x_axis.label(v))
            .y_label_formatter(&|v| y_axis.label(v))
            .draw()?;

        let k = self.labels.len();
        for i in 0..k {
            for j in 0..k {
                let rho = self.median[i][j];
                let (x0, x1) = (count_to_f64(j), count_to_f64(j) + 1.0);
                let (y0, y1) = (count_to_f64(i), count_to_f64(i) + 1.0);
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x0, y0), (x1, y1)],
                    rho_color(rho).filled(),
                )))?;
                // A thin white seam between cells, so the grid reads without
                // mesh lines.
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x0, y0), (x1, y1)],
                    WHITE.stroke_width(1),
                )))?;

                let ink = text_on(rho);
                let value = rho.map_or_else(|| "n/a".to_string(), |r| format!("{r:.2}"));
                let centre = (x0 + 0.5, y0 + 0.5);
                chart.draw_series(std::iter::once(
                    EmptyElement::at(centre)
                        + Text::new(
                            value,
                            (0, 0),
                            ("sans-serif", 12)
                                .into_font()
                                .color(&ink)
                                .pos(Pos::new(HPos::Center, VPos::Center)),
                        ),
                ))?;
            }
        }
        Ok(())
    }
}

/// The ρ scale every [`MetricDependence`] panel is filled on, as its own file
/// — one per run, since the ramp is fixed at `[-1, 1]` for every geometry and
/// N.
pub struct DependenceColorbar;

impl Figure for DependenceColorbar {
    fn name(&self) -> String {
        "exp2_dependence_colorbar".to_string()
    }

    fn size(&self) -> (u32, u32) {
        COLORBAR
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        const STEPS: usize = 200;

        let ticks = LinearTicks::new((-1.0, 1.0), 5);
        let mut chart = ChartBuilder::on(root)
            .margin(4)
            .margin_left(30)
            .margin_right(30)
            .x_label_area_size(38)
            .y_label_area_size(0)
            .build_cartesian_2d(ticks.clone(), 0.0f64..1.0f64)?;
        style_mesh!(chart.configure_mesh())
            .disable_mesh()
            .disable_y_axis()
            .x_desc("Spearman ρ")
            .x_label_formatter(&|v| ticks.label(v))
            .draw()?;
        let step = 2.0 / count_to_f64(STEPS);
        chart.draw_series((0..STEPS).map(|s| {
            let x0 = -1.0 + count_to_f64(s) * step;
            let mid = x0 + step / 2.0;
            Rectangle::new([(x0, 0.0), (x0 + step, 1.0)], rho_color(Some(mid)).filled())
        }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;

    /// One trial line with the given projected readings; anything not named
    /// is absent, as an `obj10` sweep leaves `distance_consistency`.
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
        assert_eq!(p.name(), "exp2_dependence_hyperbolic_N1000");
    }

    #[test]
    fn rho_color_runs_blue_white_vermillion() {
        assert_eq!(rho_color(Some(0.0)), WHITE);
        assert_eq!(rho_color(Some(-1.0)), OK_BLUE);
        assert_eq!(rho_color(Some(1.0)), OK_VERMILLION);
        assert_eq!(rho_color(None), OK_GREY);
        assert_eq!(rho_color(Some(f64::NAN)), OK_GREY);
        // Half-way is half-way.
        let RGBColor(r, g, b) = rho_color(Some(0.5));
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
