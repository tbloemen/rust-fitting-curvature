//! Experiment 2 (`metric-results`) — what effect does curvature have on the
//! established visualisation metrics?
//!
//! [`MetricTrend`] is the figure this module draws: **curvature on x, the
//! metric reading on y, every metric bounded in `[0, 1]` overlaid on one pair
//! of axes, one figure per embedding geometry.** Two figures per x axis, not
//! three — see *No Euclidean panel* below.
//!
//! ### Two x axes
//!
//! "Curvature" is two different numbers here, and the figure is drawn once
//! against each ([`XAxis`]):
//!
//! * **`|K|`**, the `curvature_magnitude` hyperparameter — what the optimiser
//!   *chose*, log-searched over `1e-6..5` (`config/params.json`);
//! * **`κ = |K|·R_rms²`**, the dimensionless embedding curvature — what the
//!   embedding *ended up at* once its spread is folded in
//!   (`TrialRecord::kappa`, thesis `@eq:kappa`).
//!
//! Neither reads for the other. κ is the quantity the thesis argues is
//! comparable across embeddings; `|K|` is the knob a practitioner turns. And on
//! the sphere κ is blind to `|K|` altogether — `crates/analysis/CLAUDE.md`
//! § *One κ, one gauge* — so the `|K|` panel is the only one of the two on
//! which the spherical trials actually move. The two share every other choice
//! below and a filename that differs only in its slug, `metric_vs_kappa` or
//! `metric_vs_curvature`.
//!
//! ### What is drawn, and from which trials
//!
//! The crate's own rule is that a panel must say what population it draws from,
//! because a front-point distribution describes the searched corpus rather than
//! an unbiased sample. This one draws **every trial of the `all_off` cells** at
//! the given N and geometry, pooled over all datasets:
//!
//! * *all trials*, not front points — the question is how a metric responds to
//!   κ, not what the search chose to keep;
//! * *`all_off` only*, so the auxiliary loss weights are pinned at zero and do
//!   not vary along with κ. Pooling every setting would mix two effects.
//!
//! Each curve is the **median of its metric in x bins**, drawn only where a bin
//! holds at least [`MIN_PER_BIN`] trials. Every curve is binned on one shared
//! set of edges ([`super::BinScale::edges`] once, then
//! [`super::binned_median_on`] per metric), so the overlaid lines are sampled at
//! the same x positions and can be read against each other — which is the whole
//! reason they share a panel.
//!
//! ### Two axis scales, two files
//!
//! The bins and the axis are one choice ([`super::BinScale`]), because a
//! geometric bin centre drawn on a linear axis lands in the wrong place. A
//! geometry whose x spans a decade or more is rendered **twice**: once
//! logarithmic, which is the natural reading of a quantity covering seven
//! decades, and once linear, which shows where the trials actually sit — 64% of
//! the hyperbolic corpus is below κ = 1.8, and the log axis deliberately
//! flattens that. Neither substitutes for the other, so both are written and
//! the linear one carries a `_linear` in its filename.
//!
//! Below a decade there is only the linear rendering, and it carries no suffix:
//! it is that geometry's only figure, and a suffix would imply a log companion
//! that does not exist. That is the spherical κ panel — plotters finds no key
//! points inside a sub-decade log range and drew no x ticks at all. The
//! spherical `|K|` panel spans the searched decades like the hyperbolic ones
//! and gets the pair; the rule is data-driven, not per geometry.
//!
//! ### Which metrics count as "0 to 1"
//!
//! Exactly [`super::OBJECTIVES`]: `QualityMetric::is_objective` *means* bounded
//! in `[0, 1]`, and the registry is the only place that fact lives. The five
//! `_manifold` twins are equally bounded and are left out **on purpose** — the
//! thesis judges the 2-D visualisation, and `crates/analysis/CLAUDE.md` records
//! the figure that was wrong for exactly this reason, putting a manifold and a
//! projected reading of one metric on a single axis. Adding them is a change to
//! the list this module iterates, and nothing else.
//!
//! Orientation comes off the registry too: `normalized_stress` is the one
//! minimised metric, so it is drawn as `1 - stress` and labelled that way.
//!
//! A metric with no column in the loaded sweeps is dropped from the panel and
//! from the legend rather than drawn at zero. That matters today: every file
//! under `results/` is a legacy `obj10` sweep with no `distance_consistency`,
//! and `objectives::oriented` — which maps *absent* to the worst case, correctly,
//! for the R2 indicator — would draw it as a flat line along the bottom. Hence
//! [`reading`] rather than `oriented`.
//!
//! ### No Euclidean panel
//!
//! `κ = |K|·R_rms²` and Euclidean space has `K = 0` exactly, so every Euclidean
//! trial sits at `κ = 0`: one column of points, no axis, nothing to read. The
//! panel is not drawn. A flat reference, if one is wanted, is a different figure
//! — the same metrics' *distributions* at κ = 0 — not this one with another
//! argument.
//!
//! The spherical κ panel **is** drawn, and its κ window is a factor of ~1.4
//! rather than the hyperbolic seven decades. That is not a plotting failure: it
//! is the pole mismatch between `Sphere::center` and `lift_pca_to_manifold`
//! documented in `crates/analysis/CLAUDE.md` § *One κ, one gauge*, which pins
//! spherical κ at `mean(θ²)` and makes it blind to `|K|`. Reading it beside the
//! spherical `|K|` panel is how that shows.
//!
//! ### Layout
//!
//! Each panel is half of Exp 1's canvas ([`PANEL`]), so two of them occupy the
//! width one Exp 1 figure does and sit side by side at the A4 text width. No
//! panel carries the legend: it is its own file, [`MetricLegend`], a portrait
//! canvas exactly [`PANEL`] tall, meant to be set at the **left** of a pair so
//! the row reads legend, hyperbolic, spherical. Every canvas in the row is the
//! same height, so the pair lines up without any alignment trick. One legend
//! serves both x axes — the curves are the same metrics in the same colours
//! whichever way the trials are binned — so it carries no axis slug.
//!
//! ### Not this module's other two figures
//!
//! [`MetricPanels`] below is still a skeleton. The results chapter asks for two
//! further figures under `<metric-results>` — a dataset-by-metric panel grouped
//! by metric family over synthetic *and* real datasets, and the Spearman
//! metric-dependence heatmap of `<metric-dependence>`, asking whether the
//! metrics order the same visualisations the same way. Neither is drawn yet.

use plotters::coord::Shift;
use plotters::prelude::*;

use fitting_core::metrics::Metric;

use super::{
    binned_median_on, draw_legend_grid, log_tick, metric_color, metric_dash, padded_log_range,
    padded_range, BinScale, CellMap, Figure, LegendEntry, LinearTicks, ObjectiveSpace, Res, CURVED,
    OBJECTIVES, OK_BLACK,
};
use crate::objectives::is_minimized_metric;
use crate::records::TrialRecord;
use crate::style_mesh;

/// The loss-weight setting the panels draw from: the baseline, where every
/// auxiliary weight is zero, so curvature is the only thing varying with the
/// metric.
const SETTING: &str = "all_off";

/// Bins across the panel's whole range, spaced as the axis is.
const N_BINS: usize = 30;

/// Which curvature a [`MetricTrend`] panel puts on x — see *Two x axes* in the
/// module doc. It decides the value read off each trial, the axis title and
/// the filename slug, and nothing else: binning, scale choice and drawing are
/// the same code for both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XAxis {
    /// `κ = |K|·R_rms²`, the embedding curvature ([`TrialRecord::kappa`]).
    Kappa,
    /// `|K|`, the `curvature_magnitude` hyperparameter the search chose.
    Curvature,
}

impl XAxis {
    /// The x value of one trial, or `None` where the record does not carry it.
    fn value(self, record: &TrialRecord) -> Option<f64> {
        match self {
            Self::Kappa => record.kappa(),
            // The same `|curvature|` fallback `kappa()` uses. Only Euclidean
            // sweeps omit `curvature_magnitude`, and Euclidean has no panel
            // here, so it is robustness rather than a path any panel takes.
            Self::Curvature => record
                .curvature_magnitude
                .or_else(|| record.curvature.map(f64::abs)),
        }
    }

    /// The axis title. Greek resolves in both backends; arrows and geometric
    /// shapes do not, so the orientation is stated in words on the y axis, and
    /// the magnitude bars are ASCII.
    fn desc(self) -> &'static str {
        match self {
            Self::Kappa => "κ",
            Self::Curvature => "|K|",
        }
    }

    /// The filename slug: `exp2_metric_vs_<tag>_…`.
    fn tag(self) -> &'static str {
        match self {
            Self::Kappa => "kappa",
            Self::Curvature => "curvature",
        }
    }
}

/// Trials a bin needs before its median is drawn. The pooled corpus is ~9,000
/// trials per (geometry, N) over 24 bins, so a bin at this floor is already a
/// sparse one — which on a log axis is exactly the tail, where a median over a
/// handful of points flips on one of them.
const MIN_PER_BIN: usize = 20;

/// Half of `exp1::MatchedGain::size()` (740 x 450), so two of these occupy the
/// width one Exp 1 figure does and the pair sits side by side at the A4 text
/// width.
///
/// plotters font sizes are absolute canvas units, so halving the canvas does
/// *not* halve the text: at ~80 mm on the page `style_mesh!`'s 13 px tick
/// labels land at roughly 8 pt, against Exp 1's ~6 pt. That is what a
/// half-width figure wants — it is read at half the size.
const PANEL: (u32, u32) = (370, 300);

/// Width of the [`MetricLegend`] canvas. One column of the metrics' full wire
/// names at the 12 px [`draw_legend_grid`] font: `1-normalized_stress` is the
/// longest at ~125 px of text, plus the 18 px swatch and its gutters. Narrower
/// than [`PANEL`] is tall, so the canvas is portrait.
const LEGEND_WIDTH: u32 = 130;

/// Height of one legend row. [`draw_legend_grid`] divides whatever area it is
/// given evenly between its rows, so the legend hands it a block of exactly
/// this many pixels per entry and centres that block on the canvas, rather
/// than letting five entries drift apart over [`PANEL`]'s full height.
const LEGEND_ROW: u32 = 20;

/// One metric's trend curve.
struct Series {
    metric: Metric,
    label: String,
    color: RGBColor,
    dash: Option<(i32, i32)>,
    /// `(x, median reading)` at each bin centre that cleared [`MIN_PER_BIN`].
    points: Vec<(f64, f64)>,
}

/// Every `[0, 1]`-bounded metric against one curvature ([`XAxis`]), for one
/// embedding geometry.
pub struct MetricTrend {
    x: XAxis,
    geometry: &'static str,
    n: usize,
    series: Vec<Series>,
    x_range: (f64, f64),
    /// How the x axis is spaced, and how the bins behind `series` were laid
    /// down — the two are the same choice, so they are one field.
    scale: BinScale,
    /// Whether this is the linear rendering of a geometry whose natural axis is
    /// logarithmic, which is the only thing the filename has to distinguish.
    alternate: bool,
}

impl MetricTrend {
    /// One panel per curved geometry, in [`CURVED`] order, with *x* on the x
    /// axis; a geometry with no binnable trials is absent from the result
    /// rather than returned empty, the same contract `exp4::R2Bars::panels`
    /// has.
    #[must_use]
    pub fn panels(cells: &CellMap, n: usize, x: XAxis) -> Vec<MetricTrend> {
        let mut out: Vec<MetricTrend> = Vec::new();
        for geometry in CURVED {
            // A trial without an x value is not a point on this figure at
            // all. The `> 0` is the log axis' requirement, not a quality
            // filter: κ = 0 is Euclidean, which has no panel here, and |K| is
            // searched from 1e-6 up.
            let kept: Vec<(f64, &TrialRecord)> = cells
                .iter()
                .filter(|(cell, _)| {
                    cell.setting == SETTING && cell.n == n && cell.geometry == geometry
                })
                .flat_map(|(_, records)| records.iter())
                .filter_map(|r| {
                    x.value(r)
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .map(|v| (v, r))
                })
                .collect();

            let xs: Vec<f64> = kept.iter().map(|(v, _)| *v).collect();
            // **A log axis needs a decade to label.** plotters derives a log
            // scale's key points from its endpoints and finds none inside a
            // window narrower than one decade, so the spherical κ panel —
            // pinned to a factor of ~1.4 by the wrong-pole gauge — came out
            // with no x ticks at all. Below a decade the axis is linear, which
            // is also the honest rendering: nothing about that window is
            // multiplicative.
            let natural = match padded_log_range(&xs, 0.0) {
                Some((lo, hi)) if hi / lo >= 10.0 => BinScale::Log,
                _ => BinScale::Linear,
            };

            // Where the natural axis is logarithmic, the linear rendering of
            // the same trials is drawn as well, as a second file. Seven decades
            // of x compressed onto equal-width bins is a different reading of
            // the same corpus — it shows where the trials actually *are*, which
            // the log axis deliberately flattens — and neither is a substitute
            // for the other.
            let scales: &[BinScale] = match natural {
                BinScale::Log => &[BinScale::Log, BinScale::Linear],
                BinScale::Linear => &[BinScale::Linear],
            };
            for &scale in scales {
                if let Some(panel) =
                    Self::panel(x, geometry, n, &kept, &xs, scale, scale != natural)
                {
                    out.push(panel);
                }
            }
        }
        out
    }

    /// One panel at one axis scale, or `None` when nothing bins.
    fn panel(
        x: XAxis,
        geometry: &'static str,
        n: usize,
        kept: &[(f64, &TrialRecord)],
        xs: &[f64],
        scale: BinScale,
        alternate: bool,
    ) -> Option<MetricTrend> {
        let edges = scale.edges(xs, N_BINS)?;
        let series: Vec<Series> = OBJECTIVES
            .iter()
            .filter_map(|&metric| Series::build(metric, &edges, scale, kept))
            .collect();
        if series.is_empty() {
            return None;
        }

        // The axis spans the *drawn* points, not the trials: a curve stops at
        // the last bin that cleared MIN_PER_BIN, and the sparse tail beyond it
        // — there are trials there, just too few per bin to take a median of
        // — would otherwise read as empty axis. Padded a little in the axis'
        // own metric so the end points do not sit on the frame; not snapped to
        // whole decades, which would reopen exactly that gap. plotters puts a
        // log axis' ticks on powers of ten whatever the endpoints are.
        let drawn: Vec<f64> = series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.0))
            .collect();
        let span = (edges[0], edges[edges.len() - 1]);
        let x_range = match scale {
            BinScale::Log => padded_log_range(&drawn, 0.03).unwrap_or(span),
            BinScale::Linear => padded_range(&drawn, 0.03).unwrap_or(span),
        };

        Some(MetricTrend {
            x,
            geometry,
            n,
            series,
            x_range,
            scale,
            alternate,
        })
    }

    /// Always true for a panel [`MetricTrend::panels`] returned; kept so the
    /// driver reads the same as every other figure's.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.series.is_empty()
    }
}

impl Series {
    /// One metric's binned trend, or `None` when the sweeps do not carry it or
    /// no bin cleared [`MIN_PER_BIN`].
    fn build(
        metric: Metric,
        edges: &[f64],
        scale: BinScale,
        kept: &[(f64, &TrialRecord)],
    ) -> Option<Series> {
        let (xs, ys): (Vec<f64>, Vec<f64>) = kept
            .iter()
            .filter_map(|(x, r)| reading(metric, r).map(|v| (*x, v)))
            .unzip();
        let (centres, medians) = binned_median_on(edges, scale, &xs, &ys, MIN_PER_BIN);
        if centres.is_empty() {
            return None;
        }
        Some(Series {
            metric,
            label: label(metric),
            color: metric_color(metric.name()),
            dash: metric_dash(metric.name()),
            points: centres.into_iter().zip(medians).collect(),
        })
    }

    fn legend_entry(&self) -> LegendEntry {
        let entry = LegendEntry::new(self.label.clone(), self.color);
        match self.dash {
            Some((dash, gap)) => entry.with_dash(dash, gap),
            None => entry,
        }
    }
}

/// One metric's reading off a trial, oriented so higher is better.
///
/// **Not `objectives::oriented`**, which substitutes the worst case for a
/// reading that is absent. That is right for the R2 indicator, where a trial
/// that did not measure an objective must score badly rather than vanish, and
/// wrong for a trend curve, where it would draw a metric the sweeps never wrote
/// as a flat line at zero. Here absent stays absent and the metric is dropped.
fn reading(metric: Metric, record: &TrialRecord) -> Option<f64> {
    let v = record.metrics.get(metric)?;
    if !v.is_finite() {
        return None;
    }
    Some(if is_minimized_metric(metric) {
        1.0 - v
    } else {
        v
    })
}

/// The series label: the metric's wire name, marked as flipped where the metric
/// is minimised. The full name rather than `Metric::short()` — a thesis figure
/// is read once and slowly, and `1-normalized_stress` says what `1-stress` only
/// implies. It is what sizes [`LEGEND_COLS`].
///
/// ASCII hyphen, not U+2212 — the bitmap backend renders anything outside
/// Latin-1 + Greek as tofu.
fn label(metric: Metric) -> String {
    if is_minimized_metric(metric) {
        format!("1-{}", metric.name())
    } else {
        metric.name().to_string()
    }
}

impl Figure for MetricTrend {
    fn name(&self) -> String {
        // Only the *alternate* rendering is marked. The spherical κ panel is
        // linear too, but it is that geometry's only figure on that axis — a
        // suffix there would imply a log companion that does not exist.
        let axis = if self.alternate { "_linear" } else { "" };
        format!(
            "exp2_metric_vs_{}_{}{axis}_N{}",
            self.x.tag(),
            self.geometry,
            self.n
        )
    }

    fn size(&self) -> (u32, u32) {
        PANEL
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        // The geometry is the one identifying thing drawn: two of these are
        // read side by side and have to be tellable apart. N, setting and
        // objective space stay in the filename, as Exp 1's do. The legend is
        // [`MetricLegend`], a separate file.
        let mut builder = ChartBuilder::on(root);
        builder
            .margin(6)
            .margin_right(12)
            .caption(
                self.geometry,
                ("sans-serif", 14)
                    .into_font()
                    .style(FontStyle::Bold)
                    .color(&OK_BLACK),
            )
            .x_label_area_size(34)
            .y_label_area_size(46);

        let (lo, hi) = self.x_range;
        // The bins behind `series` were laid down on this same scale, so the
        // medians sit mid-bin as this axis renders them.
        if self.scale == BinScale::Log {
            let mut chart = builder.build_cartesian_2d((lo..hi).log_scale(), 0.0f64..1.0f64)?;
            style_mesh!(chart.configure_mesh())
                .x_desc(self.x.desc())
                .y_desc(Y_DESC)
                .x_label_formatter(&log_tick)
                // A hint, not a count: plotters walks decades on a log axis and
                // rounds to whole ones. Ten decades at this width put "0.0001"
                // next to "0.001", which collides once the figure is set at half
                // the A4 text width; asking for fewer thins them to every other
                // decade instead.
                .x_labels(5)
                .y_labels(5)
                .draw()?;
            self.draw_curves(&mut chart)?;
        } else {
            // Own ticks rather than plotters': its float walk drops the tick at
            // the right end of the axis — see `LinearTicks`.
            let ticks = LinearTicks::new((lo, hi), 4);
            let mut chart = builder.build_cartesian_2d(ticks.clone(), 0.0f64..1.0f64)?;
            style_mesh!(chart.configure_mesh())
                .x_desc(self.x.desc())
                .y_desc(Y_DESC)
                .x_label_formatter(&|v| ticks.label(v))
                .y_labels(5)
                .draw()?;
            self.draw_curves(&mut chart)?;
        }
        Ok(())
    }
}

/// Every series is oriented so higher is better — `normalized_stress` is drawn
/// as `1 - stress` — and the axis says so, because that is the one thing about
/// this figure a reader cannot recover from the curves.
const Y_DESC: &str = "metric";

impl MetricTrend {
    /// Draw every series onto *chart*. Generic over the x coordinate so the log
    /// and linear branches of [`MetricTrend::draw`] share one body.
    fn draw_curves<DB, X, Y>(&self, chart: &mut ChartContext<DB, Cartesian2d<X, Y>>) -> Res
    where
        DB: DrawingBackend,
        DB::ErrorType: 'static,
        X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
        Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    {
        for series in &self.series {
            let color = series.color;
            match series.dash {
                Some((dash, gap)) => {
                    chart.draw_series(DashedLineSeries::new(
                        series.points.iter().copied(),
                        dash,
                        gap,
                        color.stroke_width(2),
                    ))?;
                }
                None => {
                    chart.draw_series(LineSeries::new(
                        series.points.iter().copied(),
                        color.stroke_width(2),
                    ))?;
                }
            }
        }
        Ok(())
    }
}

/// The legend shared by every [`MetricTrend`] panel at one N, on both x axes,
/// as its own portrait file: [`LEGEND_WIDTH`] wide, [`PANEL`] tall, one entry
/// per row.
///
/// It is the union of the panels' series, in [`OBJECTIVES`] order — the order
/// the curves are drawn in — so a metric that bins on one geometry and not
/// another is still listed, and a metric the sweeps never wrote is not.
pub struct MetricLegend {
    n: usize,
    entries: Vec<LegendEntry>,
}

impl MetricLegend {
    /// The legend for *panels*, which should be everything
    /// [`MetricTrend::panels`] returned at *n*, over every [`XAxis`].
    #[must_use]
    pub fn from_panels(panels: &[MetricTrend], n: usize) -> Self {
        let entries = OBJECTIVES
            .iter()
            .filter_map(|&metric| {
                panels
                    .iter()
                    .flat_map(|p| p.series.iter())
                    .find(|s| s.metric == metric)
                    .map(Series::legend_entry)
            })
            .collect();
        Self { n, entries }
    }

    /// False when no panel had a series to name — in which case there are no
    /// panels either, and a legend on its own would label nothing.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.entries.is_empty()
    }
}

impl Figure for MetricLegend {
    fn name(&self) -> String {
        format!("exp2_metric_legend_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (LEGEND_WIDTH, PANEL.1)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        // A block of LEGEND_ROW per entry, centred vertically; the grid
        // helper would otherwise space five rows over the full panel height.
        let rows = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        let block = rows.saturating_mul(LEGEND_ROW).min(PANEL.1);
        let pad = (PANEL.1 - block) / 2;
        draw_legend_grid(&root.margin(pad, pad, 0, 0), &self.entries, 1)
    }
}

// ─── Skeletons ───────────────────────────────────────────────────────────────

/// Metric readings across the three embedding geometries, one panel per metric.
///
/// **A skeleton: nothing is drawn yet**, so [`MetricPanels::has_data`] returns
/// `false` and the driver's "figures with no data are skipped" rule keeps an
/// empty SVG off disk. See the module doc for what the results chapter asks of
/// it — it is not [`MetricTrend`], which answers a different question about
/// the same section.
pub struct MetricPanels<'a> {
    #[expect(dead_code, reason = "read once the figure is drawn")]
    cells: &'a CellMap,
    n: usize,
    #[expect(dead_code, reason = "read once the figure is drawn")]
    space: ObjectiveSpace,
}

impl<'a> MetricPanels<'a> {
    #[must_use]
    pub fn new(cells: &'a CellMap, n: usize, space: ObjectiveSpace) -> Self {
        Self { cells, n, space }
    }

    /// Always `false` while this is a skeleton.
    #[must_use]
    pub fn has_data(&self) -> bool {
        false
    }
}

impl Figure for MetricPanels<'_> {
    fn name(&self) -> String {
        format!("exp2_metric_panels_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (1500, 1000)
    }

    fn draw<DB: DrawingBackend>(&self, _root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        Ok(())
    }
}
