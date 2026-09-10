//! Experiment 4 (`ablation-results`) — do auxiliary global loss terms improve
//! the visualisations?
//!
//! Two figures over the loss-weight settings (`all_off`, `centering_only`,
//! `global_only`, `norm_only`, `all_free`), each comparison made within one
//! (dataset, geometry):
//!
//! * [`StackedFronts`] — the fronts themselves, one panel per (dataset,
//!   geometry), one curve per setting.
//! * [`R2Bars`] — the R2 levels of `results/r2_delta_*.jsonl` as grouped bars,
//!   one chart per (dataset, geometry, N), preference regions along x.
//!
//! ─── Stacked fronts ─────────────────────────────────────────────────────────
//!
//! Five fronts share a panel and they overlap heavily — on fashion-MNIST the
//! settings differ by 1–2% of the panel's y-span — so the rendering carries the
//! whole burden of keeping them apart:
//!
//! * fronts are drawn as the **staircases** they are ([`step_polyline`]),
//! * each setting gets its own **dash pattern** on top of its colour, so the
//!   figure survives greyscale printing and coincident curves stay traceable,
//! * strokes are **translucent** and markers **opaque**: overlapping curves
//!   blend into visible overlap instead of the last one drawn hiding the rest,
//!   while the front points themselves keep their colour and stay crisp,
//! * axes use [`robust_range`], because a single diverged front point used to
//!   set the scale and squash the informative knee into a sliver.
//!
//! ─── R2 bars ────────────────────────────────────────────────────────────────
//!
//! Same table `scripts/r2_delta_typst.py` renders, read the other way round: the
//! table is meant to be scanned *down* a column, this is meant to be read
//! *across* the priorities. x groups are the preference regions of
//! `@preference-regions`; the bars inside a group are the loss-weight settings,
//! so "which setting wins under this priority, and does the winner change with
//! the priority" is a single glance rather than a row-by-row comparison.
//!
//! Four choices worth stating:
//!
//! - **The level `r2`, not the gain `delta_r2`.** So `all_off` is a bar like any
//!   other setting rather than the zero line, and the regions are comparable
//!   against each other — `W_shep` is a costlier region to serve than `W_trust`
//!   whatever the setting does. The price is resolution: the settings of one
//!   region typically differ by well under a percent of the level, which is why
//!   every bar carries its value as a label.
//! - **The y axis starts at zero**, because a bar's length is its value. A
//!   truncated axis would turn those sub-percent differences into dramatic
//!   steps, which is the one thing this chart must not do.
//! - **R2 is a cost** (distance to the ideal point, @eq:r2), so a *shorter* bar
//!   is the better front. That runs opposite to every quality metric and the
//!   title says so.
//! - **Everything is scaled by [`SCALE`]**, exactly as the table is, so a chart
//!   and the table row it comes from carry the same digits.
//!
//! These go in their own `<out-dir>/experiment_4` subdirectory: 9 datasets x 3
//! geometries x 2 N is a lot of files to leave loose among the other figures.

use fitting_core::cast::{count_to_f64, to_i32};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::{
    draw_legend, finite_xy, robust_range, setting_color, CellMap, Figure, LegendEntry,
    ObjectiveSpace, Res, GEOMETRIES, OK_BLACK, REAL_DATASETS, SETTING_ORDER, X_LABEL, X_METRIC,
    Y_LABEL, Y_METRIC,
};
use crate::aggregate::{DeltaRow, BASELINE};
use crate::cell::Cell;
use crate::error::{Error, Result};
use crate::pareto::{slice_front_2d, step_polyline};
use crate::records::load_jsonl;
use crate::style_mesh;

// ─── Stacked fronts ───────────────────────────────────────────────────────────

/// `(dash, gap)` per setting, or `None` for the solid baseline. Redundant with
/// the colour on purpose — dashes are what survives a greyscale print.
fn setting_dash(setting: &str) -> Option<(i32, i32)> {
    match setting {
        BASELINE => None,
        "centering_only" => Some((14, 6)),
        "global_only" => Some((8, 5)),
        "norm_only" => Some((4, 4)),
        "all_free" => Some((2, 4)),
        _ => Some((6, 6)),
    }
}

/// Translucent enough that two curves on top of each other read as two.
const LINE_ALPHA: f64 = 0.6;
/// The baseline is the reference every other curve is read against, so it sits
/// a little more solid than the rest — and is drawn last.
const BASELINE_ALPHA: f64 = 0.85;

pub struct StackedFronts<'a> {
    cells: &'a CellMap,
    n: usize,
}

/// The per-panel front polyline for one setting, in plot coordinates.
struct Curve {
    setting: &'static str,
    points: Vec<(f64, f64)>,
}

impl Curve {
    /// The lowest-stress point: the end of the curve that must stay on-screen
    /// so the setting cannot vanish from its panel entirely.
    fn cheapest(&self) -> Option<(f64, f64)> {
        self.points
            .iter()
            .copied()
            .reduce(|a, b| if b.1 < a.1 { b } else { a })
    }
}

impl<'a> StackedFronts<'a> {
    #[must_use]
    pub fn new(cells: &'a CellMap, n: usize) -> Self {
        Self { cells, n }
    }

    fn curves(&self, dataset: &str, geometry: &str) -> Vec<Curve> {
        let mut out = Vec::new();
        for setting in SETTING_ORDER {
            let key = Cell::new(setting, dataset, self.n, geometry);
            let Some(recs) = self.cells.get(&key) else {
                continue;
            };
            let (x, y) = finite_xy(recs, X_METRIC, Y_METRIC);
            if x.len() < 3 {
                continue;
            }
            // Trustworthiness up, stress down.
            let idx = slice_front_2d(&x, &y, true, false);
            out.push(Curve {
                setting,
                points: idx.into_iter().map(|i| (x[i], y[i])).collect(),
            });
        }
        out
    }

    /// True when at least one panel has something to draw.
    #[must_use]
    pub fn has_data(&self) -> bool {
        REAL_DATASETS
            .iter()
            .any(|ds| GEOMETRIES.iter().any(|g| !self.curves(ds, g).is_empty()))
    }
}

impl Figure for StackedFronts<'_> {
    fn name(&self) -> String {
        format!("exp4_stacked_fronts_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (1500, 1750)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            &format!(
                "Experiment 4 — Pareto fronts by loss setting (N={}): trustworthiness vs normalised stress",
                self.n
            ),
            ("sans-serif", 22).into_font().color(&OK_BLACK),
        )?;
        let (legend, grid) = root.split_vertically(38);
        let entries: Vec<LegendEntry> = SETTING_ORDER
            .iter()
            .map(|setting| {
                let e = LegendEntry::new(*setting, setting_color(setting));
                match setting_dash(setting) {
                    Some((dash, gap)) => e.with_dash(dash, gap),
                    None => e,
                }
            })
            .collect();
        draw_legend(&legend, &entries)?;

        let panels = grid.split_evenly((REAL_DATASETS.len(), GEOMETRIES.len()));
        for (row, dataset) in REAL_DATASETS.iter().enumerate() {
            for (col, geometry) in GEOMETRIES.iter().enumerate() {
                let panel = &panels[row * GEOMETRIES.len() + col];
                let curves = self.curves(dataset, geometry);
                draw_panel(panel, &curves, dataset, geometry, row, col)?;
            }
        }
        Ok(())
    }
}

/// Draw a single (dataset, geometry) panel of the stacked fronts.
fn draw_panel<DB: DrawingBackend>(
    panel: &DrawingArea<DB, Shift>,
    curves: &[Curve],
    dataset: &str,
    geometry: &str,
    row: usize,
    col: usize,
) -> Res
where
    DB::ErrorType: 'static,
{
    let all_x: Vec<f64> = curves
        .iter()
        .flat_map(|c| c.points.iter().map(|p| p.0))
        .collect();
    let all_y: Vec<f64> = curves
        .iter()
        .flat_map(|c| c.points.iter().map(|p| p.1))
        .collect();
    let (mut x_lo, mut x_hi) = robust_range(&all_x, 0.06).unwrap_or((0.0, 1.0));
    let (mut y_lo, mut y_hi) = robust_range(&all_y, 0.06).unwrap_or((0.0, 1.0));
    // The fence is computed over the pooled points, so in principle
    // it could exclude a whole curve. Widen until every setting
    // keeps at least its cheapest-stress point.
    for point in curves.iter().filter_map(Curve::cheapest) {
        x_lo = x_lo.min(point.0);
        x_hi = x_hi.max(point.0);
        y_lo = y_lo.min(point.1);
        y_hi = y_hi.max(point.1);
    }

    let mut chart = ChartBuilder::on(panel)
        .margin(6)
        .margin_top(if row == 0 { 4 } else { 6 })
        .caption(
            if row == 0 { geometry } else { "" },
            ("sans-serif", 18).into_font().style(FontStyle::Bold),
        )
        .x_label_area_size(44)
        .y_label_area_size(70)
        .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

    let y_desc = if col == 0 {
        format!("{dataset}   {Y_LABEL}")
    } else {
        String::new()
    };
    let x_desc = if row == REAL_DATASETS.len() - 1 {
        X_LABEL
    } else {
        ""
    };
    // The mesh style is shared with every other figure; the label size is
    // overridden after the macro because this canvas is larger.
    style_mesh!(chart.configure_mesh())
        .label_style(("sans-serif", 15).into_font().color(&RGBColor(60, 60, 60)))
        .x_desc(x_desc)
        .y_desc(y_desc)
        .x_labels(5)
        .y_labels(5)
        .draw()?;

    let off_scale = draw_curves(&mut chart, curves, x_lo, x_hi, y_lo, y_hi)?;

    if off_scale > 0 {
        draw_off_scale_label(&mut chart, off_scale, x_lo, x_hi, y_lo, y_hi)?;
    }
    Ok(())
}

/// Label a panel whose robust range cut some front points, offset from the
/// corner the points left through.
fn draw_off_scale_label<DB, X, Y>(
    chart: &mut ChartContext<DB, Cartesian2d<X, Y>>,
    off_scale: usize,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
) -> Res
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
{
    // No glyphs: the bitmap backend has no arrows or geometric
    // shapes, and renders them as tofu.
    let label = if off_scale == 1 {
        "1 pt off-scale".to_string()
    } else {
        format!("{off_scale} pts off-scale")
    };
    chart.draw_series(std::iter::once(Text::new(
        label,
        (x_lo + (x_hi - x_lo) * 0.03, y_hi - (y_hi - y_lo) * 0.03),
        ("sans-serif", 13)
            .into_font()
            .color(&RGBColor(110, 110, 110))
            .pos(Pos::new(HPos::Left, VPos::Top)),
    )))?;
    Ok(())
}

/// Draw one setting's front on a chart. Returns how many front points fell
/// outside the robust range (marked at the edge they left through).
fn draw_curves<DB, X, Y>(
    chart: &mut ChartContext<DB, Cartesian2d<X, Y>>,
    curves: &[Curve],
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
) -> std::result::Result<usize, Box<dyn std::error::Error>>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
{
    // Baseline last, so the reference is never buried.
    let order = curves
        .iter()
        .filter(|c| c.setting != BASELINE)
        .chain(curves.iter().filter(|c| c.setting == BASELINE));
    let mut off_scale = 0usize;
    for curve in order {
        let color = setting_color(curve.setting);
        let steps = step_polyline(&curve.points);
        match setting_dash(curve.setting) {
            Some((dash, gap)) => {
                chart.draw_series(DashedLineSeries::new(
                    steps,
                    dash,
                    gap,
                    color.mix(LINE_ALPHA).stroke_width(2),
                ))?;
            }
            None => {
                chart.draw_series(LineSeries::new(
                    steps,
                    color.mix(BASELINE_ALPHA).stroke_width(3),
                ))?;
            }
        }
        // Opaque markers on the real front points only — never on
        // the staircase corners, which are not trials.
        chart.draw_series(
            curve
                .points
                .iter()
                .map(|point| Circle::new(*point, 3, color.filled())),
        )?;

        // Points the robust range cut: marked at the edge they left
        // through, so a clipped panel never looks complete.
        let (pad_x, pad_y) = ((x_hi - x_lo) * 0.02, (y_hi - y_lo) * 0.02);
        let clipped: Vec<(f64, f64)> = curve
            .points
            .iter()
            .filter(|(x, y)| *x < x_lo || *x > x_hi || *y < y_lo || *y > y_hi)
            .map(|(x, y)| {
                (
                    x.clamp(x_lo + pad_x, x_hi - pad_x),
                    y.clamp(y_lo + pad_y, y_hi - pad_y),
                )
            })
            .collect();
        off_scale += clipped.len();
        chart.draw_series(
            clipped
                .into_iter()
                .map(|point| TriangleMarker::new(point, 5, color.filled())),
        )?;
    }
    Ok(off_scale)
}

// ─── R2 bars ──────────────────────────────────────────────────────────────────

/// Every entry is multiplied by this, the way the Typst table multiplies it:
/// R2 sits around 0.1, so the unscaled axis would spend three of its four digits
/// on leading zeros.
const SCALE: f64 = 1000.0;

/// Settings that are not drawn.
///
/// `rms_anchored` fixes the curvature gauge for Experiment 3 rather than
/// ablating a loss term, and it exists for hyperbolic only — so it is not one of
/// the alternatives this chart compares, and its (often much larger) level sets
/// the y scale for bars it does not belong beside. The Typst table drops it for
/// the same reason.
const EXCLUDED: [&str; 1] = ["rms_anchored"];

/// Height reserved under the plot for the group labels.
const LABEL_AREA: i32 = 26;

/// Fraction of a group's width left empty at each end, so neighbouring groups'
/// bars do not touch.
const GROUP_PAD: f64 = 0.10;
/// Fraction of a bar's slot left empty, as the gap between bars of one group.
const BAR_GAP: f64 = 0.15;

/// Fraction of the tallest bar left clear above it, for the value labels.
const HEAD_ROOM: f64 = 0.16;

/// Load the R2 rows written by `r2 aggregate --deltas`.
///
/// An **absent** table is not an error: it is a separate `r2` run, and the bar
/// charts are simply not drawn without it. A table that is there and will not
/// parse still fails.
///
/// # Errors
///
/// Returns `Err` if the file is present but malformed. A missing file returns
/// an empty `Vec`.
pub fn load_deltas(path: &Path) -> Result<Vec<DeltaRow>> {
    match load_jsonl(path) {
        Ok(rows) => Ok(rows),
        Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(Vec::new())
        }
        Err(e) => Err(e),
    }
}

/// A level at the scale the axis is in: decimals drop as the magnitude grows,
/// the way the Typst table's `fixed` does it, so the labels stay three or four
/// significant figures wide.
fn fixed(scaled: f64) -> String {
    let decimals = if scaled.abs() < 10.0 {
        2
    } else {
        usize::from(scaled.abs() < 100.0)
    };
    format!("{scaled:.decimals$}")
}

/// One preference region's column of bars.
struct Group {
    label: String,
    /// R2 per setting, in [`R2Bars::settings`] order; `None` where that setting
    /// has no row for this region (an incomplete run, not a zero).
    values: Vec<Option<f64>>,
}

/// The R2 levels of one (dataset, geometry) cell block at one N.
pub struct R2Bars {
    dataset: String,
    geometry: String,
    n: usize,
    /// Bar order within every group.
    settings: Vec<String>,
    groups: Vec<Group>,
}

impl R2Bars {
    /// One chart per (dataset, geometry) present at this N.
    #[must_use]
    pub fn panels(rows: &[DeltaRow], n: usize, space: ObjectiveSpace) -> Vec<R2Bars> {
        // (dataset, geometry) → setting → region → row.
        type Block<'a> = BTreeMap<&'a str, BTreeMap<&'a str, &'a DeltaRow>>;
        let mut blocks: BTreeMap<(&str, &str), Block> = BTreeMap::new();
        for r in rows.iter().filter(|r| r.n == n) {
            blocks
                .entry((&r.dataset, &r.geometry))
                .or_default()
                .entry(&r.setting)
                .or_default()
                .insert(&r.region, r);
        }

        blocks
            .into_iter()
            .map(|((dataset, geometry), by_setting)| {
                let settings = bar_order(&by_setting);
                let groups = crate::r2::region_labels(space)
                    .into_iter()
                    .filter_map(|(region, label)| {
                        let values: Vec<Option<f64>> = settings
                            .iter()
                            .map(|s| {
                                by_setting[s.as_str()]
                                    .get(region.as_str())
                                    .map(|r| r.r2)
                                    .filter(|v| v.is_finite())
                            })
                            .collect();
                        if values.iter().all(Option::is_none) {
                            return None;
                        }
                        Some(Group { label, values })
                    })
                    .collect();
                R2Bars {
                    dataset: dataset.to_string(),
                    geometry: geometry.to_string(),
                    n,
                    settings,
                    groups,
                }
            })
            .filter(R2Bars::has_data)
            .collect()
    }

    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.settings.is_empty() && !self.groups.is_empty()
    }

    /// The tallest bar, scaled; `0.0` when there is nothing to draw.
    fn scaled_max(&self) -> f64 {
        self.groups
            .iter()
            .flat_map(|g| g.values.iter().flatten().map(|v| v * SCALE))
            .fold(0.0f64, f64::max)
    }
}

/// The settings to draw, in [`SETTING_ORDER`], with anything present but
/// unlisted appended rather than dropped — except [`EXCLUDED`].
fn bar_order(by_setting: &BTreeMap<&str, BTreeMap<&str, &DeltaRow>>) -> Vec<String> {
    let present: BTreeSet<&str> = by_setting
        .keys()
        .copied()
        .filter(|s| !EXCLUDED.contains(s))
        .collect();
    let mut out: Vec<String> = SETTING_ORDER
        .iter()
        .filter(|s| present.contains(*s))
        .map(|s| (*s).to_string())
        .collect();
    out.extend(
        present
            .iter()
            .filter(|s| !SETTING_ORDER.contains(s))
            .map(|s| (*s).to_string()),
    );
    out
}

impl Figure for R2Bars {
    fn name(&self) -> String {
        format!("r2_bars_{}_{}_N{}", self.dataset, self.geometry, self.n)
    }

    fn size(&self) -> (u32, u32) {
        // Wide enough that a bar can carry its own value: the settings of one
        // region differ in the third digit, so the labels are the chart's
        // resolution and they must not collide.
        (
            200 + 150 * u32::try_from(self.groups.len()).expect("a small number of groups"),
            520,
        )
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            &format!(
                "R2 indicator by preference region — {} / {}, N={} (a cost: shorter is better)",
                self.dataset, self.geometry, self.n
            ),
            ("sans-serif", 18).into_font().color(&OK_BLACK),
        )?;
        let (legend, body) = root.split_vertically(32);

        let entries: Vec<LegendEntry> = self
            .settings
            .iter()
            .map(|s| LegendEntry::new(s.clone(), setting_color(s)))
            .collect();
        draw_legend(&legend, &entries)?;

        // The group labels are drawn by hand rather than as x tick labels: the
        // x axis is continuous (a group is the interval [i, i+1], so the bars
        // inside it can be placed at any fraction), and plotters picks its own
        // key points on a continuous axis — they would not land on the group
        // centres. Splitting the strip off first keeps the plot's own geometry
        // untouched.
        let (plot_area, label_area) = body.split_vertically(
            i32::try_from(body.dim_in_pixel().1).unwrap_or(i32::MAX) - LABEL_AREA,
        );

        let n_groups = self.groups.len();
        // Zero-based, always: the bar's length is the value it reports.
        let y_hi = self.scaled_max() * (1.0 + HEAD_ROOM);
        let y_hi = if y_hi > 0.0 { y_hi } else { 1.0 };

        let mut chart = ChartBuilder::on(&plot_area)
            .margin(10)
            .x_label_area_size(0)
            .y_label_area_size(64)
            .build_cartesian_2d(0f64..count_to_f64(n_groups), 0f64..y_hi)?;

        style_mesh!(chart.configure_mesh())
            .disable_x_mesh()
            .disable_x_axis()
            .y_desc("R2 (units of 1e-3)")
            .draw()?;

        // Alternating bands, so a bar is read against its own group rather than
        // against its neighbour across a group boundary.
        chart.draw_series((0..n_groups).filter(|g| g % 2 == 1).map(|g| {
            Rectangle::new(
                [(count_to_f64(g), 0.0), (count_to_f64(g) + 1.0, y_hi)],
                RGBColor(246, 246, 246).filled(),
            )
        }))?;

        let slot = (1.0 - 2.0 * GROUP_PAD) / count_to_f64(self.settings.len());
        for (i, setting) in self.settings.iter().enumerate() {
            let color = setting_color(setting);
            let bars: Vec<(f64, f64, f64)> = self
                .groups
                .iter()
                .enumerate()
                .filter_map(|(g, group)| {
                    let value = group.values[i]? * SCALE;
                    let x0 =
                        count_to_f64(g) + GROUP_PAD + count_to_f64(i) * slot + slot * BAR_GAP / 2.0;
                    Some((x0, x0 + slot * (1.0 - BAR_GAP), value))
                })
                .collect();

            chart.draw_series(bars.iter().map(|(x0, x1, v)| {
                Rectangle::new([(*x0, 0.0), (*x1, *v)], color.mix(0.9).filled())
            }))?;

            // The value on top of its bar, reading upwards. Horizontal text at
            // this bar width would overlap its neighbour, and the differences
            // the chart is read for live in the last digit.
            //
            // The anchor is stated in the *text's* frame whatever the rotation
            // (plotters' `text_anchor` doc), and `Rotate270` maps the text's
            // left edge to the bottom of the strip on screen — so `Left` is the
            // end that sits on the bar and `Center` centres the strip on it.
            chart.draw_series(bars.iter().map(|(x0, x1, v)| {
                Text::new(
                    fixed(*v),
                    ((x0 + x1) / 2.0, v + y_hi * 0.012),
                    ("sans-serif", 11)
                        .into_font()
                        .color(&RGBColor(60, 60, 60))
                        .transform(FontTransform::Rotate270)
                        .pos(Pos::new(HPos::Left, VPos::Center)),
                )
            }))?;
        }

        // Group labels, placed from the plot's pixel range. Both ranges are
        // absolute backend coordinates, so the difference is the offset into the
        // label strip, whose own draw calls are relative to its top-left.
        let (plot_px, _) = chart.plotting_area().get_pixel_range();
        let strip_x0 = label_area.get_pixel_range().0.start;
        let width = f64::from(plot_px.end - plot_px.start);
        for (g, group) in self.groups.iter().enumerate() {
            let centre = plot_px.start
                + to_i32((width * (count_to_f64(g) + 0.5) / count_to_f64(n_groups)).round());
            label_area.draw(&Text::new(
                group.label.clone(),
                (centre - strip_x0, 13),
                ("sans-serif", 14)
                    .into_font()
                    .color(&OK_BLACK)
                    .pos(Pos::new(HPos::Center, VPos::Center)),
            ))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unabbreviated label renders at full width and overlaps its
    /// neighbours instead of erroring, so every region needs a short form —
    /// in **both** spaces, whose region sets do not overlap beyond `all`.
    #[test]
    fn every_region_label_is_abbreviated() {
        for space in ObjectiveSpace::ALL {
            for (name, label) in crate::r2::region_labels(space) {
                if name == "all" {
                    continue;
                }
                assert_ne!(
                    label,
                    format!("W_{name}"),
                    "region `{name}` of {space} has no abbreviation, so it renders as `{label}`"
                );
            }
        }
    }

    /// The bar chart's x axis is `r2::build_regions`' output order; a region
    /// added to one and not the other silently mislabels every bar after it.
    /// The two spaces build different region sets, so both are checked.
    #[test]
    fn labels_match_the_regions_the_indicator_builds() {
        for space in ObjectiveSpace::ALL {
            let built: Vec<String> = crate::r2::Weights::new(space)
                .regions
                .iter()
                .map(|r| r.name.clone())
                .collect();
            let labelled: Vec<String> = crate::r2::region_labels(space)
                .into_iter()
                .map(|(n, _)| n)
                .collect();
            assert_eq!(labelled, built, "region labels desynced for {space}");
        }
    }
}
