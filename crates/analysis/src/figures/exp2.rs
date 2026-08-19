//! Experiment 2 — stacked Pareto fronts, one panel per (dataset, geometry),
//! one curve per loss-weight setting.
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

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::*;
use crate::aggregate::BASELINE;
use crate::cell::Cell;
use crate::pareto::{slice_front_2d, step_polyline};
use crate::style_mesh;

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
    pub fn has_data(&self) -> bool {
        REAL_DATASETS
            .iter()
            .any(|ds| GEOMETRIES.iter().any(|g| !self.curves(ds, g).is_empty()))
    }
}

impl Figure for StackedFronts<'_> {
    fn name(&self) -> String {
        format!("exp2_stacked_fronts_N{}", self.n)
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
                "Experiment 2 — Pareto fronts by loss setting (N={}): trustworthiness vs normalised stress",
                self.n
            ),
            ("sans-serif", 22).into_font().color(&OK_BLACK),
        )?;
        let (legend, grid) = root.split_vertically(38);
        let entries: Vec<LegendEntry> = SETTING_ORDER
            .iter()
            .map(|s| {
                let e = LegendEntry::new(*s, setting_color(s));
                match setting_dash(s) {
                    Some((d, g)) => e.with_dash(d, g),
                    None => e,
                }
            })
            .collect();
        draw_legend(&legend, &entries)?;

        let panels = grid.split_evenly((REAL_DATASETS.len(), GEOMETRIES.len()));
        for (i, dataset) in REAL_DATASETS.iter().enumerate() {
            for (j, geometry) in GEOMETRIES.iter().enumerate() {
                let panel = &panels[i * GEOMETRIES.len() + j];
                let curves = self.curves(dataset, geometry);

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
                for p in curves.iter().filter_map(Curve::cheapest) {
                    x_lo = x_lo.min(p.0);
                    x_hi = x_hi.max(p.0);
                    y_lo = y_lo.min(p.1);
                    y_hi = y_hi.max(p.1);
                }

                let mut chart = ChartBuilder::on(panel)
                    .margin(6)
                    .margin_top(if i == 0 { 4 } else { 6 })
                    .caption(
                        if i == 0 { *geometry } else { "" },
                        ("sans-serif", 18).into_font().style(FontStyle::Bold),
                    )
                    .x_label_area_size(44)
                    .y_label_area_size(70)
                    .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

                let y_desc = if j == 0 {
                    format!("{dataset}   {Y_LABEL}")
                } else {
                    String::new()
                };
                let x_desc = if i == REAL_DATASETS.len() - 1 {
                    X_LABEL
                } else {
                    ""
                };
                // The mesh style is shared with Exp 3–5; the label size is
                // overridden after the macro because this canvas is larger.
                style_mesh!(chart.configure_mesh())
                    .label_style(("sans-serif", 15).into_font().color(&RGBColor(60, 60, 60)))
                    .x_desc(x_desc)
                    .y_desc(y_desc)
                    .x_labels(5)
                    .y_labels(5)
                    .draw()?;

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
                            .map(|p| Circle::new(*p, 3, color.filled())),
                    )?;

                    // Points the robust range cut: marked at the edge they left
                    // through, so a clipped panel never looks complete.
                    let (px, py) = ((x_hi - x_lo) * 0.02, (y_hi - y_lo) * 0.02);
                    let clipped: Vec<(f64, f64)> = curve
                        .points
                        .iter()
                        .filter(|(x, y)| *x < x_lo || *x > x_hi || *y < y_lo || *y > y_hi)
                        .map(|(x, y)| {
                            (x.clamp(x_lo + px, x_hi - px), y.clamp(y_lo + py, y_hi - py))
                        })
                        .collect();
                    off_scale += clipped.len();
                    chart.draw_series(
                        clipped
                            .into_iter()
                            .map(|p| TriangleMarker::new(p, 5, color.filled())),
                    )?;
                }

                if off_scale > 0 {
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
                }
            }
        }
        Ok(())
    }
}
