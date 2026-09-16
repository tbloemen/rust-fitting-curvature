//! Experiment 4 (`ablation-results`) — which loss setting improves the front,
//! and for which dataset and geometry?
//!
//! [`GainDots`] is the figure this module draws: **one row per dataset, the R2
//! gain of a loss setting over `all_off` along x, one panel per embedding
//! geometry, one marker per setting on its own sub-row.** The stacked fronts
//! show the settings as curves that mostly coincide, and the R2 bar charts
//! show one (dataset, geometry) at a time; neither answers the chapter's
//! question — *does a setting help, and does that depend on the data or on
//! the geometry?* — without the reader carrying numbers between panels. Here
//! the four settings of one cell sit stacked a few pixels apart on one row,
//! so "which one moved the front" is the marker furthest from the zero rule,
//! and the same row in the neighbouring panel is the same dataset under
//! another curvature.
//!
//! ### The numbers are the table's
//!
//! The rows of `results/r2_delta_<space>.jsonl` under the `all` region — the
//! same `delta_r2` the Typst table prints — at one N, times [`SCALE`]. `ΔR2 =
//! R2(all_off) − R2(setting)`, so a marker **right** of zero is a setting
//! that served the uniform preference better than the baseline; R2 is a cost.
//! `all_off` is the zero rule rather than a marker, and `rms_anchored` is not
//! drawn ([`EXCLUDED`]) for the reason `exp4::R2Bars` gives: it fixes the
//! curvature gauge for Experiment 3 rather than ablating a loss term.
//!
//! ### The axis is asinh, not linear
//!
//! Most gains are within a unit or two of zero; a handful (`hyperbolic_shells`
//! under every geometry, the norm loss on the Euclidean real datasets) are
//! ten to forty. On a linear axis the majority collapses onto the zero rule
//! and the figure becomes a picture of one dataset; on a log axis zero and
//! the sign are lost. [`AsinhAxis`] is linear within about a unit of zero and
//! logarithmic beyond, so ±1 and ±35 are both legible on one axis, and the
//! ticks are labelled with the value, not the transform. The three panels
//! share the axis so a marker's position is comparable across geometries.
//!
//! ### The band around zero
//!
//! The shaded band is ±[`RESOLUTION`], one unit of the ×1000 scale: the
//! Typst table prints the real datasets' levels (100–200) to zero decimals,
//! so a gain inside the band is one the table cannot show and the stacked
//! fronts draw as coincident curves. It is a reading aid, not a significance
//! test — the cross-dataset Wilcoxon of `r2 aggregate` is that — and it is
//! also why ±1 needs no tick of its own.
//!
//! ### Rows, sub-rows, marks
//!
//! Datasets synthetic first, in the order the results chapter introduces
//! them, then the real ones, as the region-gain heatmap orders them; a
//! dataset absent from every geometry at this N is not a row. Inside a row
//! the settings are stacked in [`SETTINGS`] order, the three single terms
//! above `all_free`, so whether the combination matches its best component
//! or exceeds it is read down the sub-rows. Each marker has its setting's
//! colour (`setting_color`) and its own shape, outlined in black, so the
//! four stay apart in greyscale; a stem from the zero rule carries the sign
//! when the marker sits inside the band. A setting the sweep did not run for
//! that geometry — `norm_only` on the sphere — is written `n/a` in grey on
//! its sub-row, so the absence is not read as a zero.

use plotters::coord::ranged1d::{DefaultFormatting, KeyPointHint, Ranged};
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use fitting_core::cast::{count_to_f64, to_i32};

use super::exp1::dataset_label;
use super::exp2_dumbbell::SlotAxis;
use super::{
    plot_x, setting_color, Figure, Res, GEOMETRIES, OK_BLACK, OK_GREY, REAL_DATASETS,
    SYNTH_DATASETS,
};
use crate::aggregate::DeltaRow;
use crate::r2::REGION_ALL;
use crate::style_mesh;

/// Every gain is multiplied by this, the way the Typst table multiplies it.
const SCALE: f64 = 1000.0;

/// Settings that are not drawn; see the module doc.
const EXCLUDED: [&str; 1] = ["rms_anchored"];

/// Half-width of the band around zero, in scaled units.
const RESOLUTION: f64 = 1.0;

/// The marker shape of a setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Circle,
    Triangle,
    Square,
    Diamond,
}

/// The settings drawn, top to bottom within a row, with their shape and the
/// sub-row's offset from the row centre in slots.
pub const SETTINGS: [(&str, Shape, f64); 4] = [
    ("centering_only", Shape::Circle, -0.3),
    ("global_only", Shape::Triangle, -0.1),
    ("norm_only", Shape::Square, 0.1),
    ("all_free", Shape::Diamond, 0.3),
];

/// Canvas width: one full text width in the thesis, as the Exp 2 dumbbell.
const WIDTH: u32 = 740;
/// Pixels per dataset row: four sub-rows of a marker plus a gap each.
const ROW: u32 = 44;
/// The strip above the panels carrying the legend.
const LEGEND_STRIP: u32 = 26;
/// The strip below the panels carrying the shared x description.
const DESC_STRIP: u32 = 24;
/// Room above each panel for the geometry caption.
const CAPTION: u32 = 22;
/// Room for the tick labels under a panel.
const X_LABEL_AREA: u32 = 24;
/// Room for the dataset labels left of the first panel.
const Y_LABEL_AREA: u32 = 116;
const MARGIN: u32 = 8;
/// Marker radius; every shape is drawn to this half-size.
const DOT: i32 = 4;

/// One dataset's row.
#[derive(Debug, Clone)]
pub struct Row {
    dataset: String,
    /// Scaled gain per geometry of [`GEOMETRIES`], per setting of
    /// [`SETTINGS`]; `None` where the sweep has no such cell.
    values: Vec<[Option<f64>; SETTINGS.len()]>,
    /// Row centre, in slots from the top.
    centre: f64,
}

impl Row {
    /// The dataset this row is.
    #[must_use]
    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    /// The scaled gain of *setting* under *geometry*, if drawn.
    #[must_use]
    pub fn gain(&self, geometry: &str, setting: &str) -> Option<f64> {
        let g = GEOMETRIES.iter().position(|x| *x == geometry)?;
        let s = SETTINGS.iter().position(|(x, _, _)| *x == setting)?;
        self.values[g][s]
    }
}

/// The per-setting R2 gain dot plot at one N.
pub struct GainDots {
    n: usize,
    rows: Vec<Row>,
    axis: AsinhAxis,
}

impl GainDots {
    /// The figure over the stage-2 rows at N. Rows are the datasets with at
    /// least one drawn gain, in chapter order.
    #[must_use]
    pub fn new(rows: &[DeltaRow], n: usize) -> Self {
        let mut out = Vec::new();
        let mut max_abs: f64 = 0.0;
        for dataset in SYNTH_DATASETS.iter().chain(REAL_DATASETS.iter()) {
            let values: Vec<[Option<f64>; SETTINGS.len()]> = GEOMETRIES
                .iter()
                .map(|geometry| {
                    let mut cell = [None; SETTINGS.len()];
                    for (i, (setting, _, _)) in SETTINGS.iter().enumerate() {
                        if EXCLUDED.contains(setting) {
                            continue;
                        }
                        cell[i] = rows
                            .iter()
                            .find(|r| {
                                r.n == n
                                    && r.region == REGION_ALL
                                    && r.dataset == *dataset
                                    && r.geometry == *geometry
                                    && r.setting == *setting
                            })
                            .and_then(|r| r.delta_r2)
                            .map(|d| d * SCALE)
                            .filter(|d| d.is_finite());
                    }
                    cell
                })
                .collect();
            if values.iter().flatten().all(Option::is_none) {
                continue;
            }
            for v in values.iter().flatten().flatten() {
                max_abs = max_abs.max(v.abs());
            }
            out.push(Row {
                dataset: (*dataset).to_string(),
                values,
                centre: count_to_f64(out.len()) + 0.5,
            });
        }
        Self {
            n,
            rows: out,
            axis: AsinhAxis::symmetric(max_abs),
        }
    }

    /// True when there is at least one row to draw.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.rows.is_empty()
    }

    /// The rows, top to bottom.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// The shared x axis.
    #[must_use]
    pub fn axis(&self) -> &AsinhAxis {
        &self.axis
    }

    fn total(&self) -> f64 {
        count_to_f64(self.rows.len())
    }
}

/// An axis linear near zero and logarithmic in the tails: `asinh(v)`, which
/// is `v` for `|v| ≪ 1` and `±ln(2|v|)` for `|v| ≫ 1`. Symmetric about zero,
/// with ticks at `±3·10^k` and `±10^k` inside the range, labelled by value.
#[derive(Debug, Clone)]
pub struct AsinhAxis {
    hi: f64,
    ticks: Vec<f64>,
}

impl AsinhAxis {
    /// The axis covering `±max_abs`, padded so the outermost marker clears
    /// the frame. A range under one unit is widened to it, so an all-zero
    /// figure still shows the band.
    #[must_use]
    pub fn symmetric(max_abs: f64) -> Self {
        let hi = (max_abs.max(RESOLUTION).asinh() + 0.25).sinh();
        let mut ticks = vec![0.0];
        let mut decade = 1.0;
        while decade <= hi {
            for m in [1.0, 3.0] {
                let t = m * decade;
                // The unit tick is the band's edge and would only crowd it.
                if t <= hi && t > RESOLUTION {
                    ticks.push(t);
                    ticks.push(-t);
                }
            }
            decade *= 10.0;
        }
        ticks.sort_by(f64::total_cmp);
        Self { hi, ticks }
    }

    /// The tick's label: the value, as an integer.
    #[must_use]
    pub fn label(v: &f64) -> String {
        format!("{v:.0}")
    }

    /// The end of the axis; the start is its negative.
    #[must_use]
    pub fn hi(&self) -> f64 {
        self.hi
    }

    /// The tick positions, in axis order.
    #[must_use]
    pub fn ticks(&self) -> &[f64] {
        &self.ticks
    }
}

impl Ranged for AsinhAxis {
    type FormatOption = DefaultFormatting;
    type ValueType = f64;

    fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
        let t = self.hi.asinh();
        RangedCoordf64::from(-t..t).map(&value.asinh(), limit)
    }

    fn key_points<Hint: KeyPointHint>(&self, _hint: Hint) -> Vec<f64> {
        self.ticks.clone()
    }

    fn range(&self) -> std::ops::Range<f64> {
        -self.hi..self.hi
    }
}

/// Draw *shape* centred on the pixel *at*, filled in *color* and outlined
/// in black.
fn draw_marker<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    at: (i32, i32),
    shape: Shape,
    color: RGBColor,
) -> Res
where
    DB::ErrorType: 'static,
{
    let (x, y) = at;
    match shape {
        Shape::Circle => {
            area.draw(&Circle::new(at, DOT, color.filled()))?;
            area.draw(&Circle::new(at, DOT, OK_BLACK.stroke_width(1)))?;
        }
        Shape::Square => {
            let corners = [(x - DOT, y - DOT), (x + DOT, y + DOT)];
            area.draw(&Rectangle::new(corners, color.filled()))?;
            area.draw(&Rectangle::new(corners, OK_BLACK.stroke_width(1)))?;
        }
        Shape::Triangle | Shape::Diamond => {
            // A hair larger than the circle so the shapes read as the same
            // size: a polygon's area is well inside its radius.
            let r = DOT + 1;
            let points = if shape == Shape::Triangle {
                vec![(x - r, y + r - 1), (x + r, y + r - 1), (x, y - r)]
            } else {
                vec![(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
            };
            area.draw(&Polygon::new(points.clone(), color.filled()))?;
            let mut outline = points;
            outline.push(outline[0]);
            area.draw(&PathElement::new(outline, OK_BLACK.stroke_width(1)))?;
        }
    }
    Ok(())
}

impl Figure for GainDots {
    fn name(&self) -> String {
        format!("exp4_gain_dots_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        let plot = u32::try_from(to_i32((self.total() * f64::from(ROW)).ceil())).unwrap_or(0);
        (
            WIDTH,
            LEGEND_STRIP + CAPTION + plot + X_LABEL_AREA + DESC_STRIP + 2 * MARGIN,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let (legend, rest) = root.split_vertically(LEGEND_STRIP);
        Self::draw_legend(&legend)?;
        let (_, desc_h) = rest.dim_in_pixel();
        let (panels, desc) = rest.split_vertically(desc_h.saturating_sub(DESC_STRIP));

        // The first panel is wider by the label area, so the three plotting
        // areas come out the same width.
        let (width, _) = panels.dim_in_pixel();
        let inner = width.saturating_sub(2 * MARGIN + Y_LABEL_AREA);
        let each = inner / u32::try_from(GEOMETRIES.len()).unwrap_or(1);
        let first = MARGIN + Y_LABEL_AREA + each;
        let areas = panels.split_by_breakpoints([first, first + each], [] as [u32; 0]);

        let y_axis = SlotAxis {
            total: self.total(),
            centres: self.rows.iter().map(|r| r.centre).collect(),
        };
        let labels: Vec<&str> = self
            .rows
            .iter()
            .map(|r| dataset_label(&r.dataset))
            .collect();
        let y_label = |v: &f64| {
            self.rows
                .iter()
                .position(|r| (r.centre - v).abs() < 1e-9)
                .map_or_else(String::new, |i| labels[i].to_string())
        };

        for (col, (geometry, area)) in GEOMETRIES.iter().zip(&areas).enumerate() {
            let mut builder = ChartBuilder::on(area);
            builder
                .margin_top(2)
                .margin_bottom(MARGIN)
                .margin_right(6)
                .caption(
                    *geometry,
                    ("sans-serif", 15).into_font().style(FontStyle::Bold),
                )
                .x_label_area_size(X_LABEL_AREA);
            if col == 0 {
                builder.margin_left(MARGIN).y_label_area_size(Y_LABEL_AREA);
            } else {
                builder.margin_left(6).y_label_area_size(0);
            }
            let mut chart = builder.build_cartesian_2d(self.axis.clone(), y_axis.clone())?;
            let mut mesh = chart.configure_mesh();
            style_mesh!(mesh)
                .disable_y_mesh()
                .x_label_formatter(&AsinhAxis::label)
                .y_label_formatter(&y_label);
            if col != 0 {
                mesh.disable_y_axis();
            }
            mesh.draw()?;

            let (lo, hi) = (-self.axis.hi, self.axis.hi);
            // Alternate rows shaded, so a row's four sub-rows read as one.
            for (i, row) in self.rows.iter().enumerate() {
                if i % 2 == 1 {
                    chart.draw_series(std::iter::once(Rectangle::new(
                        [(lo, row.centre - 0.5), (hi, row.centre + 0.5)],
                        RGBColor(240, 240, 240).filled(),
                    )))?;
                }
            }
            // The band the table cannot resolve, then the zero rule over it.
            chart.draw_series(std::iter::once(Rectangle::new(
                [(-RESOLUTION, 0.0), (RESOLUTION, self.total())],
                OK_GREY.mix(0.22).filled(),
            )))?;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(0.0, 0.0), (0.0, self.total())],
                RGBColor(90, 90, 90).stroke_width(1),
            )))?;

            // Stems in data coordinates, markers by pixel on the panel area
            // so the four shapes share one drawing routine with the legend.
            let origin = (
                area.get_pixel_range().0.start,
                area.get_pixel_range().1.start,
            );
            let na_font = ("sans-serif", 11)
                .into_font()
                .color(&OK_GREY)
                .pos(Pos::new(HPos::Left, VPos::Center));
            for row in &self.rows {
                for ((setting, shape, off), value) in SETTINGS.iter().zip(&row.values[col]) {
                    let y = row.centre + off;
                    let Some(v) = value else {
                        let (px, py) = chart.backend_coord(&(0.0, y));
                        area.draw(&Text::new(
                            "n/a",
                            (px - origin.0 + DOT + 2, py - origin.1),
                            na_font.clone(),
                        ))?;
                        continue;
                    };
                    let color = setting_color(setting);
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(0.0, y), (*v, y)],
                        color.mix(0.55).stroke_width(2),
                    )))?;
                    let (px, py) = chart.backend_coord(&(*v, y));
                    draw_marker(area, (px - origin.0, py - origin.1), *shape, color)?;
                }
            }
        }

        // The x description once, centred under the three panels.
        let (dw, dh) = desc.dim_in_pixel();
        let x = plot_x(
            &(to_i32(f64::from(MARGIN + Y_LABEL_AREA))..to_i32(f64::from(dw - MARGIN))),
            0.5,
        );
        desc.draw(&Text::new(
            "\u{394}R2 \u{d7} 1000 against all_off (right of zero: the setting improves the front; \
             band: differences under one unit)",
            (x, to_i32(f64::from(dh) / 2.0)),
            ("sans-serif", 13)
                .into_font()
                .color(&RGBColor(30, 30, 30))
                .pos(Pos::new(HPos::Center, VPos::Center)),
        ))?;
        Ok(())
    }
}

impl GainDots {
    /// The legend strip: one marker and name per drawn setting.
    fn draw_legend<DB: DrawingBackend>(area: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let (width, height) = area.dim_in_pixel();
        let font = ("sans-serif", 14).into_font().color(&OK_BLACK);
        let inner = f64::from(width) - 2.0 * f64::from(MARGIN);
        let slot = to_i32(inner / count_to_f64(SETTINGS.len()));
        let cy = to_i32(f64::from(height) / 2.0);
        for (i, (setting, shape, _)) in SETTINGS.iter().enumerate() {
            let x0 = to_i32(f64::from(MARGIN)) + DOT + to_i32(count_to_f64(i)) * slot;
            draw_marker(area, (x0, cy), *shape, setting_color(setting))?;
            area.draw(&Text::new(
                *setting,
                (x0 + 12, cy),
                font.clone().pos(Pos::new(HPos::Left, VPos::Center)),
            ))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(dataset: &str, geometry: &str, setting: &str, region: &str, d: f64) -> DeltaRow {
        DeltaRow {
            space: "obj5".into(),
            n: 5000,
            geometry: geometry.into(),
            setting: setting.into(),
            dataset: dataset.into(),
            region: region.into(),
            r2: 0.1 - d,
            r2_baseline: Some(0.1),
            delta_r2: Some(d),
        }
    }

    #[test]
    fn rows_follow_chapter_order_and_drop_the_baseline_and_rms() {
        let rows = vec![
            delta("mnist", "euclidean", "norm_only", "all", 0.02),
            delta("mnist", "euclidean", "all_off", "all", 0.0),
            delta("grid", "hyperbolic", "rms_anchored", "all", 0.01),
            delta("grid", "hyperbolic", "all_free", "all", 0.0025),
            // Another region and another N are not this figure's.
            delta("tree", "spherical", "all_free", "trustworthiness", 0.5),
            DeltaRow {
                n: 1000,
                ..delta("tree", "spherical", "all_free", "all", 0.5)
            },
        ];
        let fig = GainDots::new(&rows, 5000);
        let datasets: Vec<&str> = fig.rows().iter().map(Row::dataset).collect();
        assert_eq!(
            datasets,
            ["grid", "mnist"],
            "synthetic first, absent skipped"
        );
        assert_eq!(fig.rows()[0].gain("hyperbolic", "all_free"), Some(2.5));
        assert_eq!(fig.rows()[1].gain("euclidean", "norm_only"), Some(20.0));
        // Neither the baseline nor the gauge setting is a marker.
        assert_eq!(fig.rows()[1].gain("euclidean", "all_off"), None);
        assert_eq!(fig.rows()[0].gain("hyperbolic", "rms_anchored"), None);
    }

    #[test]
    fn missing_setting_is_none_not_zero() {
        let rows = vec![delta("sphere", "spherical", "all_free", "all", 0.001)];
        let fig = GainDots::new(&rows, 5000);
        assert_eq!(fig.rows()[0].gain("spherical", "norm_only"), None);
        assert_eq!(fig.rows()[0].gain("spherical", "all_free"), Some(1.0));
        assert_eq!(fig.rows()[0].gain("euclidean", "all_free"), None);
    }

    #[test]
    fn asinh_axis_is_symmetric_monotone_and_compresses_the_tails() {
        let axis = AsinhAxis::symmetric(37.0);
        assert!(axis.hi() > 37.0);
        let px = |v: f64| axis.map(&v, (0, 1000));
        assert_eq!(px(0.0), 500);
        // Plotters rounds to a pixel, so mirror images may differ by one.
        assert!((px(-20.0) - (1000 - px(20.0))).abs() <= 1);
        assert!(px(1.0) < px(3.0) && px(3.0) < px(10.0) && px(10.0) < px(30.0));
        // The first unit takes more pixels than the run from 10 to 30.
        assert!(px(1.0) - px(0.0) > (px(30.0) - px(10.0)) / 2);
        assert_eq!(
            axis.ticks(),
            &[-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0],
            "the unit tick is the band edge, not a tick"
        );
    }

    #[test]
    fn a_flat_figure_still_shows_the_band() {
        let axis = AsinhAxis::symmetric(0.0);
        assert!(axis.hi() > RESOLUTION);
        assert_eq!(axis.ticks(), &[0.0]);
    }
}
