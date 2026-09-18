//! The scaffolding Experiment 4's per-setting dot plots share.
//!
//! [`super::exp4_gain_dots::GainDots`] and
//! [`super::exp4_epsilon_dots::EpsilonDots`] are the same figure with a
//! different quantity on x: **one row per dataset, one panel per embedding
//! geometry, one sub-row per loss setting inside every row**, so the two can
//! be set under each other in the thesis and a row read across both. What
//! is the same lives here — the settings and their marks, the row layout,
//! the legend, the canvas, and [`draw_panels`], which draws everything up to
//! the marks themselves and hands each panel back to the figure for those.
//! What differs — the x axis, what a sub-row holds and how it is marked — is
//! the figure's.

use plotters::coord::cartesian::Cartesian2d;
use plotters::coord::ranged1d::{DefaultFormatting, Ranged};
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use fitting_core::cast::{count_to_f64, to_i32};

use super::exp1::dataset_label;
use super::exp2_dumbbell::SlotAxis;
use super::{
    plot_x, setting_color, Res, GEOMETRIES, OK_BLACK, OK_GREY, REAL_DATASETS, SYNTH_DATASETS,
};
use crate::cell::setting_applies;
use crate::style_mesh;

/// Every value is multiplied by this, the way the Typst tables multiply
/// theirs: R2 and ε both sit around 0.01–0.1, so the unscaled axis would
/// spend its digits on leading zeros.
pub(super) const SCALE: f64 = 1000.0;

/// Settings that are not drawn.
///
/// `rms_anchored` fixes the curvature gauge for Experiment 3 rather than
/// ablating a loss term, and it exists for hyperbolic only — so it is not one
/// of the alternatives these figures compare. The Typst tables drop it for
/// the same reason.
pub(super) const EXCLUDED: [&str; 1] = ["rms_anchored"];

/// The marker shape of a setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Circle,
    Triangle,
    Square,
    Diamond,
}

/// The settings drawn, top to bottom within a row, with their shape and the
/// sub-row's offset from the row centre in slots: the three single terms
/// above `all_free`, so whether the combination matches its best component
/// or exceeds it is read down the sub-rows.
pub const SETTINGS: [(&str, Shape, f64); 4] = [
    ("centering_only", Shape::Circle, -0.3),
    ("global_only", Shape::Triangle, -0.1),
    ("norm_only", Shape::Square, 0.1),
    ("all_free", Shape::Diamond, 0.3),
];

/// Canvas width: one full text width in the thesis, as the Exp 2 dumbbell.
pub(super) const WIDTH: u32 = 740;
/// Pixels per dataset row: four sub-rows of a marker plus a gap each.
pub(super) const ROW: u32 = 44;
/// Height of one legend row above the panels.
pub(super) const LEGEND_ROW: u32 = 26;
/// The strip below the panels carrying the shared x axis label.
pub(super) const DESC_STRIP: u32 = 24;
/// Room above each panel for the geometry caption.
pub(super) const CAPTION: u32 = 22;
/// Room for the tick labels under a panel.
pub(super) const X_LABEL_AREA: u32 = 24;
/// Room for the dataset labels left of the first panel.
pub(super) const Y_LABEL_AREA: u32 = 116;
pub(super) const MARGIN: u32 = 8;
/// Horizontal margin on each side of a panel: two of these separate
/// neighbouring panels, enough that the outer tick labels of one panel
/// (`40` and `-40` on a linear axis) do not touch the next panel's.
pub(super) const PANEL_GAP: u32 = 13;
/// Marker radius; every shape is drawn to this half-size.
pub(super) const DOT: i32 = 4;

/// One dataset's row: a value per geometry of [`GEOMETRIES`] per setting of
/// [`SETTINGS`], `None` where the sweep has no such cell or where the
/// setting's weight is inert under that geometry ([`setting_applies`]).
#[derive(Debug, Clone)]
pub struct Row<T> {
    dataset: String,
    values: Vec<[Option<T>; SETTINGS.len()]>,
    /// Row centre, in slots from the top.
    centre: f64,
}

impl<T: Copy> Row<T> {
    /// The dataset this row is.
    #[must_use]
    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    /// The value of *setting* under *geometry*, if drawn.
    #[must_use]
    pub fn value(&self, geometry: &str, setting: &str) -> Option<T> {
        let g = GEOMETRIES.iter().position(|x| *x == geometry)?;
        let s = SETTINGS.iter().position(|(x, _, _)| *x == setting)?;
        self.values[g][s]
    }

    /// The row's centre in slots.
    #[must_use]
    pub fn centre(&self) -> f64 {
        self.centre
    }

    /// The values of one panel, in [`SETTINGS`] order.
    #[must_use]
    pub fn panel(&self, col: usize) -> &[Option<T>; SETTINGS.len()] {
        &self.values[col]
    }

    /// Every drawn value of the row, in panel then setting order.
    pub fn drawn(&self) -> impl Iterator<Item = T> + '_ {
        self.values.iter().flatten().flatten().copied()
    }
}

/// The rows of a figure: every dataset with at least one drawn value, in
/// chapter order — synthetic first, as the results chapter introduces them,
/// then the real ones, the order the region-gain heatmap uses. *lookup*
/// gives the value of a (dataset, geometry, setting) or `None`; it is never
/// asked about an [`EXCLUDED`] setting, nor about a cell whose weight is
/// inert under its geometry ([`setting_applies`]) — a `centering_only` cell
/// off the hyperboloid, or an `all_free` cell on the sphere, is a repeat of
/// the search without that weight, and is drawn as `n/a` like a cell that
/// was never run.
pub(super) fn collect_rows<T: Copy>(lookup: impl Fn(&str, &str, &str) -> Option<T>) -> Vec<Row<T>> {
    let mut out = Vec::new();
    for dataset in SYNTH_DATASETS.iter().chain(REAL_DATASETS.iter()) {
        let values: Vec<[Option<T>; SETTINGS.len()]> = GEOMETRIES
            .iter()
            .map(|geometry| {
                let mut cell = [None; SETTINGS.len()];
                for (i, (setting, _, _)) in SETTINGS.iter().enumerate() {
                    if EXCLUDED.contains(setting) || !setting_applies(setting, geometry) {
                        continue;
                    }
                    cell[i] = lookup(dataset, geometry, setting);
                }
                cell
            })
            .collect();
        if values.iter().flatten().all(Option::is_none) {
            continue;
        }
        out.push(Row {
            dataset: (*dataset).to_string(),
            values,
            centre: count_to_f64(out.len()) + 0.5,
        });
    }
    out
}

/// The canvas for *rows* dataset rows under *`legend_rows`* rows of legend.
pub(super) fn canvas(rows: usize, legend_rows: u32) -> (u32, u32) {
    let plot = u32::try_from(to_i32((count_to_f64(rows) * f64::from(ROW)).ceil())).unwrap_or(0);
    (
        WIDTH,
        legend_rows * LEGEND_ROW + CAPTION + plot + X_LABEL_AREA + DESC_STRIP + 2 * MARGIN,
    )
}

/// Draw *shape* centred on the pixel *at*: filled in *color* and outlined in
/// black, or — `filled = false` — white inside with the outline in *color*.
pub(super) fn draw_marker<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    at: (i32, i32),
    shape: Shape,
    color: RGBColor,
    filled: bool,
) -> Res
where
    DB::ErrorType: 'static,
{
    let (fill, outline) = if filled {
        (color.filled(), OK_BLACK.stroke_width(1))
    } else {
        (WHITE.filled(), color.stroke_width(2))
    };
    let (x, y) = at;
    match shape {
        Shape::Circle => {
            area.draw(&Circle::new(at, DOT, fill))?;
            area.draw(&Circle::new(at, DOT, outline))?;
        }
        Shape::Square => {
            let corners = [(x - DOT, y - DOT), (x + DOT, y + DOT)];
            area.draw(&Rectangle::new(corners, fill))?;
            area.draw(&Rectangle::new(corners, outline))?;
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
            area.draw(&Polygon::new(points.clone(), fill))?;
            let mut path = points;
            path.push(path[0]);
            area.draw(&PathElement::new(path, outline))?;
        }
    }
    Ok(())
}

/// One legend row: a filled marker and name per drawn setting.
pub(super) fn draw_settings_legend<DB: DrawingBackend>(area: &DrawingArea<DB, Shift>) -> Res
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
        draw_marker(area, (x0, cy), *shape, setting_color(setting), true)?;
        area.draw(&Text::new(
            *setting,
            (x0 + 12, cy),
            font.clone().pos(Pos::new(HPos::Left, VPos::Center)),
        ))?;
    }
    Ok(())
}

/// `n/a` in grey just right of the pixel *at* on the panel area: the mark of
/// a setting the sweep did not run for that geometry, so the absence is not
/// read as a value.
pub(super) fn draw_na<DB: DrawingBackend>(area: &DrawingArea<DB, Shift>, at: (i32, i32)) -> Res
where
    DB::ErrorType: 'static,
{
    area.draw(&Text::new(
        "n/a",
        (at.0 + DOT + 2, at.1),
        ("sans-serif", 11)
            .into_font()
            .color(&OK_GREY)
            .pos(Pos::new(HPos::Left, VPos::Center)),
    ))?;
    Ok(())
}

/// The chart of one panel.
pub(super) type Panel<'a, DB, X> = ChartContext<'a, DB, Cartesian2d<X, SlotAxis>>;

/// Everything of the figure that is not a mark.
///
/// Splits the canvas into the legend rows, the three panels and the x label
/// strip; builds one chart per geometry on the shared *axis* — the first
/// carrying the dataset labels, all three the same plotting width — with the
/// geometry caption centred on the plotting area (plotters' own caption
/// centres on the whole area, label column included); shades alternate rows
/// so a row's sub-rows read as one; draws *rule* as a vertical line, if
/// there is one; and writes *`x_desc`* once under the panels. Then, for each
/// panel, calls *`draw_cell`* with the chart, the panel's area and the area's
/// pixel origin, once per row — marks are drawn by pixel on the area so the
/// figures share [`draw_marker`] with the legend — and finally *`draw_legend`*
/// on the legend strip.
#[allow(clippy::too_many_arguments)]
pub(super) fn draw_panels<DB, X, T>(
    root: &DrawingArea<DB, Shift>,
    rows: &[Row<T>],
    legend_rows: u32,
    axis: &X,
    x_label: &dyn Fn(&f64) -> String,
    rule: Option<f64>,
    x_desc: &str,
    draw_legend: impl FnOnce(&DrawingArea<DB, Shift>) -> Res,
    mut draw_cell: impl FnMut(
        usize,
        &mut Panel<'_, DB, X>,
        &DrawingArea<DB, Shift>,
        (i32, i32),
        &Row<T>,
    ) -> Res,
) -> Res
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: Ranged<ValueType = f64, FormatOption = DefaultFormatting> + Clone,
    T: Copy,
{
    let (legend, rest) = root.split_vertically(legend_rows * LEGEND_ROW);
    draw_legend(&legend)?;
    let (_, rest_h) = rest.dim_in_pixel();
    let (panels, desc) = rest.split_vertically(rest_h.saturating_sub(DESC_STRIP));

    // The first panel is wider by the label area, so the three plotting
    // areas come out the same width.
    let (width, _) = panels.dim_in_pixel();
    let inner = width.saturating_sub(2 * MARGIN + Y_LABEL_AREA);
    let each = inner / u32::try_from(GEOMETRIES.len()).unwrap_or(1);
    let first = MARGIN + Y_LABEL_AREA + each;
    let areas = panels.split_by_breakpoints([first, first + each], [] as [u32; 0]);

    let total = count_to_f64(rows.len());
    let y_axis = SlotAxis {
        total,
        centres: rows.iter().map(|r| r.centre).collect(),
    };
    let labels: Vec<&str> = rows.iter().map(|r| dataset_label(&r.dataset)).collect();
    let y_label = |v: &f64| {
        rows.iter()
            .position(|r| (r.centre - v).abs() < 1e-9)
            .map_or_else(String::new, |i| labels[i].to_string())
    };

    for (col, (geometry, area)) in GEOMETRIES.iter().zip(&areas).enumerate() {
        let mut builder = ChartBuilder::on(area);
        builder
            .margin_top(CAPTION)
            .margin_bottom(MARGIN)
            .margin_right(PANEL_GAP)
            .x_label_area_size(X_LABEL_AREA);
        if col == 0 {
            builder.margin_left(MARGIN).y_label_area_size(Y_LABEL_AREA);
        } else {
            builder.margin_left(PANEL_GAP).y_label_area_size(0);
        }
        let mut chart = builder.build_cartesian_2d(axis.clone(), y_axis.clone())?;
        let mut mesh = chart.configure_mesh();
        style_mesh!(mesh)
            .disable_y_mesh()
            .x_label_formatter(x_label)
            .y_label_formatter(&y_label);
        if col != 0 {
            mesh.disable_y_axis();
        }
        mesh.draw()?;

        let (lo, hi) = (axis.range().start, axis.range().end);
        for (i, row) in rows.iter().enumerate() {
            if i % 2 == 1 {
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(lo, row.centre - 0.5), (hi, row.centre + 0.5)],
                    RGBColor(240, 240, 240).filled(),
                )))?;
            }
        }
        // The rule, over the shading and under the marks.
        if let Some(x) = rule {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, total)],
                RGBColor(90, 90, 90).stroke_width(1),
            )))?;
        }

        let origin = (
            area.get_pixel_range().0.start,
            area.get_pixel_range().1.start,
        );
        let plot_px = chart.plotting_area().get_pixel_range();
        area.draw(&Text::new(
            *geometry,
            (
                plot_x(&plot_px.0, 0.5) - origin.0,
                to_i32(f64::from(CAPTION) / 2.0),
            ),
            ("sans-serif", 15)
                .into_font()
                .style(FontStyle::Bold)
                .color(&OK_BLACK)
                .pos(Pos::new(HPos::Center, VPos::Center)),
        ))?;
        for row in rows {
            draw_cell(col, &mut chart, area, origin, row)?;
        }
    }

    // The x axis label once, centred under the three panels. Only the
    // quantity: the reading direction, the scale and the rule are the
    // thesis caption's to state, as on every other figure.
    let (dw, dh) = desc.dim_in_pixel();
    let x = plot_x(
        &(to_i32(f64::from(MARGIN + Y_LABEL_AREA))..to_i32(f64::from(dw - MARGIN))),
        0.5,
    );
    desc.draw(&Text::new(
        x_desc.to_string(),
        (x, to_i32(f64::from(dh) / 2.0)),
        ("sans-serif", 13)
            .into_font()
            .color(&RGBColor(30, 30, 30))
            .pos(Pos::new(HPos::Center, VPos::Center)),
    ))?;
    Ok(())
}
