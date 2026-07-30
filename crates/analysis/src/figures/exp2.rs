//! Experiment 2 — stacked Pareto fronts, one panel per (dataset, geometry),
//! one curve per loss-weight setting.

use plotters::coord::Shift;
use plotters::prelude::*;

use super::*;
use crate::cell::Cell;
use crate::pareto::slice_front_2d;
use crate::style_mesh;

pub struct StackedFronts<'a> {
    cells: &'a CellMap,
    n: usize,
}

/// The per-panel front polyline for one setting, in plot coordinates.
struct Curve {
    setting: &'static str,
    points: Vec<(f64, f64)>,
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
        (1100, 1300)
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
            ("sans-serif", 20).into_font().color(&OK_BLACK),
        )?;
        let (legend, grid) = root.split_vertically(34);
        let entries: Vec<LegendEntry> = SETTING_ORDER
            .iter()
            .map(|s| LegendEntry::new(*s, setting_color(s)))
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
                let (x_lo, x_hi) = padded_range(&all_x, 0.06).unwrap_or((0.0, 1.0));
                let (y_lo, y_hi) = padded_range(&all_y, 0.06).unwrap_or((0.0, 1.0));

                let mut chart = ChartBuilder::on(panel)
                    .margin(6)
                    .margin_top(if i == 0 { 4 } else { 6 })
                    .caption(
                        if i == 0 { *geometry } else { "" },
                        ("sans-serif", 16).into_font().style(FontStyle::Bold),
                    )
                    .x_label_area_size(38)
                    .y_label_area_size(62)
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
                style_mesh!(chart.configure_mesh())
                    .x_desc(x_desc)
                    .y_desc(y_desc)
                    .x_labels(5)
                    .y_labels(5)
                    .draw()?;

                for curve in &curves {
                    let color = setting_color(curve.setting);
                    chart.draw_series(LineSeries::new(
                        curve.points.iter().copied(),
                        color.mix(0.9).stroke_width(2),
                    ))?;
                    chart.draw_series(
                        curve
                            .points
                            .iter()
                            .map(|p| Circle::new(*p, 2, color.mix(0.9).filled())),
                    )?;
                }
            }
        }
        Ok(())
    }
}
