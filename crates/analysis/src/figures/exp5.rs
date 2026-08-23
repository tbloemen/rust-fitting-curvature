//! Experiment 5 — the `all_off` Pareto-front grid with its convex envelope, and
//! the hyperparameter marginals of those fronts.

use plotters::coord::Shift;
use plotters::prelude::*;

use super::exp3::{draw_histogram, log_histogram, Bin};
use super::*;
use crate::cell::Cell;
use crate::pareto::pareto_front_records;
use crate::style_mesh;

// ─── Front grid ───────────────────────────────────────────────────────────────

pub struct FrontGrid<'a> {
    cells: &'a CellMap,
    n: usize,
}

impl<'a> FrontGrid<'a> {
    pub fn new(cells: &'a CellMap, n: usize) -> Self {
        Self { cells, n }
    }

    fn front_xy(&self, dataset: &str, geometry: &str) -> (Vec<f64>, Vec<f64>) {
        let key = Cell::new("all_off", dataset, self.n, geometry);
        match self.cells.get(&key) {
            Some(recs) => finite_xy(&pareto_front_records(recs), X_METRIC, Y_METRIC),
            None => (Vec::new(), Vec::new()),
        }
    }

    pub fn has_data(&self) -> bool {
        REAL_DATASETS.iter().any(|ds| {
            GEOMETRIES
                .iter()
                .any(|g| !self.front_xy(ds, g).0.is_empty())
        })
    }
}

impl Figure for FrontGrid<'_> {
    fn name(&self) -> String {
        format!("exp5_front_grid_N{}", self.n)
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
                "Experiment 5 — all_off Pareto fronts, trustworthiness vs stress (N={}); black line = convex envelope",
                self.n
            ),
            ("sans-serif", 20).into_font().color(&OK_BLACK),
        )?;

        let panels = root.split_evenly((REAL_DATASETS.len(), GEOMETRIES.len()));
        for (i, dataset) in REAL_DATASETS.iter().enumerate() {
            for (j, geometry) in GEOMETRIES.iter().enumerate() {
                let (x, y) = self.front_xy(dataset, geometry);
                let (x_lo, x_hi) = padded_range(&x, 0.06).unwrap_or((0.0, 1.0));
                let (y_lo, y_hi) = padded_range(&y, 0.06).unwrap_or((0.0, 1.0));

                let mut chart = ChartBuilder::on(&panels[i * GEOMETRIES.len() + j])
                    .margin(6)
                    .caption(
                        if i == 0 { *geometry } else { "" },
                        ("sans-serif", 16).into_font().style(FontStyle::Bold),
                    )
                    .x_label_area_size(38)
                    .y_label_area_size(62)
                    .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

                style_mesh!(chart.configure_mesh())
                    .x_desc(if i == REAL_DATASETS.len() - 1 {
                        X_LABEL
                    } else {
                        ""
                    })
                    .y_desc(if j == 0 {
                        format!("{dataset}   {Y_LABEL}")
                    } else {
                        String::new()
                    })
                    .x_labels(5)
                    .y_labels(5)
                    .draw()?;

                let color = geometry_color(geometry);
                chart.draw_series(
                    x.iter()
                        .zip(&y)
                        .map(|(a, b)| Circle::new((*a, *b), 3, color.mix(0.6).filled())),
                )?;
                if x.len() >= 3 {
                    let hull = convex_lower_hull(&x, &y);
                    chart.draw_series(LineSeries::new(
                        hull.into_iter().map(|k| (x[k], y[k])),
                        OK_BLACK.stroke_width(2),
                    ))?;
                }
            }
        }
        Ok(())
    }
}

// ─── Hyperparameter marginals ─────────────────────────────────────────────────

/// Hyperparameters that vary under `all_off` (auxiliary loss weights are fixed
/// at 0), and whether they are plotted on a log axis.
const HYPERPARAMS: [(&str, bool); 4] = [
    ("learning_rate", true),
    ("perplexity_ratio", false),
    ("early_exaggeration_factor", false),
    ("curvature_magnitude", true),
];

const N_BINS: usize = 18;

pub struct Marginals<'a> {
    cells: &'a CellMap,
    n: usize,
}

impl<'a> Marginals<'a> {
    pub fn new(cells: &'a CellMap, n: usize) -> Self {
        Self { cells, n }
    }

    /// Positive, finite values of one parameter over a cell's Pareto front.
    fn values(&self, dataset: &str, geometry: &str, param: &str) -> Vec<f64> {
        // curvature_magnitude is meaningless for euclidean (fixed): skip.
        if param == "curvature_magnitude" && geometry == "euclidean" {
            return Vec::new();
        }
        let key = Cell::new("all_off", dataset, self.n, geometry);
        let Some(recs) = self.cells.get(&key) else {
            return Vec::new();
        };
        pareto_front_records(recs)
            .iter()
            .filter_map(|r| r.param(param))
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect()
    }

    pub fn has_data(&self) -> bool {
        REAL_DATASETS.iter().any(|ds| {
            HYPERPARAMS
                .iter()
                .any(|(p, _)| GEOMETRIES.iter().any(|g| self.values(ds, g, p).len() >= 2))
        })
    }
}

impl Figure for Marginals<'_> {
    fn name(&self) -> String {
        format!("exp5_marginals_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (
            400 * HYPERPARAMS.len() as u32,
            300 * REAL_DATASETS.len() as u32,
        )
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            &format!(
                "Experiment 5 — all_off Pareto-front hyperparameter marginals (N={})",
                self.n
            ),
            ("sans-serif", 20).into_font().color(&OK_BLACK),
        )?;
        let (legend, grid) = root.split_vertically(32);
        let entries: Vec<LegendEntry> = GEOMETRIES
            .iter()
            .map(|g| LegendEntry::new(*g, geometry_color(g)))
            .collect();
        draw_legend(&legend, &entries)?;

        let panels = grid.split_evenly((REAL_DATASETS.len(), HYPERPARAMS.len()));
        for (i, dataset) in REAL_DATASETS.iter().enumerate() {
            for (j, (param, logx)) in HYPERPARAMS.iter().enumerate() {
                let panel = &panels[i * HYPERPARAMS.len() + j];
                let series: Vec<(&str, Vec<f64>)> = GEOMETRIES
                    .iter()
                    .map(|g| (*g, self.values(dataset, g, param)))
                    .filter(|(_, v)| v.len() >= 2)
                    .collect();

                let all: Vec<f64> = series.iter().flat_map(|(_, v)| v.clone()).collect();
                let range = if *logx {
                    padded_log_range(&all, 0.02)
                } else {
                    padded_range(&all, 0.02)
                };
                // Bins span the data; the axis is widened to whole decades only
                // for readable log ticks, so the bars keep the data's resolution.
                let (bin_lo, bin_hi) =
                    range.unwrap_or(if *logx { (0.1, 10.0) } else { (0.0, 1.0) });
                let (x_lo, x_hi) = if *logx {
                    snap_to_decades((bin_lo, bin_hi))
                } else {
                    (bin_lo, bin_hi)
                };

                // All geometries share one set of edges so the overlay compares
                // like with like (matplotlib binned each series separately).
                let binned: Vec<(&str, Vec<Bin>)> = series
                    .iter()
                    .map(|(g, v)| {
                        let bins = if *logx {
                            log_histogram(v, bin_lo, bin_hi, N_BINS)
                        } else {
                            super::exp3::histogram(v, bin_lo, bin_hi, N_BINS)
                        };
                        (*g, bins)
                    })
                    .collect();
                let y_hi = binned
                    .iter()
                    .flat_map(|(_, b)| b.iter().map(|b| b.density))
                    .fold(0.0f64, f64::max)
                    .max(1e-9)
                    * 1.15;

                let caption = if i == 0 { *param } else { "" };
                let x_desc = if i == REAL_DATASETS.len() - 1 {
                    *param
                } else {
                    ""
                };
                let y_desc = if j == 0 {
                    format!("{dataset}   density")
                } else {
                    String::new()
                };

                // The log and linear cases differ only in the x coordinate type,
                // which is baked into the chart's type — hence the two arms.
                if *logx {
                    let mut chart = ChartBuilder::on(panel)
                        .margin(8)
                        .caption(caption, ("sans-serif", 15).into_font())
                        .x_label_area_size(42)
                        .y_label_area_size(72)
                        .build_cartesian_2d((x_lo..x_hi).log_scale(), 0.0..y_hi)?;
                    style_mesh!(chart.configure_mesh())
                        .x_desc(x_desc)
                        .y_desc(y_desc)
                        .x_label_formatter(&log_tick)
                        .draw()?;
                    for (g, bins) in &binned {
                        draw_histogram(&mut chart, bins, geometry_color(g), 0.45)?;
                    }
                } else {
                    let mut chart = ChartBuilder::on(panel)
                        .margin(8)
                        .caption(caption, ("sans-serif", 15).into_font())
                        .x_label_area_size(42)
                        .y_label_area_size(72)
                        .build_cartesian_2d(x_lo..x_hi, 0.0..y_hi)?;
                    style_mesh!(chart.configure_mesh())
                        .x_desc(x_desc)
                        .y_desc(y_desc)
                        .x_labels(5)
                        .draw()?;
                    for (g, bins) in &binned {
                        draw_histogram(&mut chart, bins, geometry_color(g), 0.45)?;
                    }
                }
            }
        }
        Ok(())
    }
}
