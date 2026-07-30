//! Experiment 4 — ρ_man-proj(κ): manifold vs projected rank agreement against
//! curvature.
//!
//! One point per (dataset, geometry, setting, N) cell: x = median κ over the
//! cell's trials, y = Spearman(manifold variant, 2D variant) over the trials.
//! Cells are independent, so they are scattered and overlaid with a
//! binned-median trend — never connected, which would imply an ordering.

use plotters::coord::Shift;
use plotters::prelude::*;

use super::*;
use crate::stats::{median, spearman};
use crate::style_mesh;

/// A cell needs at least this many usable trials for its ρ to mean anything.
const MIN_TRIALS: usize = 10;
const N_BINS: usize = 6;

pub struct RhoManProj<'a> {
    cells: &'a CellMap,
    ns: Vec<usize>,
}

impl<'a> RhoManProj<'a> {
    pub fn new(cells: &'a CellMap, ns: &[usize]) -> Self {
        Self {
            cells,
            ns: ns.to_vec(),
        }
    }

    /// (median κ, ρ_man-proj) for every cell of one geometry and sample size.
    fn points(&self, metric: &str, geometry: &str, n: usize) -> Vec<(f64, f64)> {
        let manifold = format!("{metric}_manifold");
        let mut pts = Vec::new();
        for (key, recs) in self.cells {
            if key.geometry != geometry || key.n != n {
                continue;
            }
            let (mut proj, mut man, mut ks) = (Vec::new(), Vec::new(), Vec::new());
            for r in recs {
                let (Some(pv), Some(mv), Some(kv)) =
                    (r.objective(metric), r.objective(&manifold), r.kappa())
                else {
                    continue;
                };
                if pv.is_finite() && mv.is_finite() {
                    proj.push(pv);
                    man.push(mv);
                    ks.push(kv);
                }
            }
            if proj.len() < MIN_TRIALS {
                continue;
            }
            if let (Some((rho, _)), Some(mk)) = (spearman(&man, &proj), median(&ks)) {
                if rho.is_finite() {
                    pts.push((mk, rho));
                }
            }
        }
        pts
    }

    pub fn has_data(&self) -> bool {
        PAIRED_METRICS.iter().any(|m| {
            CURVED
                .iter()
                .any(|g| self.ns.iter().any(|n| !self.points(m, g, *n).is_empty()))
        })
    }
}

impl Figure for RhoManProj<'_> {
    fn name(&self) -> String {
        "exp4_rho_man_proj".to_string()
    }

    fn size(&self) -> (u32, u32) {
        (400 * PAIRED_METRICS.len() as u32, 480)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            "Experiment 4 — manifold vs projected rank agreement vs curvature (points = cells, lines = binned median)",
            ("sans-serif", 20).into_font().color(&OK_BLACK),
        )?;
        let (legend, grid) = root.split_vertically(32);

        let mut entries: Vec<LegendEntry> = Vec::new();
        for geometry in CURVED {
            for (k, n) in self.ns.iter().enumerate() {
                let e = LegendEntry::new(format!("{geometry} N={n}"), geometry_color(geometry));
                entries.push(if k == 0 { e } else { e.secondary() });
            }
        }
        draw_legend(&legend, &entries)?;

        let panels = grid.split_evenly((1, PAIRED_METRICS.len()));
        for (m_idx, metric) in PAIRED_METRICS.iter().enumerate() {
            // A shared x range across geometries keeps the panels comparable.
            let all_x: Vec<f64> = CURVED
                .iter()
                .flat_map(|g| {
                    self.ns
                        .iter()
                        .flat_map(move |n| self.points(metric, g, *n).into_iter().map(|p| p.0))
                })
                .collect();
            let (x_lo, x_hi) =
                snap_to_decades(padded_log_range(&all_x, 0.08).unwrap_or((1e-3, 1e1)));

            let mut chart = ChartBuilder::on(&panels[m_idx])
                .margin(10)
                .caption(*metric, ("sans-serif", 15).into_font())
                .x_label_area_size(52)
                .y_label_area_size(if m_idx == 0 { 66 } else { 34 })
                .build_cartesian_2d((x_lo..x_hi).log_scale(), -0.8f64..1.1f64)?;

            style_mesh!(chart.configure_mesh())
                .x_desc("median κ (cell)")
                .y_desc(if m_idx == 0 { "ρ_man-proj" } else { "" })
                .x_label_formatter(&log_tick)
                .draw()?;

            // Reference line at perfect agreement.
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x_lo, 1.0), (x_hi, 1.0)],
                RGBColor(187, 187, 187).stroke_width(1),
            )))?;

            for geometry in CURVED {
                let color = geometry_color(geometry);
                for (k, n) in self.ns.iter().enumerate() {
                    let pts = self.points(metric, geometry, *n);
                    if pts.is_empty() {
                        continue;
                    }
                    // Faint cell cloud ...
                    let first_n = k == 0;
                    chart.draw_series(
                        pts.iter()
                            .map(|(x, y)| marker(*x, *y, 3, color.mix(0.25).filled(), first_n)),
                    )?;

                    // ... plus the binned-median trend.
                    let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
                    let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
                    let (bx, by) = binned_median(&xs, &ys, N_BINS);
                    if bx.is_empty() {
                        continue;
                    }
                    let line: Vec<(f64, f64)> =
                        bx.iter().copied().zip(by.iter().copied()).collect();
                    let style = color.mix(0.95).stroke_width(3);
                    if first_n {
                        chart.draw_series(LineSeries::new(line.clone(), style))?;
                    } else {
                        // The second N is dashed, so overlaid sizes stay apart.
                        chart.draw_series(DashedLineSeries::new(line.clone(), 8, 6, style))?;
                    }
                    chart.draw_series(
                        line.iter()
                            .map(|(x, y)| marker(*x, *y, 5, color.filled(), first_n)),
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Circle for the first sample size, triangle for the second — the marker
/// distinction the Python figure used.
fn marker<DB: DrawingBackend>(
    x: f64,
    y: f64,
    size: i32,
    style: ShapeStyle,
    circle: bool,
) -> DynElement<'static, DB, (f64, f64)> {
    if circle {
        Circle::new((x, y), size, style).into_dyn()
    } else {
        TriangleMarker::new((x, y), size, style).into_dyn()
    }
}
