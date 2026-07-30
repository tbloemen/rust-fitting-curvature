//! Experiment 3 — median Pareto-front κ against the data-intrinsic κ_data, and
//! the unanchored-vs-`rms_anchored` κ overlay.

use std::collections::BTreeMap;

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use super::*;
use crate::cell::Cell;
use crate::pareto::pareto_front_records;
use crate::stats::spearman;
use crate::style_mesh;

// ─── κ vs κ_data scatter ──────────────────────────────────────────────────────

pub struct KappaScatter<'a> {
    cells: &'a CellMap,
    kappa_data: BTreeMap<String, KappaData>,
    n: usize,
}

/// One plotted dataset: its intrinsic κ_data, its median front κ, and whether it
/// is a real (circle) or synthetic (square) dataset.
struct Point {
    dataset: String,
    kappa_data: f64,
    kappa: f64,
    real: bool,
}

impl<'a> KappaScatter<'a> {
    pub fn new(cells: &'a CellMap, results_dir: &std::path::Path, n: usize) -> Self {
        Self {
            cells,
            kappa_data: load_kappa_data(results_dir, n),
            n,
        }
    }

    /// True when a κ_data table for this N was found at all.
    pub fn has_kappa_data(&self) -> bool {
        !self.kappa_data.is_empty()
    }

    fn points(&self, geometry: &str) -> Vec<Point> {
        let mut out = Vec::new();
        for dataset in all_datasets() {
            let key = Cell::new("all_off", dataset, self.n, geometry);
            let Some(recs) = self.cells.get(&key) else {
                continue;
            };
            let Some(kd) = self.kappa_data.get(dataset) else {
                continue;
            };
            let (Some(mk), Some(kdata)) = (median_front_kappa(recs), kd.for_geometry(geometry))
            else {
                continue;
            };
            if !kdata.is_finite() || kdata <= 0.0 {
                continue;
            }
            out.push(Point {
                dataset: dataset.to_string(),
                kappa_data: kdata,
                kappa: mk,
                real: REAL_DATASETS.contains(&dataset),
            });
        }
        out
    }
}

impl Figure for KappaScatter<'_> {
    fn name(&self) -> String {
        format!("exp3_kappa_vs_kappadata_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (1200, 560)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            &format!(
                "Experiment 3 — Pareto-favoured vs intrinsic curvature (N={})   (circles = real datasets, squares = synthetic)",
                self.n
            ),
            ("sans-serif", 20).into_font().color(&OK_BLACK),
        )?;
        let panels = root.split_evenly((1, CURVED.len()));

        for (col, geometry) in CURVED.iter().enumerate() {
            let panel = &panels[col];
            let pts = self.points(geometry);

            // Spearman ρ across datasets for this geometry class.
            let mut title = format!("{geometry} embeddings");
            if pts.len() >= 3 {
                let xs: Vec<f64> = pts.iter().map(|p| p.kappa_data).collect();
                let ys: Vec<f64> = pts.iter().map(|p| p.kappa).collect();
                if let Some((rho, p)) = spearman(&xs, &ys) {
                    title += &format!("   (Spearman ρ={rho:+.2}, p={p:.2}, n={})", pts.len());
                }
            } else if !pts.is_empty() {
                title += &format!("   (n={}, too few for ρ)", pts.len());
            }

            let xs: Vec<f64> = pts.iter().map(|p| p.kappa_data).collect();
            let ys: Vec<f64> = pts.iter().map(|p| p.kappa).collect();
            let (x_lo, x_hi) = snap_to_decades(padded_log_range(&xs, 0.12).unwrap_or((0.1, 10.0)));
            let (y_lo, y_hi) = snap_to_decades(padded_log_range(&ys, 0.12).unwrap_or((0.1, 10.0)));

            let mut chart = ChartBuilder::on(panel)
                .margin(10)
                .caption(&title, ("sans-serif", 15).into_font())
                .x_label_area_size(52)
                .y_label_area_size(72)
                .build_cartesian_2d((x_lo..x_hi).log_scale(), (y_lo..y_hi).log_scale())?;

            style_mesh!(chart.configure_mesh())
                .x_desc("data-intrinsic κ_data = |K|·d_rms²")
                .y_desc(if col == 0 {
                    "median Pareto-front κ = |K|·R_rms²"
                } else {
                    ""
                })
                .x_label_formatter(&log_tick)
                .y_label_formatter(&log_tick)
                .draw()?;

            let color = geometry_color(geometry);
            let label_font = ("sans-serif", 12)
                .into_font()
                .color(&RGBColor(51, 51, 51))
                .pos(Pos::new(HPos::Left, VPos::Bottom));

            // Markers are sized and offset in *pixels* (anchored at the data
            // point via EmptyElement) rather than in data units: the κ axes span
            // several decades, so a data-unit marker would be invisible at one
            // end of the panel and a slab at the other.
            let r = 7i32;
            chart.draw_series(pts.iter().filter(|p| p.real).map(|p| {
                EmptyElement::at((p.kappa_data, p.kappa))
                    + Circle::new((0, 0), r, color.mix(0.85).filled())
                    + Circle::new((0, 0), r, OK_BLACK.stroke_width(1))
                    + Text::new(p.dataset.clone(), (r + 3, -r), label_font.clone())
            }))?;
            chart.draw_series(pts.iter().filter(|p| !p.real).map(|p| {
                EmptyElement::at((p.kappa_data, p.kappa))
                    + Rectangle::new([(-r, -r), (r, r)], color.mix(0.85).filled())
                    + Rectangle::new([(-r, -r), (r, r)], OK_BLACK.stroke_width(1))
                    + Text::new(p.dataset.clone(), (r + 3, -r), label_font.clone())
            }))?;
        }
        Ok(())
    }
}

// ─── unanchored vs rms_anchored κ ─────────────────────────────────────────────

pub struct RmsAnchored<'a> {
    cells: &'a CellMap,
    n: usize,
    datasets: Vec<String>,
}

impl<'a> RmsAnchored<'a> {
    pub fn new(cells: &'a CellMap, n: usize) -> Self {
        let datasets = all_datasets()
            .into_iter()
            .filter(|ds| {
                cells
                    .keys()
                    .any(|k| k.dataset == *ds && k.geometry == "hyperbolic")
            })
            .map(|s| s.to_string())
            .collect();
        Self { cells, n, datasets }
    }

    /// Whether any `rms_anchored` run exists at all; the figure is meaningless
    /// without one.
    pub fn has_anchored(&self) -> bool {
        self.cells.keys().any(|k| k.setting == "rms_anchored")
    }

    /// (unanchored κ, anchored κ) over the hyperbolic fronts of one dataset.
    fn kappas(&self, dataset: &str) -> (Vec<f64>, Vec<f64>) {
        let (mut unanchored, mut anchored) = (Vec::new(), Vec::new());
        for (key, recs) in self.cells {
            if key.dataset != dataset || key.n != self.n || key.geometry != "hyperbolic" {
                continue;
            }
            let ks = pareto_front_records(recs)
                .iter()
                .filter_map(|r| r.kappa())
                .collect::<Vec<_>>();
            if key.setting == "rms_anchored" {
                anchored.extend(ks);
            } else {
                unanchored.extend(ks);
            }
        }
        (unanchored, anchored)
    }
}

impl Figure for RmsAnchored<'_> {
    fn name(&self) -> String {
        format!("exp3_rms_anchored_kappa_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (320 * self.datasets.len().max(1) as u32, 420)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let root = root.titled(
            &format!(
                "Experiment 3 — unanchored vs rms_anchored κ (hyperbolic, N={})",
                self.n
            ),
            ("sans-serif", 18).into_font().color(&OK_BLACK),
        )?;
        let (legend, grid) = root.split_vertically(30);
        draw_legend(
            &legend,
            &[
                LegendEntry::new("unanchored", OK_BLUE),
                LegendEntry::new("rms_anchored", OK_ORANGE),
            ],
        )?;

        let panels = grid.split_evenly((1, self.datasets.len().max(1)));
        for (j, dataset) in self.datasets.iter().enumerate() {
            let (unanchored, anchored) = self.kappas(dataset);
            let all: Vec<f64> = unanchored.iter().chain(&anchored).copied().collect();
            let (x_lo, x_hi) = padded_range(&all, 0.05).unwrap_or((0.0, 1.0));

            let un = histogram(&unanchored, x_lo, x_hi, 20);
            let an = histogram(&anchored, x_lo, x_hi, 20);
            let y_hi = un
                .iter()
                .chain(&an)
                .map(|b| b.density)
                .fold(0.0f64, f64::max)
                .max(1e-9)
                * 1.1;

            let mut chart = ChartBuilder::on(&panels[j])
                .margin(8)
                .caption(dataset, ("sans-serif", 14).into_font())
                .x_label_area_size(42)
                .y_label_area_size(if j == 0 { 60 } else { 30 })
                .build_cartesian_2d(x_lo..x_hi, 0.0..y_hi)?;
            style_mesh!(chart.configure_mesh())
                .x_desc("κ")
                .y_desc(if j == 0 { "density" } else { "" })
                .x_labels(4)
                .draw()?;

            draw_histogram(&mut chart, &un, OK_BLUE, 0.5)?;
            draw_histogram(&mut chart, &an, OK_ORANGE, 0.5)?;
        }
        Ok(())
    }
}

// ─── Histogram helpers (shared with Exp 5) ────────────────────────────────────

/// One histogram bar: its span and its *density* height (matplotlib's
/// `density=True`: count / (N · bin width), so unequal log-spaced bins are
/// comparable).
pub struct Bin {
    pub lo: f64,
    pub hi: f64,
    pub density: f64,
}

/// Equal-width bins over `[lo, hi]`.
pub fn histogram(values: &[f64], lo: f64, hi: f64, n_bins: usize) -> Vec<Bin> {
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| lo + (hi - lo) * i as f64 / n_bins as f64)
        .collect();
    bin_by_edges(values, &edges)
}

/// Log-spaced bins over `[lo, hi]`; both bounds must be positive.
pub fn log_histogram(values: &[f64], lo: f64, hi: f64, n_bins: usize) -> Vec<Bin> {
    let (a, b) = (lo.log10(), hi.log10());
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| 10f64.powf(a + (b - a) * i as f64 / n_bins as f64))
        .collect();
    bin_by_edges(values, &edges)
}

fn bin_by_edges(values: &[f64], edges: &[f64]) -> Vec<Bin> {
    let n = values.len();
    if n == 0 || edges.len() < 2 {
        return Vec::new();
    }
    let mut counts = vec![0usize; edges.len() - 1];
    for v in values {
        if !v.is_finite() {
            continue;
        }
        // The last bin is closed on the right, as in numpy's histogram.
        let mut k = counts.len();
        for (i, w) in edges.windows(2).enumerate() {
            if *v >= w[0] && (*v < w[1] || (i == counts.len() - 1 && *v <= w[1])) {
                k = i;
                break;
            }
        }
        if k < counts.len() {
            counts[k] += 1;
        }
    }
    edges
        .windows(2)
        .zip(counts)
        .map(|(w, c)| Bin {
            lo: w[0],
            hi: w[1],
            density: if w[1] > w[0] {
                c as f64 / (n as f64 * (w[1] - w[0]))
            } else {
                0.0
            },
        })
        .collect()
}

/// Draw filled, semi-transparent bars so overlaid histograms stay readable.
pub fn draw_histogram<DB, X, Y>(
    chart: &mut ChartContext<DB, Cartesian2d<X, Y>>,
    bins: &[Bin],
    color: RGBColor,
    alpha: f64,
) -> Res
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
    X: plotters::coord::ranged1d::Ranged<ValueType = f64>,
    Y: plotters::coord::ranged1d::Ranged<ValueType = f64>,
{
    chart.draw_series(
        bins.iter()
            .filter(|b| b.density > 0.0)
            .map(|b| Rectangle::new([(b.lo, 0.0), (b.hi, b.density)], color.mix(alpha).filled())),
    )?;
    Ok(())
}
