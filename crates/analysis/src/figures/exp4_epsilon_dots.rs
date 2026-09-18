//! Experiment 4 (`ablation-results`) — the ε-indicator companion to the R2
//! gain dot plot.
//!
//! [`EpsilonDots`] is the figure this module draws: **the same rows, panels
//! and sub-rows as [`super::exp4_gain_dots::GainDots`], with each setting's
//! sub-row carrying the binary additive ε-indicator against `all_off` in
//! both directions as a dumbbell on one log axis.** The R2 figure answers
//! "did the setting help" under a preference model — `S = 5`, the regions,
//! the mass threshold are all choices — and the ε-indicator is the
//! parameter-free cross-check the thesis reports beside it (`indicators.rs`,
//! Zitzler et al. 2003). Drawing the two in one layout is what lets a reader
//! set them under each other and check, row by row, that the R2 verdict is
//! not an artefact of the preference model.
//!
//! ### The numbers are `r2 compare`'s
//!
//! The rows of `results/r2_epsilon_<space>.jsonl` at one N
//! ([`EpsilonRow`]), times [`dot_panels::SCALE`]: `I_ε+(setting, all_off)`
//! and `I_ε+(all_off, setting)`, in objective units of the oriented space.
//! Δε is **not** read from the table; it is the pair. `rms_anchored` is not
//! drawn ([`dot_panels::EXCLUDED`]), and is not in the table anyway.
//!
//! ### Both directions, always
//!
//! The indicator is asymmetric, and when two fronts cross neither direction
//! alone settles the comparison — the rule every ε consumer in this crate
//! follows. So a sub-row is a **dumbbell**: a hollow marker at
//! `I(setting, all_off)` — how far the setting's front must be shifted to
//! cover the baseline's — a filled one at `I(all_off, setting)`, and the
//! segment between them. The reading: the **filled marker right of the
//! hollow one** means the baseline needs the larger shift, i.e. the setting
//! served the front better — the same direction as a positive Δε and ΔR2;
//! a short segment is two fronts that trade blows evenly; both markers far
//! right is two fronts that cross badly both ways, which a single Δε would
//! hide as a small number.
//!
//! ### A plain log axis
//!
//! Both directions are non-negative and, at N=5000, span three decades
//! (0.3 to 354 on the ×1000 scale), so the axis is an ordinary `log10` one
//! ([`LogTicks`] over [`padded_log_range`]), shared by the three panels;
//! no sign, no gutter. A value **at or below zero** — an outright cover,
//! the strongest verdict the indicator gives; one row at N=5000 — has no
//! place on a log axis and is **drawn at the axis' lower bound**, so it
//! stays visible as the extreme it is rather than being dropped.
//!
//! ### What is not drawn
//!
//! Front sizes. The indicator is blind to cardinality, and the crate's rule
//! is to report the sizes beside it; the ε table carries them, and 96
//! sub-rows of `n/m` would be clutter, so this figure leaves them to the
//! table and the caption. Nothing else differs from the R2 figure — the
//! layout is [`dot_panels`]'s.

use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use fitting_core::cast::to_i32;

use super::dot_panels::{
    canvas, collect_rows, draw_marker, draw_na, draw_panels, draw_settings_legend, Row, Shape, DOT,
    LEGEND_ROW, MARGIN, SCALE, SETTINGS,
};
use super::{padded_log_range, setting_color, Figure, LogTicks, Res, OK_BLACK};
use crate::indicators::EpsilonRow;

/// One sub-row's pair: `(I(setting, all_off), I(all_off, setting))`, scaled.
pub type Pair = (f64, f64);

/// The per-setting ε-indicator dot plot at one N.
pub struct EpsilonDots {
    n: usize,
    rows: Vec<Row<Pair>>,
    axis: LogTicks,
}

/// Padding of the log range, as a fraction of its width in decades.
const PAD: f64 = 0.05;

impl EpsilonDots {
    /// The figure over `r2 compare`'s rows at N. Rows are the datasets with
    /// at least one drawn pair, in chapter order.
    #[must_use]
    pub fn new(rows: &[EpsilonRow], n: usize) -> Self {
        let out = collect_rows(|dataset, geometry, setting| {
            rows.iter()
                .find(|r| {
                    r.n == n
                        && r.dataset == dataset
                        && r.geometry == geometry
                        && r.setting == setting
                })
                .map(|r| {
                    (
                        r.eps_setting_vs_baseline * SCALE,
                        r.eps_baseline_vs_setting * SCALE,
                    )
                })
                .filter(|(a, b)| a.is_finite() && b.is_finite())
        });
        let values: Vec<f64> = out
            .iter()
            .flat_map(Row::drawn)
            .flat_map(|(a, b)| [a, b])
            .collect();
        // At least a decade, so an axis exists when every ε is alike.
        let (lo, hi) = padded_log_range(&values, PAD).unwrap_or((1.0, 10.0));
        let hi = hi.max(lo * 10.0);
        Self {
            n,
            rows: out,
            axis: LogTicks::new((lo, hi), 6),
        }
    }

    /// True when there is at least one row to draw.
    #[must_use]
    pub fn has_data(&self) -> bool {
        !self.rows.is_empty()
    }

    /// The rows, top to bottom.
    #[must_use]
    pub fn rows(&self) -> &[Row<Pair>] {
        &self.rows
    }

    /// The shared x axis.
    #[must_use]
    pub fn axis(&self) -> &LogTicks {
        &self.axis
    }

    /// Where *v* sits on the axis: itself, or the lower bound for a value
    /// a log axis cannot place — an outright cover.
    #[must_use]
    pub fn placed(&self, v: f64) -> f64 {
        if v > 0.0 {
            v
        } else {
            self.axis.range().start
        }
    }
}

impl Figure for EpsilonDots {
    fn name(&self) -> String {
        format!("exp4_epsilon_dots_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        canvas(self.rows.len(), 2)
    }

    fn draw<DB: DrawingBackend>(&self, root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        let lo = self.axis.range().start;
        draw_panels(
            root,
            &self.rows,
            2,
            &self.axis,
            &|v| self.axis.label(v),
            None,
            "\u{3b5} \u{d7} 1000",
            draw_legend,
            |col, chart, area, origin, row| {
                for ((setting, shape, off), value) in SETTINGS.iter().zip(row.panel(col)) {
                    let y = row.centre() + off;
                    let Some((sb, bs)) = value else {
                        let (px, py) = chart.backend_coord(&(lo, y));
                        draw_na(area, (px - origin.0, py - origin.1))?;
                        continue;
                    };
                    let (sb, bs) = (self.placed(*sb), self.placed(*bs));
                    let color = setting_color(setting);
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(sb, y), (bs, y)],
                        color.mix(0.55).stroke_width(2),
                    )))?;
                    // The hollow end first, so where the two coincide the
                    // filled one — the setting's own shift — is on top.
                    let (px, py) = chart.backend_coord(&(sb, y));
                    draw_marker(area, (px - origin.0, py - origin.1), *shape, color, false)?;
                    let (px, py) = chart.backend_coord(&(bs, y));
                    draw_marker(area, (px - origin.0, py - origin.1), *shape, color, true)?;
                }
                Ok(())
            },
        )
    }
}

/// Two legend rows: the settings, then what hollow and filled mean.
fn draw_legend<DB: DrawingBackend>(area: &DrawingArea<DB, Shift>) -> Res
where
    DB::ErrorType: 'static,
{
    let (settings, directions) = area.split_vertically(LEGEND_ROW);
    draw_settings_legend(&settings)?;

    let (width, height) = directions.dim_in_pixel();
    let font = ("sans-serif", 14).into_font().color(&OK_BLACK);
    let cy = to_i32(f64::from(height) / 2.0);
    let slot = to_i32((f64::from(width) - 2.0 * f64::from(MARGIN)) / 2.0);
    let entries = [
        (false, "hollow: I(setting, all_off)"),
        (true, "filled: I(all_off, setting)"),
    ];
    for (i, (filled, text)) in entries.iter().enumerate() {
        let x0 =
            to_i32(f64::from(MARGIN)) + DOT + to_i32(fitting_core::cast::count_to_f64(i)) * slot;
        draw_marker(&directions, (x0, cy), Shape::Circle, OK_BLACK, *filled)?;
        directions.draw(&Text::new(
            *text,
            (x0 + 12, cy),
            font.clone().pos(Pos::new(HPos::Left, VPos::Center)),
        ))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eps(dataset: &str, geometry: &str, setting: &str, n: usize, sb: f64, bs: f64) -> EpsilonRow {
        EpsilonRow {
            dataset: dataset.into(),
            geometry: geometry.into(),
            n,
            setting: setting.into(),
            n_front_setting: 10,
            n_front_baseline: 12,
            eps_setting_vs_baseline: sb,
            eps_baseline_vs_setting: bs,
            delta_eps: bs - sb,
            setting_covers_baseline: sb <= 0.0,
            baseline_covers_setting: bs <= 0.0,
        }
    }

    #[test]
    fn rows_follow_chapter_order_and_keep_the_pair_in_order() {
        let rows = vec![
            eps("mnist", "euclidean", "norm_only", 5000, 0.025, 0.150),
            eps("grid", "hyperbolic", "all_free", 5000, 0.005, 0.014),
            eps("grid", "hyperbolic", "rms_anchored", 5000, 0.9, 0.9),
            eps("tree", "spherical", "global_only", 1000, 0.5, 0.5),
        ];
        let fig = EpsilonDots::new(&rows, 5000);
        let datasets: Vec<&str> = fig.rows().iter().map(Row::dataset).collect();
        assert_eq!(
            datasets,
            ["grid", "mnist"],
            "synthetic first, other N skipped"
        );
        // (setting-vs-baseline, baseline-vs-setting), in that order, ×1000.
        assert_eq!(
            fig.rows()[1].value("euclidean", "norm_only"),
            Some((25.0, 150.0))
        );
        assert_eq!(
            fig.rows()[0].value("hyperbolic", "all_free"),
            Some((5.0, 14.0))
        );
        assert_eq!(fig.rows()[0].value("hyperbolic", "rms_anchored"), None);
        assert_eq!(
            fig.rows()[0].value("hyperbolic", "norm_only"),
            None,
            "missing, not zero"
        );
        assert_eq!(fig.name(), "exp4_epsilon_dots_N5000");
    }

    #[test]
    fn a_cover_sits_at_the_lower_bound() {
        let rows = vec![
            eps("sphere", "hyperbolic", "centering_only", 5000, 0.25, 0.0),
            eps("sphere", "hyperbolic", "all_free", 5000, 0.002, 0.03),
        ];
        let fig = EpsilonDots::new(&rows, 5000);
        let lo = fig.axis().range().start;
        assert!(
            lo > 0.0 && lo < 2.0,
            "range starts under the smallest positive value"
        );
        assert!((fig.placed(0.0) - lo).abs() < 1e-12);
        assert!((fig.placed(250.0) - 250.0).abs() < 1e-12);
        assert!(fig.axis().range().end > 250.0);
    }

    #[test]
    fn a_flat_table_still_has_a_decade_of_axis() {
        let rows = vec![eps("grid", "euclidean", "all_free", 5000, 0.01, 0.01)];
        let fig = EpsilonDots::new(&rows, 5000);
        let r = fig.axis().range();
        assert!(r.end / r.start >= 10.0);
    }
}
