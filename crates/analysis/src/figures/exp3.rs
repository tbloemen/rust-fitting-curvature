//! Experiment 3 (`curvature-tuning-results`) — can curvature be tuned as its
//! own hyperparameter?
//!
//! **A skeleton: nothing is drawn yet.** The module is wired into
//! `bin/figures.rs` so `--exp 3` has a slot, and [`KappaLanding::has_data`]
//! returns `false`, so the driver skips it and no empty file is written.
//!
//! The question this figure has to answer: `curvature_magnitude` is a searched
//! hyperparameter, so given that freedom, **where does the search leave κ**?
//! κ = |`K|·R_rms²` (`@eq:kappa`) is the dimensionless quantity to plot — the
//! numerical curvature `K` alone is not comparable across embeddings, since
//! rescaling an embedding changes `K` without changing κ
//! (`<curvature-tuning-results>`, *Separating Curvature from Embedding Scale*).
//!
//! [`crate::records::TrialRecord::kappa`] reads it per trial and
//! [`super::median_front_kappa`] reduces a cell's front to one value; the
//! log-axis helpers ([`super::padded_log_range`], [`super::snap_to_decades`],
//! [`super::log_tick`]) are kept in `figures/mod.rs` for this figure.
//!
//! Two hazards carried over from the deleted κ figures, both worth pinning
//! before reading a trend off this one:
//!
//! * the κ ≈ 2e-7 spike is *collapsed* embeddings, not near-flat space;
//! * on the sphere `R_rms` is measured from the wrong pole, so κ there is a
//!   pure angular statistic bounded in `[0, π²]` and blind to `|K|`. See the
//!   *One κ, one gauge* section of the crate's `CLAUDE.md`.
//!
//! This is not the deleted `exp3.rs`. That one plotted the front's κ against
//! the data-intrinsic `κ_data`, which is the exploratory chapter's
//! `<curvature-magnitude-results>` — a different question from whether
//! curvature is a useful knob.

use plotters::coord::Shift;
use plotters::prelude::*;

use super::{CellMap, Figure, ObjectiveSpace, Res};

/// Where the search leaves κ when curvature is a free hyperparameter.
pub struct KappaLanding<'a> {
    #[expect(dead_code, reason = "read once the figure is drawn")]
    cells: &'a CellMap,
    n: usize,
    #[expect(dead_code, reason = "read once the figure is drawn")]
    space: ObjectiveSpace,
}

impl<'a> KappaLanding<'a> {
    #[must_use]
    pub fn new(cells: &'a CellMap, n: usize, space: ObjectiveSpace) -> Self {
        Self { cells, n, space }
    }

    /// Always `false` while this is a skeleton, so the driver's
    /// "figures with no data are skipped" rule keeps an empty SVG off disk.
    #[must_use]
    pub fn has_data(&self) -> bool {
        false
    }
}

impl Figure for KappaLanding<'_> {
    fn name(&self) -> String {
        format!("exp3_kappa_landing_N{}", self.n)
    }

    fn size(&self) -> (u32, u32) {
        (1200, 560)
    }

    fn draw<DB: DrawingBackend>(&self, _root: &DrawingArea<DB, Shift>) -> Res
    where
        DB::ErrorType: 'static,
    {
        Ok(())
    }
}
