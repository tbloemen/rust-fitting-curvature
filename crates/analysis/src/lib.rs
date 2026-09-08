//! Post-hoc analysis of the qParEGO sweeps: Pareto fronts, the R2 indicator, and
//! the statistics behind the thesis result figures.
//!
//! The six qParEGO objectives (see `crates/optimizer/src/pareto.rs` ::
//! `default_pareto_metrics`) are all measured on the 2D projection, and all
//! bounded in `[0, 1]` by construction. One of them (normalised stress) is
//! minimised; the rest are maximised. Everything here works in an *oriented*
//! space where every objective is mapped into `[0, 1]` with higher = better, so
//! the ideal point is `(1, …, 1)` and the R2 indicator of `r2.rs` measures
//! distance to it under a stated set of weights.
//!
//! The manifold (pre-projection) variants are still recorded on every trial and
//! still read by `figures/exp4.rs`; they are simply no longer objectives. See
//! [`objectives::OBJECTIVES`] for why the set is bounded-and-projected.
//! `indicators.rs` measures the same fronts without any weights at all, as the
//! parameter-free cross-check on that preference model.
//!
//! Nothing in this crate prints: everything that can fail returns [`Error`], and
//! the two binaries render it in one place, by returning it from `main`. The
//! figure half is feature-gated behind `plots` because plotters needs a system
//! font stack.

pub mod aggregate;
pub mod cell;
pub mod error;
pub mod indicators;
pub mod objectives;
pub mod pareto;
pub mod r2;
pub mod records;
pub mod stats;

#[cfg(feature = "plots")]
pub mod figures;

pub use cell::{
    discover_cells, parse_cell_stem, parse_cell_stem_variant, truth_of, Cell, CellFile, Variant,
    SYNTH_TRUTH,
};
pub use error::{Error, IoContext, Result};
pub use indicators::{epsilon_additive, epsilon_pair, EpsilonPair};
pub use objectives::{oriented_matrix, oriented_value, OBJECTIVES};
pub use pareto::{pareto_front_mask, pareto_front_records};
pub use r2::{cell_summary, r2, CellSummary, Weights};
pub use records::{load_jsonl, trial_records, write_jsonl, TrialRecord};
