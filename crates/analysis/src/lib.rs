//! Post-hoc analysis of the qParEGO sweeps: Pareto fronts, hypervolume, and the
//! statistics behind the thesis result figures.
//!
//! This is the Rust port of what used to be `pareto_utils.py` / `hv_stats.py` /
//! `hv_aggregate.py` / `analyze_experiments.py`. It exists as its own crate so
//! the numeric half stays dependency-light enough to build and run on a bare
//! cluster compute node (no Python, no venv, no numpy — see `bin/hv.rs`), while
//! the figure half is feature-gated behind `plots` because plotters needs a
//! system font stack.
//!
//! The ten qParEGO objectives (see `crates/optimizer/src/pareto.rs` ::
//! `default_pareto_metrics`) are, in canonical order, the 2D (post-projection)
//! and manifold (pre-projection) variants of five DR-quality metrics. Two of
//! them (normalised stress) are minimised; the rest are maximised. Everything
//! here works in an *oriented* space where every objective is mapped into
//! `[0, 1]` with higher = better, so a fixed hypervolume reference point at the
//! origin and a unit reference box are exact.

pub mod aggregate;
pub mod cell;
pub mod hypervolume;
pub mod objectives;
pub mod pareto;
pub mod records;
pub mod stats;

#[cfg(feature = "plots")]
pub mod figures;

pub use cell::{parse_cell_stem, Cell};
pub use hypervolume::{cell_hypervolume, monte_carlo_hypervolume, HvSummary};
pub use objectives::{oriented_matrix, oriented_value, OBJECTIVES};
pub use pareto::{pareto_front_mask, pareto_front_records};
pub use records::{load_jsonl, trial_records, TrialRecord};
