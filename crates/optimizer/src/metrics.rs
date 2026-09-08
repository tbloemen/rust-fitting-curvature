//! Re-exports of the metric registry, which lives in `fitting-core`.
//!
//! This module used to hold `AllMetrics` (16 fields), `AllMetrics::mean` (one
//! line per field), `enum Metric` (13 variants), `Metric::ALL`, `name`,
//! `direction`, `value` and `from_str` — seven parallel lists of the same
//! metrics in one file, each of which had to be edited to add or remove one.
//! They are now one `impl QualityMetric` block per metric in
//! `fitting_core::metrics`, which `crates/web` and `crates/analysis` read too.

pub use fitting_core::metrics::{Direction, Metric, MetricValues, OBJECTIVES};
