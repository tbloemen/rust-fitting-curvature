//! Curvature / geometry detection from a pairwise distance matrix.
//!
//! Submodules:
//!
//! - [`signature`] — Wilson et al. (2014) radius-of-curvature fit by a
//!   constant-curvature Gram matrix and signature residual, plus the
//!   top-level [`signature::detect_geometry`] that labels a dataset
//!   Euclidean / spherical / hyperbolic.  Its hyperbolic decision is
//!   delegated to the growing-ball test in [`gromov_ball_curve`].
//! - [`histogram`] — the naive shell-density (geodesic-sphere area)
//!   detector; a baseline that shows curvature *sign* can't be read off
//!   the residuals (see [`gromov_ball_curve`] for the principled test).
//! - [`gromov`] — shared low-level primitives for the Gromov 4-point δ
//!   estimators (`quad_delta`, `median_pairwise_distance`,
//!   `four_distinct`).
//! - [`old_detection`] — the legacy global 90th-percentile Gromov δ
//!   scalar (`gromov_hyperbolicity`), still used by the histogram tests.
//! - [`gromov_ball_curve`] — the NeTS-proposal growing-ball δ(k) curve
//!   and the saturation-based [`gromov_ball_curve::detect_hyperbolic`].
//!
//! The signature and growing-ball APIs are re-exported here for
//! convenience (`curvature_detection::detect_geometry`,
//! `::gromov_delta_curve`, …).  The histogram detector and the legacy
//! scalar are reached through their submodule paths to avoid name
//! clashes (both modules also define a `detect_*`).

pub mod gromov;
pub mod gromov_ball_curve;
pub mod histogram;
pub mod old_detection;
pub mod signature;

pub use gromov_ball_curve::*;
pub use signature::*;
