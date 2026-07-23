//! `--mode detect`: export the curvature-detection diagnostics for a dataset.
//!
//! This runs the full geometry-detection pipeline from `fitting_core`
//! (`curvature_detection`) on a dataset's high-dimensional distance matrix and
//! writes one JSONL record per dataset. It does **not** fit any embedding — it
//! characterises the *data's* intrinsic curvature so the Python analysis can put
//! each dataset on the same dimensionless κ axis as the embeddings.
//!
//! ## What κ_data is
//!
//! The thesis reports a dimensionless curvature κ = |K|·R_rms² for an embedding
//! (|sectional curvature| times the squared RMS geodesic radius it occupies).
//! The data analog (methods §gauge-fixing) is `kappa_data = |K_data|·d_rms² =
//! (d_rms / r*)²`, where `K_data = ±1/r*²` is the signed sectional curvature the
//! detector reports and `d_rms` is the **input RMS pairwise distance**
//! `sqrt((1 / n(n-1)) Σ_{i≠j} d_ij²)` — exactly the definition in the thesis.
//!
//! The record also carries the raw per-model Wilson fits (radius + residual +
//! whether the fit pinned at the flat-ward bound) and the growing-ball Gromov
//! δ(k) saturation diagnostics, so the Python side can recompute κ under either
//! the spherical or hyperbolic hypothesis, or inspect why a verdict was reached,
//! without re-running Rust.

use std::fs::OpenOptions;
use std::io::Write;

use serde::Serialize;

use crate::cli::Args;
use crate::evaluate::Evaluator;
use fitting_core::curvature_detection::{
    detect_geometry, detect_hyperbolic, fit_hyperbolic, fit_spherical,
};

/// One dataset's curvature-detection record (one JSONL line).
#[derive(Debug, Serialize)]
struct DetectionRecord {
    dataset: String,
    n_samples: usize,
    /// Embedding target dimension the Wilson models are fitted at (always 2 here,
    /// matching the optimizer's `infer_geometry`).
    embed_dim: usize,

    // ── The decision the embedding pipeline would act on ──
    /// `"spherical"`, `"hyperbolic"`, or `"euclidean"`.
    geometry_detected: &'static str,
    /// Signed sectional curvature of the verdict (`0.0` for euclidean).
    curvature: f64,
    /// Dimensionless data curvature `|curvature| · d_rms²` — the quantity that
    /// shares an axis with the embeddings' κ. `0.0` when the verdict is euclidean.
    kappa_data: f64,

    // ── Scale of the data (distance-matrix only) ──
    /// Input RMS pairwise distance `sqrt((1/n(n-1)) Σ_{i≠j} d_ij²)` (thesis d_rms).
    d_rms: f64,
    /// Largest pairwise distance (the diameter used by the Wilson search bounds).
    d_max: f64,

    // ── Wilson spherical fit ──
    sph_radius: f64,
    sph_residual: f64,
    sph_at_upper_bound: bool,
    /// Angular extent `d_max / r*`; a spherical verdict needs this ≳ 2.5.
    sph_angular_extent: f64,
    /// `+1/r*²` — curvature implied by the spherical radius.
    sph_curvature: f64,
    /// `|+1/r*²| · d_rms²` — κ under the spherical hypothesis regardless of verdict.
    sph_kappa: f64,

    // ── Wilson hyperbolic fit ──
    hyp_radius: f64,
    hyp_residual: f64,
    hyp_at_upper_bound: bool,
    /// `-1/r*²` — curvature implied by the hyperbolic radius.
    hyp_curvature: f64,
    /// `|-1/r*²| · d_rms²` — κ under the hyperbolic hypothesis regardless of verdict.
    hyp_kappa: f64,

    // ── Growing-ball Gromov δ(k) saturation diagnostics ──
    /// Whether the δ(k) curve saturates (the theoretically-backed hyperbolic gate).
    delta_is_hyperbolic: bool,
    /// Log–log tail slope of δ(k); below ~0.15 ⇒ saturated ⇒ hyperbolic.
    delta_tail_slope: f64,
    /// Saturated (tail-averaged) raw δ.
    delta_saturated: f64,
    /// Saturated δ normalised by the median pairwise distance (scale-free).
    delta_saturated_normalised: f64,
    /// Curvature from `δ = ln(1+√2)/√(−K)` when hyperbolic (`0.0` otherwise).
    delta_curvature: f64,
    /// `|delta_curvature| · d_rms²` — κ from the Gromov δ estimate.
    delta_kappa: f64,
}

/// Input RMS pairwise distance `sqrt((1 / n(n-1)) Σ_{i≠j} d_ij²)`, the thesis
/// `d_rms` in `kappa_data = (d_rms / r*)²`. The flat matrix's diagonal is zero,
/// so summing all n² entries already excludes the i=i terms; dividing by the
/// n(n-1) off-diagonal pair count gives the RMS over distinct pairs.
fn rms_pairwise(distances: &[f64], n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let sum_sq: f64 = distances.iter().map(|d| d * d).sum();
    (sum_sq / ((n as f64) * (n as f64 - 1.0))).sqrt()
}

pub fn run_detect(dataset_name: &str, args: &Args, evaluator: &Evaluator) {
    let n = evaluator.n_points();
    let distances = evaluator.distances();
    let embed_dim = 2;

    let d_max = distances.iter().cloned().fold(0.0_f64, f64::max);
    let d_rms = rms_pairwise(distances, n);
    let d_rms_sq = d_rms * d_rms;

    let verdict = detect_geometry(distances, n, embed_dim);
    let spherical = fit_spherical(distances, n, embed_dim);
    let hyperbolic = fit_hyperbolic(distances, n, embed_dim);
    let hyp_delta = detect_hyperbolic(distances, n);

    let sph_curvature = 1.0 / (spherical.radius * spherical.radius);
    let hyp_curvature = -1.0 / (hyperbolic.radius * hyperbolic.radius);

    let record = DetectionRecord {
        dataset: dataset_name.to_string(),
        n_samples: n,
        embed_dim,

        geometry_detected: verdict.best_geometry,
        curvature: verdict.curvature,
        kappa_data: verdict.curvature.abs() * d_rms_sq,

        d_rms,
        d_max,

        sph_radius: spherical.radius,
        sph_residual: spherical.residual,
        sph_at_upper_bound: spherical.at_upper_bound,
        sph_angular_extent: if spherical.radius > 0.0 {
            d_max / spherical.radius
        } else {
            0.0
        },
        sph_curvature,
        sph_kappa: sph_curvature.abs() * d_rms_sq,

        hyp_radius: hyperbolic.radius,
        hyp_residual: hyperbolic.residual,
        hyp_at_upper_bound: hyperbolic.at_upper_bound,
        hyp_curvature,
        hyp_kappa: hyp_curvature.abs() * d_rms_sq,

        delta_is_hyperbolic: hyp_delta.is_hyperbolic,
        delta_tail_slope: hyp_delta.tail_slope,
        delta_saturated: hyp_delta.saturated_delta,
        delta_saturated_normalised: hyp_delta.saturated_delta_normalised,
        delta_curvature: hyp_delta.curvature,
        delta_kappa: hyp_delta.curvature.abs() * d_rms_sq,
    };

    println!(
        "detect '{}' (n={}): {} | K={:+.4} κ_data={:.4} | sph r*={:.3} ρ={:.3}{} | hyp r*={:.3} ρ={:.3} | δ-slope={:.3} sat_δ={:.3} hyp={}",
        dataset_name,
        n,
        record.geometry_detected,
        record.curvature,
        record.kappa_data,
        record.sph_radius,
        record.sph_residual,
        if record.sph_at_upper_bound { "(pinned)" } else { "" },
        record.hyp_radius,
        record.hyp_residual,
        record.delta_tail_slope,
        record.delta_saturated,
        record.delta_is_hyperbolic,
    );

    let mut file = match OpenOptions::new().create(true).append(true).open(&args.output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("detect: failed to open {} for append: {e}", args.output);
            return;
        }
    };
    match serde_json::to_string(&record) {
        Ok(json) => {
            writeln!(file, "{}", json).ok();
        }
        Err(e) => eprintln!("detect: failed to serialise record: {e}"),
    }
}
