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
//! The thesis reports a dimensionless curvature **κ = |K|·R_rms²** (|sectional
//! curvature| times the squared RMS geodesic radius the configuration occupies),
//! and that is the gauge used throughout — for embeddings and for the detector
//! alike, so the two land on one axis.
//!
//! For a detected geometry, `K = ±1/r*²` comes from the fitted radius and
//! `R_rms` from the configuration that radius implies: each `Z(r*)` is the Gram
//! matrix of its own model, so its retained eigen-block *is* that configuration
//! (`curvature_detection::reconstruct`). `kappa_data` is therefore
//! `|K|·R_rms²` measured on the reconstruction, not `|K|·d_rms²` measured on the
//! input — the name is kept for continuity with the file it writes.
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
    detect_geometry, detect_hyperbolic, fit_euclidean, fit_hyperbolic, fit_spherical,
    reconstruct_hyperbolic, reconstruct_spherical,
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
    /// Dimensionless curvature `|curvature| · R_rms²` of the configuration the
    /// verdict's fit implies — the quantity that shares an axis with the
    /// embeddings' κ. `0.0` when the verdict is euclidean, where `K = 0` makes
    /// κ vanish on any gauge.
    kappa_data: f64,

    // ── Scale of the data (distance-matrix only) ──
    /// Input RMS pairwise distance `sqrt((1/n(n-1)) Σ_{i≠j} d_ij²)` (thesis d_rms).
    d_rms: f64,
    /// Largest pairwise distance (the diameter used by the Wilson search bounds).
    d_max: f64,

    // ── Wilson spherical fit ──
    sph_radius: f64,
    /// `Σ|λ|` over the residual (non-signature) eigenvalues of `Z(r*)` — the
    /// objective the fit minimises. Carries the units of `Z` (squared distance)
    /// and is extensive in `n`, so compare `sph_residual_normalised` across
    /// datasets, not this.
    sph_residual: f64,
    /// `sph_residual / (n · d_max²)` — the same misfit divided by two constants
    /// of the dataset, which is what makes it comparable across datasets. A
    /// spherical verdict needs this below `SPHERICAL_RESIDUAL_MAX`.
    sph_residual_normalised: f64,
    sph_at_upper_bound: bool,
    /// Angular extent `d_max / r*`; a spherical verdict needs this ≳ 2.5.
    sph_angular_extent: f64,
    /// `+1/r*²` — curvature implied by the spherical radius.
    sph_curvature: f64,
    /// `|+1/r*²| · R_rms²` of the spherical reconstruction — κ under the
    /// spherical hypothesis regardless of verdict.
    sph_kappa: f64,
    /// RMS geodesic radius of the spherical reconstruction, the gauge length
    /// behind `sph_kappa`.
    sph_r_rms: f64,

    // ── Wilson hyperbolic fit ──
    hyp_radius: f64,
    /// `Σ|λ|` over the residual eigenvalues of `Z(r*)`; see `sph_residual`.
    hyp_residual: f64,
    /// `hyp_residual / (n · d_max²)`; see `sph_residual_normalised`. Nothing
    /// thresholds this — the hyperbolic verdict comes from the δ(k) test — but
    /// it is the field to compare across datasets.
    hyp_residual_normalised: f64,
    /// Whether `r*` sits at the flat-ward cap, i.e. the radius at which the
    /// reconstruction's κ falls to `HYPERBOLIC_KAPPA_MIN`. When set,
    /// `hyp_kappa` is that floor and carries no information: the fit wanted a
    /// flatter model than the search allows.
    hyp_at_upper_bound: bool,
    /// `-1/r*²` — curvature implied by the hyperbolic radius.
    hyp_curvature: f64,
    /// `|-1/r*²| · R_rms²` of the hyperbolic reconstruction — κ under the
    /// hyperbolic hypothesis regardless of verdict.
    hyp_kappa: f64,
    /// RMS geodesic radius of the hyperbolic reconstruction, the gauge length
    /// behind `hyp_kappa`.
    hyp_r_rms: f64,

    // ── Wilson euclidean fit ──
    // The flat null model: `B = −J D∘D J / 2`, the `r → ∞` limit of both curved
    // kernels (Wilson et al. 2014, eqs. 24–26). It carries no free radius, so
    // there is no search, nothing to pin, and no reconstruction to gauge — which
    // is why it has three fields where the curved arms have seven. Reported so
    // the curved residuals can be read as a nested-model question ("does
    // allowing curvature buy a strictly better fit than flat?") rather than as
    // bare numbers.
    /// `Σ|λ|` over the residual eigenvalues of the classical-MDS kernel.
    euc_residual: f64,
    /// `euc_residual / (n · d_max²)`; the arm-comparable form, same gauge as
    /// `sph_residual_normalised` and `hyp_residual_normalised`.
    euc_residual_normalised: f64,
    /// `0` exactly: the flat model has `K = 0`, so `κ = |K| · R_rms²` vanishes
    /// on any gauge. Present so all three arms expose the same κ field.
    euc_kappa: f64,

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
    /// `|delta_curvature| · R_rms²` — κ from the Gromov δ estimate, gauged on
    /// the hyperbolic configuration that curvature implies. `0.0` when the δ
    /// test does not call the data hyperbolic.
    delta_kappa: f64,
}

/// Input RMS pairwise distance `sqrt((1 / n(n-1)) Σ_{i≠j} d_ij²)`, the thesis
/// reported for scale alongside the fits. The flat matrix's diagonal is zero,
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

    let verdict = detect_geometry(distances, n, embed_dim);
    let spherical = fit_spherical(distances, n, embed_dim);
    let hyperbolic = fit_hyperbolic(distances, n, embed_dim);
    let euclidean = fit_euclidean(distances, n, embed_dim);
    let hyp_delta = detect_hyperbolic(distances, n);

    let sph_curvature = 1.0 / (spherical.radius * spherical.radius);
    let hyp_curvature = -1.0 / (hyperbolic.radius * hyperbolic.radius);

    // κ is gauged by R_rms, which is a property of the *configuration* a fit
    // implies rather than of the radius alone — so each arm is reconstructed at
    // its fitted radius. That is an eigendecomposition, not a second fit: each
    // `Z(r*)` is the Gram matrix of its own model.
    let sph_rec = reconstruct_spherical(distances, n, embed_dim, spherical.radius);
    let hyp_rec = reconstruct_hyperbolic(distances, n, embed_dim, hyperbolic.radius);
    let (sph_kappa, hyp_kappa) = (sph_rec.kappa(n), hyp_rec.kappa(n));

    // The δ estimate names a curvature but no radius-fitting procedure, so its κ
    // is gauged on the hyperbolic configuration that curvature implies. Zero
    // curvature means the δ test did not call the data hyperbolic, and there is
    // nothing to reconstruct.
    let delta_kappa = if hyp_delta.curvature < 0.0 {
        let r = 1.0 / (-hyp_delta.curvature).sqrt();
        reconstruct_hyperbolic(distances, n, embed_dim, r).kappa(n)
    } else {
        0.0
    };

    // The verdict's κ is whichever arm it chose; euclidean has K = 0, so κ = 0
    // on any gauge.
    let kappa_data = match verdict.best_geometry {
        "spherical" => sph_kappa,
        "hyperbolic" => hyp_kappa,
        _ => 0.0,
    };

    let record = DetectionRecord {
        dataset: dataset_name.to_string(),
        n_samples: n,
        embed_dim,

        geometry_detected: verdict.best_geometry,
        curvature: verdict.curvature,
        kappa_data,

        d_rms,
        d_max,

        sph_radius: spherical.radius,
        sph_residual: spherical.residual,
        sph_residual_normalised: spherical.residual_normalised,
        sph_at_upper_bound: spherical.at_upper_bound,
        sph_angular_extent: if spherical.radius > 0.0 {
            d_max / spherical.radius
        } else {
            0.0
        },
        sph_curvature,
        sph_kappa,
        sph_r_rms: sph_rec.r_rms(n),

        hyp_radius: hyperbolic.radius,
        hyp_residual: hyperbolic.residual,
        hyp_residual_normalised: hyperbolic.residual_normalised,
        hyp_at_upper_bound: hyperbolic.at_upper_bound,
        hyp_curvature,
        hyp_kappa,
        hyp_r_rms: hyp_rec.r_rms(n),

        euc_residual: euclidean.residual,
        euc_residual_normalised: euclidean.residual_normalised,
        euc_kappa: 0.0,

        delta_is_hyperbolic: hyp_delta.is_hyperbolic,
        delta_tail_slope: hyp_delta.tail_slope,
        delta_saturated: hyp_delta.saturated_delta,
        delta_saturated_normalised: hyp_delta.saturated_delta_normalised,
        delta_curvature: hyp_delta.curvature,
        delta_kappa,
    };

    println!(
        "detect '{}' (n={}): {} | K={:+.4} κ_data={:.4} | sph r*={:.3} res={:.3e}{} | hyp r*={:.3} res={:.3e} | δ-slope={:.3} sat_δ={:.3} hyp={}",
        dataset_name,
        n,
        record.geometry_detected,
        record.curvature,
        record.kappa_data,
        record.sph_radius,
        record.sph_residual_normalised,
        if record.sph_at_upper_bound { "(pinned)" } else { "" },
        record.hyp_radius,
        record.hyp_residual_normalised,
        record.delta_tail_slope,
        record.delta_saturated,
        record.delta_is_hyperbolic,
    );

    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)
    {
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
