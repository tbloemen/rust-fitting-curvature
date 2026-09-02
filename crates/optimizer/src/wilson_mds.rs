//! `--mode wilson-mds`: score the Wilson fit as an embedding.
//!
//! `--mode detect` reports what the curvature detector *thinks* about a
//! dataset — a radius, a residual, a verdict. Those numbers live in their own
//! units and cannot be set beside the Pareto fronts the sweeps produce: the
//! signature residual `ρ` measures one model's misfit in squared-distance
//! units, while the R2 indicator measures a *set* of trade-off solutions by
//! Tchebycheff distance to an ideal point. Putting them in adjacent columns
//! would compare nothing.
//!
//! This mode closes that gap. Each `Z(r)` the signature criterion scores is the
//! Gram matrix of the model it tests for, so its retained eigen-block is an
//! embedding (`curvature_detection::reconstruct`). Reconstruct it, then score
//! it with [`crate::evaluate::metrics_from_embedding`] — the *same* function
//! that scores every t-SNE trial, so the same neighbourhood size, the same
//! projection, the same metric code. The result is one point in the same
//! ten-objective space as every point on every Pareto front, which makes the
//! whole existing comparison toolkit apply to it unchanged: R2, the
//! ε-indicator, Pareto dominance.
//!
//! ## The question this makes answerable
//!
//! Experiment 1 asks whether curvature-matched t-SNE beats the Euclidean
//! baseline. With a Wilson point in the same space, a sharper question comes
//! for free: does a *learned* curvature-aware embedding beat the *closed-form*
//! constant-curvature MDS the detector hands you for nothing? A front that
//! already covers the Wilson point has earned its thousand trials; one that
//! does not has been beaten by an eigendecomposition.
//!
//! ## Reading the output
//!
//! One JSONL record per (dataset, geometry), appended — three per dataset, one
//! for each arm, including the mismatched ones, because the off-diagonal is
//! half the experiment.
//!
//! `kappa = |K| · R_rms²` is gauged by the reconstruction's RMS geodesic radius
//! from the origin — the same formula `TrialRecord::kappa()` computes for an
//! embedding, and the one gauge used throughout the repository, so this κ shares
//! an axis with the fronts and with `--mode detect`.

use std::fs::OpenOptions;
use std::io::Write;

use serde::Serialize;

use crate::cli::Args;
use crate::evaluate::{metrics_from_embedding, Evaluator};
use crate::metrics::AllMetrics;
use fitting_core::curvature_detection::{
    fit_euclidean, fit_hyperbolic, fit_spherical, reconstruct_euclidean, reconstruct_hyperbolic,
    reconstruct_spherical, Reconstruction, WilsonFit,
};

/// Embedding target dimension the Wilson models are fitted at — 2, matching
/// `--mode detect` and the optimizer's embeddings.
const EMBED_DIM: usize = 2;

/// One reconstructed arm (one JSONL line).
#[derive(Debug, Serialize)]
struct WilsonMdsRecord {
    dataset: String,
    n_samples: usize,
    /// `"spherical"`, `"hyperbolic"`, or `"euclidean"` — the model that was
    /// fitted and reconstructed, not a verdict about the data.
    geometry: &'static str,
    embed_dim: usize,

    // ── The fit ──
    /// Best-fit radius. `null` for the euclidean arm, which has no radius:
    /// flat space carries no curvature parameter.
    wilson_radius: Option<f64>,
    /// `Σ|λ|` over the non-signature eigenvalues of `Z(r*)`. Units of `Z`
    /// (squared distance) and extensive in `n`.
    wilson_residual: f64,
    /// `wilson_residual / (n · d_max²)` — the cross-dataset-comparable form,
    /// and the `ρ` of the thesis table.
    wilson_residual_normalised: f64,
    /// Whether `r*` pinned at the flat-ward edge of its search window. When
    /// set, the radius is the bound rather than a measurement, and `curvature`
    /// and `kappa` below inherit that — `kappa` is then
    /// `HYPERBOLIC_KAPPA_MIN`, since the cap is a bound on exactly this κ.
    wilson_at_upper_bound: bool,

    // ── The reconstruction, as an embedding ──
    /// Signed sectional curvature `±1/r*²` (`0.0` flat) — the same convention
    /// as `TrialRecord::curvature`.
    curvature: f64,
    /// Eigenvalue mass the reconstruction discarded. Equal to
    /// `wilson_residual` by construction: the residual *is* this
    /// reconstruction's error. Serialised anyway, as the invariant is worth
    /// being able to check from the data.
    discarded_mass: f64,
    /// Largest geodesic distance from the manifold origin.
    r_max: f64,
    /// RMS geodesic distance from the manifold origin.
    r_rms: f64,
    /// `|curvature| · R_rms²` — the dimensionless curvature, on the same
    /// footing as `TrialRecord::kappa()` and as `--mode detect`'s, so it may
    /// share a column with a front's.
    kappa: f64,
    /// All sixteen metrics plus the radial diagnostics, from the same function
    /// that scores a t-SNE trial.
    metrics: AllMetrics,
}

/// Fit one arm and reconstruct it. The euclidean arm has no radius to fit, so
/// its `WilsonFit::radius` is infinite and it is reconstructed by plain
/// classical MDS.
fn arm(
    geometry: &'static str,
    distances: &[f64],
    n: usize,
) -> (&'static str, WilsonFit, Reconstruction) {
    match geometry {
        "spherical" => {
            let fit = fit_spherical(distances, n, EMBED_DIM);
            let rec = reconstruct_spherical(distances, n, EMBED_DIM, fit.radius);
            (geometry, fit, rec)
        }
        "hyperbolic" => {
            let fit = fit_hyperbolic(distances, n, EMBED_DIM);
            let rec = reconstruct_hyperbolic(distances, n, EMBED_DIM, fit.radius);
            (geometry, fit, rec)
        }
        _ => {
            let fit = fit_euclidean(distances, n, EMBED_DIM);
            let rec = reconstruct_euclidean(distances, n, EMBED_DIM);
            (geometry, fit, rec)
        }
    }
}

pub fn run_wilson_mds(dataset_name: &str, args: &Args, evaluator: &Evaluator) {
    let n = evaluator.n_points();
    let distances = evaluator.distances();
    let labels = evaluator.labels();

    let mut records = Vec::new();
    for geometry in ["spherical", "euclidean", "hyperbolic"] {
        let (geometry, fit, rec) = arm(geometry, distances, n);

        let metrics = metrics_from_embedding(
            distances,
            labels,
            &rec.points,
            n,
            rec.ambient_dim,
            rec.curvature,
        );

        let record = WilsonMdsRecord {
            dataset: dataset_name.to_string(),
            n_samples: n,
            geometry,
            embed_dim: EMBED_DIM,

            wilson_radius: fit.radius.is_finite().then_some(fit.radius),
            wilson_residual: fit.residual,
            wilson_residual_normalised: fit.residual_normalised,
            wilson_at_upper_bound: fit.at_upper_bound,
            curvature: rec.curvature,
            discarded_mass: rec.discarded_mass,
            r_max: metrics.r_max,
            r_rms: metrics.r_rms,
            kappa: rec.curvature.abs() * metrics.r_rms * metrics.r_rms,
            metrics,
        };

        println!(
            "wilson-mds '{}' (n={}) {:>10}: ρ={:.4e}{} κ={:.3} | trust={:.3} cont={:.3} stress={:.3}",
            dataset_name,
            n,
            record.geometry,
            record.wilson_residual_normalised,
            if record.wilson_at_upper_bound {
                "(pinned)"
            } else {
                ""
            },
            record.kappa,
            record.metrics.trustworthiness,
            record.metrics.continuity,
            record.metrics.normalized_stress,
        );
        records.push(record);
    }

    let mut file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("wilson-mds: failed to open {} for append: {e}", args.output);
            return;
        }
    };
    for record in &records {
        match serde_json::to_string(record) {
            Ok(json) => {
                writeln!(file, "{}", json).ok();
            }
            Err(e) => eprintln!("wilson-mds: failed to serialise record: {e}"),
        }
    }
}
