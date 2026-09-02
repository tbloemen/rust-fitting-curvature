//! Coordinates from a Wilson fit: the embedding the signature residual is
//! implicitly testing for.
//!
//! [`super::signature`] fits a radius by asking whether `Z(r)` has the
//! spectrum a genuine constant-curvature configuration would have, and scores
//! the misfit as `Σ|λ|` over the eigenvalues *outside* that signature block.
//! What it never does is take the step the criterion presupposes: if the
//! retained block is the configuration's Gram matrix, then its eigenvectors
//! are the configuration's coordinates.
//!
//! That step is what this module takes. Each `Z` is exactly the Gram matrix of
//! the model it belongs to:
//!
//! | Arm | `Z` | Is the Gram matrix of |
//! |---|---|---|
//! | Spherical | `r² cos(d/r)` | ambient coordinates on `Sᵈ(r) ⊂ Rᵈ⁺¹`, under the Euclidean inner product |
//! | Hyperbolic | `−r² cosh(d/r)` | points on `Hᵈ(r) ⊂ R^{d,1}`, under the *Lorentzian* inner product |
//! | Euclidean | `−J D∘D J/2` | the centred configuration, under the Euclidean inner product |
//!
//! so truncating the eigendecomposition to the retained block and scaling by
//! `√|λ|` recovers coordinates, and projecting those back onto the manifold
//! absorbs the truncation error. This is classical MDS in the flat case and
//! its spherical and Lorentzian analogues in the curved ones.
//!
//! # Why the discarded mass is the residual
//!
//! [`Reconstruction::discarded_mass`] sums `|λ|` over exactly the eigenvalues
//! the reconstruction throws away — which is the same slice
//! `signature::residual_from_spectrum` sums, with the same `retain_low` /
//! `retain_high` split. So `discarded_mass == WilsonFit::residual` for a fit at
//! the same radius, and the residual acquires a concrete reading: it is the
//! error of *this* reconstruction, not merely a misfit score. A dataset whose
//! residual is near zero reconstructs near-exactly; a dataset whose residual is
//! large is being forced into a model that cannot hold it, and the coordinates
//! say how badly.
//!
//! # Cost
//!
//! [`signature::eigen_symmetric`] is roughly 3–4× the eigenvalue-only path, so
//! reconstruction runs *once* at an already-fitted radius, never inside the
//! radius search.

use super::signature::{build_z_euclidean, build_z_hyperbolic, build_z_spherical, eigen_symmetric};
use crate::manifolds::create_manifold;

/// A configuration recovered from a fitted constant-curvature model.
///
/// The layout matches what `EmbeddingState` produces for a `dim`-dimensional
/// embedding, so these coordinates can be fed to the same metric and
/// projection code a t-SNE run's output goes through.
#[derive(Debug, Clone)]
pub struct Reconstruction {
    /// Ambient coordinates, row-major `n × ambient_dim`.
    pub points: Vec<f64>,
    /// `dim + 1` for the curved models (the ambient space a `dim`-manifold of
    /// constant curvature sits in), `dim` for the flat one — the same
    /// convention as [`crate::manifolds::Manifold::ambient_dim`].
    pub ambient_dim: usize,
    /// Sectional curvature: `+1/r²` spherical, `−1/r²` hyperbolic, `0` flat.
    /// Sign and scale match what `TrainingConfig::curvature` carries, so it can
    /// be handed straight to [`create_manifold`].
    pub curvature: f64,
    /// `Σ|λ|` over the non-signature eigenvalues — the mass this
    /// reconstruction discarded, equal to the Wilson residual at this radius.
    pub discarded_mass: f64,
}

impl Reconstruction {
    /// Geodesic distances on the manifold this configuration lives on.
    pub fn pairwise_distances(&self, n: usize) -> Vec<f64> {
        create_manifold(self.curvature).pairwise_distances(&self.points, n, self.ambient_dim)
    }

    /// Geodesic distance from each point to the manifold's natural origin —
    /// the quantity `r_max` and `r_rms` are computed from.
    pub fn distances_from_origin(&self, n: usize) -> Vec<f64> {
        create_manifold(self.curvature).distances_from_origin(&self.points, n, self.ambient_dim)
    }

    /// RMS geodesic radius — the `R_rms` of the thesis's embedding gauge
    /// (`4methods.typ` §gauge-fixing).
    ///
    /// Defined here rather than at each call site so that every κ built on it
    /// is the same quantity a t-SNE trial reports: this uses the same
    /// `Manifold::distances_from_origin` an embedding's `r_rms` comes from, and
    /// reduces it the same way.
    pub fn r_rms(&self, n: usize) -> f64 {
        let origin = self.distances_from_origin(n);
        if origin.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = origin.iter().map(|d| d * d).sum();
        (sum_sq / origin.len() as f64).sqrt()
    }

    /// `κ = |K| · R_rms²` — the dimensionless curvature of this configuration
    /// (thesis `@eq:kappa`), the same quantity `TrialRecord::kappa()` computes
    /// for a t-SNE trial. `0` for the flat model, where `K = 0` exactly.
    ///
    /// This is *the* κ of the repository: fits, `--mode detect`, the three-arm
    /// table and the embeddings all report it, so any two of them can be set
    /// side by side and differenced.
    pub fn kappa(&self, n: usize) -> f64 {
        let r = self.r_rms(n);
        self.curvature.abs() * r * r
    }
}

/// Scale an eigenvector column by `√λ` to make it a coordinate axis.
///
/// Truncation can leave a retained eigenvalue slightly the wrong sign — a
/// spherical `Z` that is not quite PSD, say. Clamping at zero collapses that
/// axis rather than producing a NaN; the mass involved is negligible by
/// construction (it was retained because it was extremal) and the projection
/// step puts the point back on the manifold regardless.
fn axis_scale(lambda: f64) -> f64 {
    lambda.max(0.0).sqrt()
}

/// Recover a spherical configuration from `Z(r) = r² cos(d/r)`.
///
/// Points on `Sᵈ(r)` span `R^{d+1}` and `Z` is their Euclidean Gram matrix, so
/// a conforming `Z` is PSD of rank `d+1`: retain the `d+1` most-positive
/// eigenvalues (matching `signature::spherical_residual`'s `(0, dim+1)` split)
/// and radially project each point back onto the sphere of radius `r`.
pub fn reconstruct_spherical(distances: &[f64], n: usize, dim: usize, r: f64) -> Reconstruction {
    let ambient = dim + 1;
    let z = build_z_spherical(distances, n, r);
    let eigen = eigen_symmetric(&z, n);

    let keep = ambient.min(n);
    let discarded_mass = eigen.values[..n - keep].iter().map(|l| l.abs()).sum();

    // Retained columns are the `keep` largest; take them in descending order
    // of eigenvalue so axis 0 carries the most variance.
    let mut points = vec![0.0; n * ambient];
    for (axis, col) in (n - keep..n).rev().enumerate() {
        let scale = axis_scale(eigen.values[col]);
        for i in 0..n {
            points[i * ambient + axis] = scale * eigen.vectors[i * n + col];
        }
    }

    // Radial projection onto the sphere. A point at the origin has no radial
    // direction to project along; send it to the pole, which is where
    // `distances_from_origin` measures from anyway.
    for i in 0..n {
        let row = &mut points[i * ambient..(i + 1) * ambient];
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-12 {
            let f = r / norm;
            for v in row.iter_mut() {
                *v *= f;
            }
        } else {
            row[0] = -r;
        }
    }

    let curvature = 1.0 / (r * r);
    center_on_manifold(&mut points, n, ambient, curvature);
    Reconstruction {
        points,
        ambient_dim: ambient,
        curvature,
        discarded_mass,
    }
}

/// Recover a hyperbolic configuration from `Z(r) = −r² cosh(d/r)`.
///
/// That kernel *is* the Lorentzian inner product `⟨xᵢ, xⱼ⟩_L` for points on the
/// hyperboloid `Hᵈ(r) ⊂ R^{d,1}` — compare `Hyperboloid::pairwise_distances`,
/// which inverts `d = r·acosh(−⟨x,y⟩_L / r²)`. So a conforming `Z` has
/// signature `(1 negative, d positive)`: the most-negative eigenvalue supplies
/// the timelike axis and the `d` most-positive the spatial ones, matching
/// `signature::hyperbolic_residual`'s `(1, dim)` split.
pub fn reconstruct_hyperbolic(distances: &[f64], n: usize, dim: usize, r: f64) -> Reconstruction {
    let ambient = dim + 1;
    let z = build_z_hyperbolic(distances, n, r);
    let eigen = eigen_symmetric(&z, n);

    let spatial = dim.min(n.saturating_sub(1));
    let discarded_mass = if n > 1 + spatial {
        eigen.values[1..n - spatial].iter().map(|l| l.abs()).sum()
    } else {
        0.0
    };

    let mut points = vec![0.0; n * ambient];

    // Axis 0 is timelike: `⟨x,x⟩_L` weights it by −1, so it is carried by the
    // most-negative eigenvalue and scaled by √(−λ₀).
    let time_scale = axis_scale(-eigen.values[0]);
    for i in 0..n {
        points[i * ambient] = time_scale * eigen.vectors[i * n];
    }
    for (axis, col) in (n - spatial..n).rev().enumerate() {
        let scale = axis_scale(eigen.values[col]);
        for i in 0..n {
            points[i * ambient + 1 + axis] = scale * eigen.vectors[i * n + col];
        }
    }

    // The hyperboloid has two sheets and the model lives on the upper one
    // (`x₀ > 0`), but an eigenvector's overall sign is arbitrary. The timelike
    // component of a `cosh` Gram matrix is single-signed, so one global flip
    // settles it; decide by majority in case truncation perturbed a few.
    let positive = (0..n).filter(|&i| points[i * ambient] > 0.0).count();
    if positive * 2 < n {
        for i in 0..n {
            points[i * ambient] = -points[i * ambient];
        }
    }

    project_to_hyperboloid(&mut points, n, ambient, r);

    let curvature = -1.0 / (r * r);
    center_on_manifold(&mut points, n, ambient, curvature);
    Reconstruction {
        points,
        ambient_dim: ambient,
        curvature,
        discarded_mass,
    }
}

/// Recover a flat configuration from `B = −J D∘D J / 2` — classical MDS.
///
/// `build_z_euclidean` has already double-centred, which is what spends the
/// `1` direction the curved models keep an extra ambient slot for; so a
/// conforming `B` is PSD of rank `dim`, matching
/// `signature::euclidean_residual`'s `(0, dim)` split. There is no manifold to
/// project back onto and no radius to fit.
pub fn reconstruct_euclidean(distances: &[f64], n: usize, dim: usize) -> Reconstruction {
    let b = build_z_euclidean(distances, n);
    let eigen = eigen_symmetric(&b, n);

    let keep = dim.min(n);
    let discarded_mass = eigen.values[..n - keep].iter().map(|l| l.abs()).sum();

    let mut points = vec![0.0; n * dim];
    for (axis, col) in (n - keep..n).rev().enumerate() {
        let scale = axis_scale(eigen.values[col]);
        for i in 0..n {
            points[i * dim + axis] = scale * eigen.vectors[i * n + col];
        }
    }

    Reconstruction {
        points,
        ambient_dim: dim,
        curvature: 0.0,
        discarded_mass,
    }
}

/// Put every point exactly on the hyperboloid `⟨x,x⟩_L = −r²`, `x₀ > 0`.
///
/// Truncation leaves points slightly off the sheet. A point already inside the
/// light cone is rescaled radially, which preserves its direction; one that
/// truncation pushed outside has no radial fix, so its spatial part is kept and
/// the timelike component recomputed from the constraint.
fn project_to_hyperboloid(points: &mut [f64], n: usize, ambient: usize, r: f64) {
    let r2 = r * r;
    for i in 0..n {
        let row = &mut points[i * ambient..(i + 1) * ambient];
        let spatial_sq: f64 = row[1..].iter().map(|v| v * v).sum();
        let q = spatial_sq - row[0] * row[0];
        if q < -1e-12 {
            let f = r / (-q).sqrt();
            for v in row.iter_mut() {
                *v *= f;
            }
        } else {
            row[0] = (r2 + spatial_sq).sqrt();
        }
    }
}

/// Move the configuration's Fréchet mean to the manifold's natural origin.
///
/// The eigen-basis is arbitrary up to an isometry of the model, so an
/// uncentred reconstruction has a meaningless `distances_from_origin` and
/// therefore a meaningless `r_rms` and `κ`. Centring fixes the frame the same
/// way `Manifold::center` does for an embedding, which is what makes the two
/// κ values comparable. Pairwise distances are unaffected — it is an isometry.
fn center_on_manifold(points: &mut [f64], n: usize, ambient: usize, curvature: f64) {
    create_manifold(curvature).center(points, n, ambient);
}
