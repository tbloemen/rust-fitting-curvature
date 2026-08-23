//! Curvature detection from a distance matrix.
//!
//! Combines two ingredients:
//!
//! - Wilson, Hancock, Pekalska & Duin (2014), *Spherical and Hyperbolic
//!   Embeddings of Data*, IEEE TPAMI 36(11): 2255–2268 — radius-of-
//!   curvature estimation by fitting a constant-curvature Gram matrix
//!   `Z(r)` and minimising the magnitude of a signature eigenvalue.
//! - Gromov 4-point δ — a coarse-geometric, embedding-free test for
//!   negatively-curved metric structure, used to gate the hyperbolic
//!   decision (see `GROMOV_THRESHOLD`).  Wilson's eigenvalue residual
//!   alone cannot reliably distinguish hyperbolic data from Euclidean
//!   data: at `r ≫ d_max` the matrix `−r² cosh(d/r) ≈ −r² − d²/2`
//!   collapses to a rank-1-plus-`d²` form whose residual resembles a
//!   Euclidean Gram residual.
//!
//! # Radius fitting (residual-eigenvalue criterion)
//!
//! Given a pairwise distance matrix `D` and a target embedding
//! dimension `d`, fit each constant-curvature model by minimising the
//! magnitude of the eigenvalues of the curvature-dependent Gram matrix
//! `Z(r)` that fall *outside* the eigen-subspace a genuine
//! `d`-dimensional constant-curvature configuration is allowed to
//! occupy.
//!
//! A set of points on `Sᵈ`/`Hᵈ` lives in a `(d+1)`-dimensional ambient
//! space, so `Z(r) = X G Xᵀ` has rank `d+1` with a fixed signature
//! (Wilson et al. 2014, Sections V.A/V.C):
//!
//! - **Spherical** `Z_{ij}(r) = r² cos(d_{ij}/r)`, Euclidean ambient
//!   `R^{d+1}` ⇒ signature `(d+1 positive, rest zero)`.  Retain the
//!   `d+1` largest (most positive) eigenvalues.
//! - **Hyperbolic** `Z_{ij}(r) = −r² cosh(d_{ij}/r)`, Lorentzian ambient
//!   `R^{d,1}` ⇒ signature `(1 negative, d positive, rest zero)`.
//!   Retain the most-negative eigenvalue plus the `d` largest positive
//!   ones.
//!
//! Everything outside that retained block *should* be zero, which is
//! exactly the paper's fitting criterion: with the spectrum ordered
//! ascending `λ₁ ≤ … ≤ λₙ` and an ambient embedding dimension `m = d+1`,
//!
//! ```text
//!     r* = arg min_r Σ_{i ≤ n−m} |λᵢ[Z(r)]|
//! ```
//!
//! (Wilson et al. 2014, Section V.B — "if we wish to find an embedding
//! space of dimension `m`, then we can try to minimise the remaining
//! `n − m` eigenvalues").  The `m = n−1` case in the paper's derivation
//! is the single leftover eigenvalue `|λ₁|`; general `m` sums the `n−m`
//! smallest.
//!
//! For the hyperbolic model one eigenvalue is reserved for the negative
//! (time) axis of the Lorentzian ambient space, so the sum runs over
//! `λ₂ … λ_{n−d}` — `λ₁` (the most negative) is retained as the time
//! direction, the `d` largest are the spatial directions, and everything
//! between them is residual.  The paper only states the hyperbolic
//! criterion for `m = n−1` (minimise `|λ₂|`); the `n−m` generalisation
//! here is the natural reading of it.
//!
//! The objective is the **raw** sum of absolute residual eigenvalues, in
//! the units of `Z` (i.e. squared distance).  It is not normalised by
//! anything that depends on `r`: the earlier normalised variant
//! (`1 − Σ_retained λᵢ² / ‖Z‖_F²`) divided by a denominator that grows
//! like `n²r⁴`, which drove the residual to 0 for *any* geometry at large
//! `r`.  The raw sum has no such attractor — at large `r` both kernels
//! tend to `±r²J ∓ D²/2`, whose leftover eigenvalues stay `O(d_max²)`
//! unless the data really is flat.
//!
//! For *reporting*, and for the threshold in [`detect_geometry`], the raw
//! sum is divided by the dataset constants `n · d_max²`
//! ([`WilsonFit::residual_normalised`]).  `d_max²` carries the units of
//! `Z`; `n` removes the extensivity of a kernel spectrum (`λᵢ(Z)/n`
//! converges to the eigenvalues of the corresponding integral operator,
//! and exactly `tr Z = ±n r²`).  Neither factor depends on `r`, so this is
//! a fixed rescale: it leaves `r*` untouched while making residuals
//! comparable across datasets of different diameter *and* sample size.
//!
//! An earlier version also carried the nuclear-norm fraction
//! `Σ_res|λ| / Σ_all|λ|` and thresholded *that*.  It was removed because
//! its denominator is a function of `r`.  `tr Z = ±n r²` exactly, and
//! `‖Z‖_* = |tr Z|` iff `Z` is PSD — which is precisely the spherical
//! model — so at a good spherical fit `Σ_all|λ| = n r*²` and the fraction
//! is exactly `Σ|λ_res| / (n r*²)`: gauged by the *fitted radius* rather
//! than by a property of the data.  A larger `r*` deflates the score,
//! flattering exactly the flat-ward fits the gate exists to reject.  The
//! two statistics differ by `(d_max/r*)²`, which stays bounded only
//! because the spherical search window is narrow — see
//! [`SPHERICAL_RESIDUAL_MAX`].
//!
//! # Eigenvalue extraction
//!
//! The criterion needs the *whole* spectrum (every residual eigenvalue,
//! not just a few extremal ones), so partial power iteration is not
//! enough.  [`eigenvalues_symmetric`] does a full symmetric
//! eigenvalue-only decomposition: Householder reduction to tridiagonal
//! form followed by the implicit-shift QL iteration.  Both are the
//! standard EISPACK/Numerical-Recipes formulations, written out here
//! because `fitting-core` carries no dependencies.  Cost is `O(n³)`
//! (dominated by the `(2/3)n³` tridiagonalisation) per candidate radius.

use std::f64::consts::PI;

use super::gromov_ball_curve::detect_hyperbolic;

/// Max implicit-QL sweeps per eigenvalue before giving up on that one.
/// The classic bound is 30; convergence is cubic, so hitting this means
/// the matrix is pathological rather than merely slow.
const QL_MAX_ITER: usize = 50;

// ── Symmetric eigenvalue decomposition ─────────────────────────────────────

/// All eigenvalues of the symmetric matrix `a` (row-major, `n × n`),
/// returned in **ascending** order.
///
/// Householder reduction to tridiagonal form + implicit-shift QL.  Only
/// the lower triangle of `a` is read; eigenvectors are not accumulated.
pub fn eigenvalues_symmetric(a: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let mut z = a.to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    tridiagonalise(&mut z, n, &mut d, &mut e);
    ql_implicit(&mut d, &mut e, n);
    d.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    d
}

/// Householder reduction of a symmetric matrix to tridiagonal form
/// (Numerical Recipes `tred2`, eigenvector accumulation dropped).
///
/// On return `d` holds the diagonal and `e[1..]` the sub-diagonal of the
/// tridiagonal matrix; `z` is destroyed.
fn tridiagonalise(z: &mut [f64], n: usize, d: &mut [f64], e: &mut [f64]) {
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let mut scale = 0.0;
            for k in 0..i {
                scale += z[i * n + k].abs();
            }
            if scale == 0.0 {
                e[i] = z[i * n + l];
            } else {
                for k in 0..i {
                    z[i * n + k] /= scale;
                    h += z[i * n + k] * z[i * n + k];
                }
                let mut f = z[i * n + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                z[i * n + l] = f - g;
                f = 0.0;
                for j in 0..i {
                    let mut g = 0.0;
                    for k in 0..=j {
                        g += z[j * n + k] * z[i * n + k];
                    }
                    for k in (j + 1)..i {
                        g += z[k * n + j] * z[i * n + k];
                    }
                    e[j] = g / h;
                    f += e[j] * z[i * n + j];
                }
                let hh = f / (h + h);
                for j in 0..i {
                    let f = z[i * n + j];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        z[j * n + k] -= f * e[k] + g * z[i * n + k];
                    }
                }
            }
        } else {
            e[i] = z[i * n + l];
        }
        d[i] = h;
    }
    e[0] = 0.0;
    for i in 0..n {
        d[i] = z[i * n + i];
    }
}

/// Eigenvalues of a symmetric tridiagonal matrix by the QL algorithm
/// with implicit shifts (Numerical Recipes `tqli`, eigenvector
/// accumulation dropped).  `d` is the diagonal (overwritten with the
/// eigenvalues, unordered), `e[1..]` the sub-diagonal (destroyed).
fn ql_implicit(d: &mut [f64], e: &mut [f64], n: usize) {
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut iter = 0;
        loop {
            // Look for a single small sub-diagonal element to split the
            // matrix; `m == l` means d[l] has converged.
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > QL_MAX_ITER {
                break;
            }

            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.abs().copysign(g));
            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            // Plane rotations chased down the sub-diagonal from m−1 to l.
            let mut underflowed = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    // Recover from underflow: drop the element and restart.
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflowed = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if underflowed {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

// ── Gram matrices ───────────────────────────────────────────────────────────

/// `Z_{ij}(r) = r² cos(d_{ij}/r)` for spherical model.
fn build_z_spherical(d: &[f64], n: usize, r: f64) -> Vec<f64> {
    let r2 = r * r;
    let inv_r = 1.0 / r;
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            z[i * n + j] = r2 * (d[i * n + j] * inv_r).cos();
        }
    }
    z
}

/// `Z_{ij}(r) = −r² cosh(d_{ij}/r)` for hyperbolic model.
fn build_z_hyperbolic(d: &[f64], n: usize, r: f64) -> Vec<f64> {
    let r2 = r * r;
    let inv_r = 1.0 / r;
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            z[i * n + j] = -r2 * (d[i * n + j] * inv_r).cosh();
        }
    }
    z
}

// ── Signature residuals ──────────────────────────────────────────────────────

/// `Σ |λ|` over the residual eigenvalues, given the full ascending
/// spectrum and which entries the signature retains.  `retain_low` is 1
/// for the Lorentzian time axis (hyperbolic) and 0 for a PSD model
/// (spherical); `retain_high` is the number of largest eigenvalues the
/// model keeps.
///
/// This is the quantity Wilson et al. minimise over `r`.  It carries the
/// units of `Z` (squared distance) and is extensive in `n`, so it is
/// comparable across radii on one dataset but not across datasets — for
/// that, divide by `n · d_max²` as [`WilsonFit::residual_normalised`] does.
fn residual_from_spectrum(lambda: &[f64], retain_low: usize, retain_high: usize) -> f64 {
    let n = lambda.len();
    if n <= retain_low + retain_high {
        return 0.0;
    }
    lambda[retain_low..n - retain_high]
        .iter()
        .map(|l| l.abs())
        .sum()
}

/// Spherical residual for a `dim`-dimensional embedding.  A `dim`-sphere
/// lives in Euclidean `R^{dim+1}`, so a conforming `Z_spherical` is PSD
/// of rank `dim+1`: retain the `dim+1` most-positive eigenvalues and sum
/// `|λ|` over the remaining `n − (dim+1)` smallest.  Negative eigenvalues
/// violate the PSD model and are exactly the ones this sum penalises.
fn spherical_residual(d: &[f64], n: usize, dim: usize, r: f64) -> f64 {
    let z = build_z_spherical(d, n, r);
    residual_from_spectrum(&eigenvalues_symmetric(&z, n), 0, dim + 1)
}

/// Hyperbolic residual for a `dim`-dimensional embedding.  A
/// `dim`-dimensional hyperbolic configuration lives in Lorentzian
/// `R^{dim,1}`, so a conforming `Z_hyperbolic` has signature
/// `(1 negative, dim positive)`: retain `λ₁` (the most negative — the
/// time axis) plus the `dim` most-positive, and sum `|λ|` over the
/// `n − (dim+1)` eigenvalues sandwiched between them.
fn hyperbolic_residual(d: &[f64], n: usize, dim: usize, r: f64) -> f64 {
    let z = build_z_hyperbolic(d, n, r);
    residual_from_spectrum(&eigenvalues_symmetric(&z, n), 1, dim)
}

// ── Search over r ───────────────────────────────────────────────────────────

/// Result of fitting one constant-curvature model.
#[derive(Debug, Clone, Copy)]
pub struct WilsonFit {
    /// Best-fit radius of curvature.  Sectional curvature is +1/r²
    /// (spherical) or −1/r² (hyperbolic).
    pub radius: f64,
    /// `Σ |λᵢ|` over the residual (non-signature) eigenvalues of
    /// `Z(radius)` — the objective minimised to find `radius`.  Small ⇒
    /// data conforms well to the model.  Carries the units of `Z`
    /// (squared distance) and is extensive in `n`; use
    /// [`WilsonFit::residual_normalised`] to compare across datasets.
    pub residual: f64,
    /// `residual / (n · d_max²)` — the same misfit divided by two
    /// constants of the dataset.  `d_max²` cancels the units of `Z` and
    /// `n` cancels the extensivity of its spectrum, so this is comparable
    /// across datasets of different diameter and sample size, and it is
    /// what [`detect_geometry`] thresholds against
    /// [`SPHERICAL_RESIDUAL_MAX`].  Neither factor depends on `r`, so
    /// dividing by them cannot move `radius`.  `0` iff the data lies
    /// exactly on the model manifold.
    pub residual_normalised: f64,
    /// True if `radius` is pinned at the upper bound of the search range.
    /// For the spherical fit this is the flat-ward edge of the coverage
    /// window (`d_max/`[`SPHERICAL_ANGULAR_MIN`]): being pinned there
    /// means the residual is still falling toward flatter radii, i.e. the
    /// data prefers a flatter model than a well-covered sphere — the
    /// signal [`detect_geometry`] uses to reject a spherical verdict.
    pub at_upper_bound: bool,
}

/// Divide a raw residual by the dataset constants `n · d_max²`, giving
/// [`WilsonFit::residual_normalised`].  Degenerate input (`n = 0`, or all
/// points coincident so `d_max = 0`) has no meaningful scale to divide by,
/// and reporting an infinite misfit makes every threshold comparison reject
/// — the safe direction.
fn normalise_residual(residual: f64, n: usize, d_max: f64) -> f64 {
    let scale = n as f64 * d_max * d_max;
    if scale > 0.0 {
        residual / scale
    } else {
        f64::INFINITY
    }
}

/// Golden-section minimisation on `[a, b]`.  Returns `(r*, f(r*))`.
fn golden_section(a: f64, b: f64, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
    let phi = 0.6180339887498949_f64;
    let mut a = a;
    let mut b = b;
    let mut r1 = a + (1.0 - phi) * (b - a);
    let mut r2 = a + phi * (b - a);
    let mut f1 = f(r1);
    let mut f2 = f(r2);
    for _ in 0..40 {
        if f1 < f2 {
            b = r2;
            r2 = r1;
            f2 = f1;
            r1 = a + (1.0 - phi) * (b - a);
            f1 = f(r1);
        } else {
            a = r1;
            r1 = r2;
            f1 = f2;
            r2 = a + phi * (b - a);
            f2 = f(r2);
        }
        if (b - a) / (a + b).max(1e-12) < 1e-5 {
            break;
        }
    }
    let r_star = 0.5 * (a + b);
    let f_star = f(r_star);
    if f_star < f1 && f_star < f2 {
        (r_star, f_star)
    } else if f1 < f2 {
        (r1, f1)
    } else {
        (r2, f2)
    }
}

/// One-dimensional minimisation of `f` on a log-spaced grid over
/// `[lo, hi]`.  Finds *every* local minimum on the grid and refines
/// each by golden section, returning the best one.  This is needed
/// because the Wilson residual function can be multi-modal: a sharp
/// minimum at the true radius (small `r`) coexists with a shallow
/// minimum at the Euclidean-limit upper bound.  A single-best refine
/// misses the sharp minimum whenever the coarse grid does.
///
/// Returns `(r*, f(r*), at_upper_bound)`.
fn minimise_log_spaced(
    lo: f64,
    hi: f64,
    n_grid: usize,
    f: &mut dyn FnMut(f64) -> f64,
) -> (f64, f64, bool) {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    let step = (log_hi - log_lo) / (n_grid - 1) as f64;
    let grid_r: Vec<f64> = (0..n_grid)
        .map(|i| (log_lo + i as f64 * step).exp())
        .collect();
    let grid_res: Vec<f64> = grid_r.iter().map(|&r| f(r)).collect();

    // Identify local minima on the coarse grid.  A point is a local min
    // if it's strictly less than its existing neighbours.  The endpoints
    // count as local minima if they beat their single neighbour.
    let mut local_min_indices: Vec<usize> = Vec::new();
    if n_grid >= 2 && grid_res[0] < grid_res[1] {
        local_min_indices.push(0);
    }
    for i in 1..n_grid.saturating_sub(1) {
        if grid_res[i] < grid_res[i - 1] && grid_res[i] < grid_res[i + 1] {
            local_min_indices.push(i);
        }
    }
    if n_grid >= 2 && grid_res[n_grid - 1] < grid_res[n_grid - 2] {
        local_min_indices.push(n_grid - 1);
    }
    if local_min_indices.is_empty() {
        // Monotone on the grid: take the global minimum index.
        local_min_indices.push(
            grid_res
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
        );
    }

    let mut best_r = grid_r[local_min_indices[0]];
    let mut best_res = grid_res[local_min_indices[0]];
    let mut best_i = local_min_indices[0];
    for &i in &local_min_indices {
        let a = grid_r[i.saturating_sub(1)];
        let b = grid_r[(i + 1).min(n_grid - 1)];
        let (r, res) = if a < b {
            golden_section(a, b, f)
        } else {
            (grid_r[i], grid_res[i])
        };
        if res < best_res {
            best_res = res;
            best_r = r;
            best_i = i;
        }
    }

    let at_upper = best_i == n_grid - 1;
    (best_r, best_res, at_upper)
}

/// Fit a `dim`-dimensional spherical model: find `r*` minimising
/// `Σ|λ|` over the eigenvalues of `Z_spherical(r)` outside its
/// rank-`(dim+1)` PSD signature block.
///
/// Search bounds — the **angular-coverage window**:
///
/// - Lower `r ≥ d_max/π` is a *hard feasibility* bound: the largest
///   geodesic distance on a radius-`r` sphere is `πr`, so `d_max ≤ πr`
///   is required (below it the farthest pair would be beyond antipodal
///   and `Z` is degenerate).  Finite sampling puts the true radius
///   *above* this floor (no exactly-antipodal pair ⇒ `d_max < πR`), so
///   it never clips a genuine minimum.
/// - Upper `r ≤ d_max/`[`SPHERICAL_ANGULAR_MIN`] is the flat-ward edge of
///   the coverage window: a spherical verdict requires
///   `d_max/r ≥ SPHERICAL_ANGULAR_MIN`, so radii above this can never be
///   labelled spherical anyway.
///
/// The window is narrow — `[d_max/π, d_max/2.5]` ≈ `[0.318, 0.400]·d_max`
/// — which sharpens the radius estimate.  A fit pinned at the upper edge
/// ([`WilsonFit::at_upper_bound`]) means the data prefers a flatter
/// sphere than the coverage threshold allows ⇒ not spherical; that flag,
/// not a post-hoc angular comparison, is what [`detect_geometry`] reads.
pub fn fit_spherical(distances: &[f64], n: usize, dim: usize) -> WilsonFit {
    let d_max = distances.iter().cloned().fold(0.0_f64, f64::max);
    let r_lower = d_max / PI;
    let r_upper = d_max / SPHERICAL_ANGULAR_MIN;

    let mut objective = |r: f64| -> f64 { spherical_residual(distances, n, dim, r) };

    let (r_star, _, at_upper) = minimise_log_spaced(r_lower, r_upper, 30, &mut objective);
    let residual = spherical_residual(distances, n, dim, r_star);
    WilsonFit {
        radius: r_star,
        residual,
        residual_normalised: normalise_residual(residual, n, d_max),
        at_upper_bound: at_upper,
    }
}

/// Smallest dimensionless curvature `κ = |K|·d_rms²` the hyperbolic search
/// is asked to resolve; it sets the flat-ward edge of the window via
/// `r_upper = d_rms/√κ_min` (see [`fit_hyperbolic`]).
///
/// A hyperbolic metric departs from the Euclidean one by a *relative*
/// `κ/6` over the extent of the data — the circumference of a geodesic
/// circle of radius `s` in `H²(r)` is
/// `2πr·sinh(s/r) = 2πs(1 + (s/r)²/6 + …)`, and `(d_rms/r)² = κ`.  At
/// `κ = 0.01` that is 0.17%: beyond this radius the model is flat to
/// within a fifth of a percent across the whole sample, so searching
/// further only fits Euclidean structure.
pub const HYPERBOLIC_KAPPA_MIN: f64 = 0.01;

/// Root-mean-square of the `n(n−1)` off-diagonal pairwise distances — the
/// scale statistic the thesis's `κ = |K|·d_rms²` gauge is written in.
fn rms_distance(distances: &[f64], n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let sum_sq: f64 = distances.iter().map(|d| d * d).sum();
    (sum_sq / (n as f64 * (n as f64 - 1.0))).sqrt()
}

/// Fit a `dim`-dimensional hyperbolic model: find `r*` minimising
/// `Σ|λ|` over the eigenvalues of `Z_hyperbolic(r)` outside its
/// `(1 negative, dim positive)` signature block.
///
/// Search bounds: r ≥ d_max/20 (keeps `cosh(d_max/r) ≤ cosh(20) ≈ 2.4·10⁸`,
/// safe from overflow) and `r ≤ d_rms/√`[`HYPERBOLIC_KAPPA_MIN`], i.e.
/// `10·d_rms`.  Hyperbolic space is non-compact, so the
/// geodesic-fits-on-space lower bound from the spherical case does not
/// apply — but for r ≫ d_rms, `cosh(d/r) ≈ 1 + d²/2r²` and `Z` degenerates
/// to the flat-limit kernel `−r²J − D²/2`, which no longer carries
/// curvature information.  The cap is stated as a floor on the
/// dimensionless curvature rather than as a multiple of `d_max` so that it
/// makes the same demand of every dataset: `d_max` is an extreme order
/// statistic that drifts with `n` and varies 30× across the thesis
/// datasets, so `r ≤ d_max` silently imposes a `κ_min` that ranges over 7×.
pub fn fit_hyperbolic(distances: &[f64], n: usize, dim: usize) -> WilsonFit {
    let d_max = distances.iter().cloned().fold(0.0_f64, f64::max);
    let r_lower = d_max / 20.0;
    let r_upper = (rms_distance(distances, n) / HYPERBOLIC_KAPPA_MIN.sqrt()).max(r_lower);

    let mut objective = |r: f64| -> f64 { hyperbolic_residual(distances, n, dim, r) };

    let (r_star, _, at_upper) = minimise_log_spaced(r_lower, r_upper, 30, &mut objective);
    let residual = hyperbolic_residual(distances, n, dim, r_star);
    WilsonFit {
        radius: r_star,
        residual,
        residual_normalised: normalise_residual(residual, n, d_max),
        at_upper_bound: at_upper,
    }
}

// ── Detection ───────────────────────────────────────────────────────────────

/// The minimal geometry-detection result the embedding pipeline acts on:
/// *which* constant-curvature space to use and its signed sectional
/// curvature.  This is intentionally decoupled from the diagnostic
/// internals (Wilson fits, Gromov tail slope) so that consumers like the
/// optimizer depend only on the decision, not on how it was reached —
/// obtain it from [`detect_geometry`].
#[derive(Debug, Clone, Copy)]
pub struct GeometryVerdict {
    /// `"spherical"`, `"hyperbolic"`, or `"euclidean"`.
    pub best_geometry: &'static str,
    /// Signed sectional curvature estimate at the detected radius.
    /// `0.0` when the detection is `"euclidean"`.
    pub curvature: f64,
}

/// Diagnostic: raw spherical signature residual `Σ|λ_res|` at a single r
/// for a `dim`-dimensional model.  Exposed so tests can plot the residual
/// curve; divide by `n · d_max²` to compare across datasets.
pub fn spherical_residual_at(distances: &[f64], n: usize, dim: usize, r: f64) -> f64 {
    spherical_residual(distances, n, dim, r)
}

/// Diagnostic: raw hyperbolic signature residual `Σ|λ_res|` at a single r
/// for a `dim`-dimensional model.
pub fn hyperbolic_residual_at(distances: &[f64], n: usize, dim: usize, r: f64) -> f64 {
    hyperbolic_residual(distances, n, dim, r)
}

/// Minimum angular extent `d_max / r*` required for the spherical fit
/// to be treated as genuine.  A uniform sample on a sphere has
/// `d_max ≈ π · r` and `d/r` is literally the subtended angle in
/// radians, so 2.5 (≈ 0.8 π) accepts near-full coverage while rejecting
/// Euclidean data which fits a small cap on a very large sphere with
/// arbitrarily small residual.  It also sets the flat-ward edge of the
/// spherical search window in [`fit_spherical`]
/// (`r_upper = d_max / SPHERICAL_ANGULAR_MIN`), so the coverage criterion
/// is enforced *by construction* of the search range rather than checked
/// after the fact.
pub const SPHERICAL_ANGULAR_MIN: f64 = 2.5;

/// Largest [`WilsonFit::residual_normalised`] a spherical fit may have and
/// still be believed.  Wilson et al.: "a small residual indicates the
/// data is close to spherical" — this is the threshold on that.
///
/// The angular-coverage window alone is not enough under the
/// residual-eigenvalue criterion.  `Σ|λ_res|` is nearly flat across the
/// window for data that is not spherical at all (the window spans only
/// `r ∈ [d_max/π, d_max/2.5]`, a factor 1.26), so which edge the minimum
/// lands on is decided by noise rather than by geometry — H² data can
/// come to rest at the *lower* edge and pass an `at_upper_bound` test it
/// should fail.  The residual is not flat at all, so it backstops that.
///
/// # Why this is gauged by `n · d_max²` and not by `‖Z‖_*`
///
/// The threshold used to sit on the nuclear-norm fraction, whose
/// denominator moves with `r` (see the module docs).  That coupling makes
/// the constant a function of [`SPHERICAL_ANGULAR_MIN`]: widening the
/// window flat-ward admits larger `r*`, and a larger `r*` shrinks the
/// fraction quadratically.  Measured on the three synthetic fixtures plus
/// mnist / fashion_mnist / pbmc / wordnet_mammals, sweeping
/// `SPHERICAL_ANGULAR_MIN` over `{2.5, 2.0, 1.5, 1.0, 0.5}`, the Euclidean
/// fixture's fraction falls `8.0e-2 → 3.3e-2 → 1.1e-2 → 2.1e-3 → 1.3e-4`
/// as its `r*` runs flat-ward — so at 2.0 it was already *under* the old
/// 0.05 and the residual test had stopped rejecting flat data at all,
/// silently handing the whole decision back to `at_upper_bound`.  Under
/// `Σ|λ_res| / (n·d_max²)` the same fixture only falls
/// `1.4e-2 → 8.5e-3 → 4.8e-3 → 2.1e-3 → 5.4e-4`, so the threshold and the
/// window stay independent knobs.
///
/// # Choice of value
///
/// On that same sweep the genuinely spherical fixture scores `≤ 1e-5` at
/// every window width and the nearest impostor is the Euclidean fixture,
/// so any value in roughly `[1e-5, 2e-3]` gives identical verdicts on all
/// seven datasets.  `1e-3` is the loose end of that band: ~100× above the
/// true sphere's worst score, 13.5× below the nearest impostor at the
/// current `SPHERICAL_ANGULAR_MIN = 2.5`, but only 2.1× below it if the
/// window is widened to 1.0.
///
/// Being loose buys little, because the ordering forbids it: the closest
/// real dataset (pbmc, `2.2e-2`) scores *worse* than the flat synthetic
/// fixture, so no threshold admits a real dataset without also admitting
/// Euclidean data.
///
/// # Noise tolerance (`examples/spherical_noise_sweep.rs`)
///
/// The exact `S²` fixture scores `~1e-7`, so the threshold's distance from
/// it says nothing about slack on real data.  Sweeping noise gives the
/// real answer, and it is severe.  Under multiplicative jitter on the
/// geodesic distance matrix the residual is **linear in σ** with slope
/// ≈ 2: `σ = 0.025% → 4.9e-4`, `0.05% → 9.9e-4`, `0.1% → 2.0e-3`,
/// `0.5% → 1.1e-2`.  So the gate stops firing at about **0.05% distance
/// noise**, and at ~0.7% the noisy sphere scores *worse than the flat
/// fixture* — past which no threshold can separate them.
///
/// Worse, feeding **Euclidean (chordal) distances instead of geodesics**
/// fails outright: perfect sphere, zero noise, ambient distances scores
/// `4.8e-2`, already 3.3× worse than flat data, and the fit pins
/// flat-ward.  `Z = r²cos(d/r)` genuinely requires geodesic `d`.
///
/// Two consequences.  First, this is a near-exact-manifold test, not a
/// noisy-data test — the value of this constant is not what stops real
/// datasets being labelled spherical.  Second, the ceiling on any
/// loosening is the flat fixture's `1.5e-2`; raising `1e-3` to `1e-2`
/// would buy roughly 10× noise tolerance (0.05% → 0.5%) but only while
/// `SPHERICAL_ANGULAR_MIN` stays at 2.5, since the impostor floor is
/// already `8.5e-3` at 2.0.
pub const SPHERICAL_RESIDUAL_MAX: f64 = 1e-3;

/// Detect curvature from a distance matrix.
///
/// Decision rule:
///
/// 1. **Spherical** if the spherical fit both *lands well* and *lands
///    inside* its angular-coverage window:
///    - its scale-free residual [`WilsonFit::residual_normalised`] is
///      below [`SPHERICAL_RESIDUAL_MAX`] — the data really does collapse
///      onto a rank-`(dim+1)` PSD Gram matrix at `r*`; and
///    - the minimum is not pinned at the flat-ward upper edge
///      (`!at_upper_bound`).  Because the window is exactly
///      `[d_max/π, d_max/SPHERICAL_ANGULAR_MIN]`, this is a literal
///      coverage criterion (`d_max/r ≥ SPHERICAL_ANGULAR_MIN`) read off
///      the fit's position, rejecting data that prefers a small cap on a
///      very large sphere.
/// 2. **Hyperbolic** if the Gromov δ(k) growing-ball curve **saturates**
///    (its log–log tail slope is below
///    [`super::gromov_ball_curve::SATURATION_SLOPE_THRESHOLD`]).  This is the
///    theoretically-backed NeTS-proposal test: only negatively-curved
///    spaces have a δ bounded below the diameter, so only they plateau
///    as the ball grows.  It replaces the earlier global-percentile
///    Gromov δ threshold, which was a scale-dependent order statistic
///    with no link to the underlying geometry.
/// 3. **Euclidean** otherwise.
///
/// `dim` is the target embedding dimension — the spherical and
/// hyperbolic models are fitted as `dim`-dimensional manifolds (rank
/// `dim+1` Gram matrices).
pub fn detect_geometry(distances: &[f64], n: usize, dim: usize) -> GeometryVerdict {
    let spherical = fit_spherical(distances, n, dim);

    // The spherical fit searches only the angular-coverage window
    // `[d_max/π, d_max/SPHERICAL_ANGULAR_MIN]`.  A minimum in its interior
    // (or at the feasibility floor) means the data genuinely prefers a
    // well-covered sphere; a minimum pinned at the flat-ward upper edge
    // means it prefers a flatter model than the coverage threshold allows.
    // Landing in the window is necessary but not sufficient: the fit must
    // also *be good there*, or non-spherical data merely settles on
    // whichever edge of a flat objective happens to be lower.
    if spherical.residual_normalised < SPHERICAL_RESIDUAL_MAX && !spherical.at_upper_bound {
        return GeometryVerdict {
            best_geometry: "spherical",
            curvature: 1.0 / (spherical.radius * spherical.radius),
        };
    }
    let hyp = detect_hyperbolic(distances, n);
    if hyp.is_hyperbolic {
        let hyperbolic = fit_hyperbolic(distances, n, dim);
        GeometryVerdict {
            best_geometry: "hyperbolic",
            curvature: -1.0 / (hyperbolic.radius * hyperbolic.radius),
        }
    } else {
        GeometryVerdict {
            best_geometry: "euclidean",
            curvature: 0.0,
        }
    }
}
