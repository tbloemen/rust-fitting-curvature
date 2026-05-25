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
//! # Radius fitting (rank-`(d+1)` signature residual)
//!
//! Given a pairwise distance matrix `D` and a target embedding
//! dimension `d`, fit each constant-curvature model by minimising the
//! Frobenius energy of the curvature-dependent Gram matrix `Z(r)` that
//! falls *outside* the eigen-subspace a genuine `d`-dimensional
//! constant-curvature configuration is allowed to occupy.
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
//! Everything outside that retained block *should* be zero, so the fit
//! minimises the normalised residual
//!
//! ```text
//!     ρ(r) = 1 − (Σ_retained λᵢ²) / ‖Z(r)‖_F²,    r* = arg min_r ρ(r)
//! ```
//!
//! `ρ ∈ [0, 1]` is the fraction of `‖Z‖_F²` lying outside the signature
//! subspace; `ρ = 0` iff the data lies exactly on a `d`-dimensional
//! constant-curvature manifold of radius `r`.  Normalising by `‖Z‖_F²`
//! removes the `r⁴` scale of the raw energy, which would otherwise bias
//! the search toward small `r`.  Wilson's original single-eigenvalue
//! criterion (`|λ₁|` spherical, `|λ₂|` hyperbolic) is the `d = n−2`
//! special case, where exactly one eigenvalue is left over.
//!
//! # Eigenvalue extraction
//!
//! Following the paper's recommendation, we use power iteration plus
//! shifting and deflation rather than a full eigendecomposition:
//!
//! - `power_max` — largest-*magnitude* eigenvalue via plain power
//!   iteration; the Rayleigh quotient recovers its sign.
//! - `power_top` / `power_bot` — largest / smallest *algebraic*
//!   eigenvalue (handles indefinite matrices by checking the sign of
//!   the magnitude winner and shifting if needed).
//! - `top_k_algebraic` — the `k` largest algebraic eigenvalues via
//!   repeated `power_top` with rank-1 deflation.
//! - `frob_sq` — `‖Z‖_F² = Σ λᵢ²`, the normaliser.
//!
//! For an n × n matrix, each call is O(n²) per iteration and converges
//! geometrically with rate set by the eigenvalue gap.

use std::f64::consts::PI;

use crate::histogram_curvature::gromov_hyperbolicity;

const POWER_MAX_ITER: usize = 500;
const POWER_TOL: f64 = 1e-10;

// ── Linear algebra primitives ──────────────────────────────────────────────

/// `y ← A x` for symmetric `A` stored row-major flat.
fn matvec(a: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let row = &a[i * n..(i + 1) * n];
        let mut s = 0.0;
        for j in 0..n {
            s += row[j] * x[j];
        }
        y[i] = s;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f64]) {
    let nm = dot(v, v).sqrt();
    if nm > 1e-20 {
        for x in v.iter_mut() {
            *x /= nm;
        }
    }
}

/// Power iteration: largest-magnitude eigenvalue of symmetric `A`.
/// Returns `(λ, v)` with `‖v‖₂ = 1`.
fn power_max(a: &[f64], n: usize) -> (f64, Vec<f64>) {
    // Deterministic, well-mixed start (sin sequence has no special
    // alignment with typical eigenvectors).
    let mut v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
    normalize(&mut v);

    let mut w = vec![0.0; n];
    let mut lambda = 0.0;

    for _ in 0..POWER_MAX_ITER {
        matvec(a, n, &v, &mut w);
        let new_lambda = dot(&v, &w);
        normalize(&mut w);
        if (new_lambda - lambda).abs() < POWER_TOL * (lambda.abs().max(1.0)) {
            return (new_lambda, w);
        }
        lambda = new_lambda;
        std::mem::swap(&mut v, &mut w);
    }
    (lambda, v)
}

/// Largest *algebraic* eigenvalue (most positive) of symmetric `A`.
///
/// `power_max` converges to the largest-*magnitude* eigenvalue, which is
/// not necessarily the largest algebraic value: an indefinite matrix
/// like Z_hyperbolic has its most-negative eigenvalue dominating in
/// magnitude.  We check the sign and shift if needed.
fn power_top(a: &[f64], n: usize) -> (f64, Vec<f64>) {
    let (lambda_mag, v_mag) = power_max(a, n);
    if lambda_mag >= 0.0 {
        return (lambda_mag, v_mag);
    }
    // Largest magnitude is negative ⇒ subtract it to make all
    // eigenvalues non-negative; the new largest is the original
    // algebraic max.
    let mut b = a.to_vec();
    for i in 0..n {
        b[i * n + i] -= lambda_mag;
    }
    let (_, v) = power_max(&b, n);
    let mut av = vec![0.0; n];
    matvec(a, n, &v, &mut av);
    (dot(&v, &av), v)
}

/// Smallest *algebraic* eigenvalue (most negative) of symmetric `A`.
///
/// Returns `(λ_min, v_min)` where `λ_min` is recovered via the Rayleigh
/// quotient on `A` directly, avoiding cancellation when shifting.
fn power_bot(a: &[f64], n: usize) -> (f64, Vec<f64>) {
    let (lambda_mag, v_mag) = power_max(a, n);

    if lambda_mag <= 0.0 {
        // Largest magnitude is non-positive ⇒ it *is* the smallest
        // algebraic.  Recompute via Rayleigh on A for precision.
        let mut av = vec![0.0; n];
        matvec(a, n, &v_mag, &mut av);
        return (dot(&v_mag, &av), v_mag);
    }

    // Largest magnitude is positive ⇒ shift B = λ_mag · I − A; B has
    // largest eigenvalue λ_mag − λ_min.  Find its eigenvector and take
    // Rayleigh on the original A.
    let mut b = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] = -a[i * n + j];
        }
        b[i * n + i] += lambda_mag;
    }
    let (_, v) = power_max(&b, n);
    let mut av = vec![0.0; n];
    matvec(a, n, &v, &mut av);
    (dot(&v, &av), v)
}

/// The `k` largest *algebraic* eigenvalues of symmetric `A`, descending.
///
/// Repeated power iteration with rank-1 deflation: after locating each
/// eigenpair `(λᵢ, vᵢ)` we subtract `C · vᵢ vᵢᵀ` with `C > λ_top − λ_bot`,
/// which drives that direction below the rest of the spectrum so the
/// next `power_top` skips it.  Deflation preserves eigenvectors, so each
/// λ is recovered via the Rayleigh quotient on the *original* `A` (the
/// shift never leaks into the reported value).
fn top_k_algebraic(a: &[f64], n: usize, k: usize) -> Vec<f64> {
    let (lt, _) = power_top(a, n);
    let (lb, _) = power_bot(a, n);
    let c = 2.0 * (lt - lb).abs() + 1.0;

    let mut deflated = a.to_vec();
    let mut av = vec![0.0; n];
    let mut out = Vec::with_capacity(k.min(n));
    for _ in 0..k.min(n) {
        let (_, v) = power_top(&deflated, n);
        matvec(a, n, &v, &mut av);
        out.push(dot(&v, &av));
        for i in 0..n {
            for j in 0..n {
                deflated[i * n + j] -= c * v[i] * v[j];
            }
        }
    }
    out
}

/// `‖A‖_F² = Σᵢⱼ Aᵢⱼ² = Σᵢ λᵢ²` — the normaliser for the signature residual.
fn frob_sq(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
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

/// Fraction of `‖Z‖_F²` outside the best signature-respecting rank-`r`
/// approximation, given the squared norms of the eigenvalues that model
/// keeps.  `1` minus the captured energy, clamped to `[0, 1]`.
fn signature_residual(total: f64, retained_sq: f64) -> f64 {
    if total < 1e-300 {
        return 0.0;
    }
    (1.0 - retained_sq / total).clamp(0.0, 1.0)
}

/// Spherical residual `ρ(r)` for a `dim`-dimensional embedding.  A
/// `dim`-sphere lives in Euclidean `R^{dim+1}`, so a conforming
/// `Z_spherical` is PSD of rank `dim+1`: retain the `dim+1` most-positive
/// eigenvalues.  Any negative eigenvalue violates the PSD model and so
/// counts fully toward the residual.
fn spherical_residual(d: &[f64], n: usize, dim: usize, r: f64) -> f64 {
    let z = build_z_spherical(d, n, r);
    let retained: f64 = top_k_algebraic(&z, n, dim + 1).iter().map(|l| l * l).sum();
    signature_residual(frob_sq(&z), retained)
}

/// Hyperbolic residual `ρ(r)` for a `dim`-dimensional embedding.  A
/// `dim`-dimensional hyperbolic configuration lives in Lorentzian
/// `R^{dim,1}`, so a conforming `Z_hyperbolic` has signature
/// `(1 negative, dim positive)`: retain the most-negative eigenvalue
/// plus the `dim` most-positive ones.
fn hyperbolic_residual(d: &[f64], n: usize, dim: usize, r: f64) -> f64 {
    let z = build_z_hyperbolic(d, n, r);
    let (lmin, _) = power_bot(&z, n);
    let pos: f64 = top_k_algebraic(&z, n, dim).iter().map(|l| l * l).sum();
    signature_residual(frob_sq(&z), lmin * lmin + pos)
}

// ── Search over r ───────────────────────────────────────────────────────────

/// Result of fitting one constant-curvature model.
#[derive(Debug, Clone, Copy)]
pub struct WilsonFit {
    /// Best-fit radius of curvature.  Sectional curvature is +1/r²
    /// (spherical) or −1/r² (hyperbolic).
    pub radius: f64,
    /// Normalised signature residual `ρ ∈ [0, 1]` at `radius` — the
    /// fraction of `‖Z‖_F²` outside the rank-`(dim+1)` signature
    /// subspace.  Small ⇒ data conforms well to the model.
    pub residual: f64,
    /// True if `radius` is at the upper bound of the search range,
    /// which usually means the data is closer to Euclidean than to the
    /// curved model at any finite radius.
    pub at_upper_bound: bool,
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

/// Fit a `dim`-dimensional spherical model: find `r*` minimising the
/// signature residual `ρ(r)` (energy outside the rank-`(dim+1)` PSD
/// subspace of `Z_spherical(r)`).
///
/// Search bounds (Section V.A): r ≥ d_max/π (so the largest geodesic
/// distance fits on the sphere) and r ≤ 5·d_max (well into the
/// Euclidean limit; the paper's quoted upper bound `3·d_min` is a typo
/// — see the discussion in this file's history).
pub fn fit_spherical(distances: &[f64], n: usize, dim: usize) -> WilsonFit {
    let d_max = distances.iter().cloned().fold(0.0_f64, f64::max);
    let r_lower = d_max / PI;
    let r_upper = 5.0 * d_max;

    let mut residual_at = |r: f64| -> f64 { spherical_residual(distances, n, dim, r) };

    let (r_star, residual, at_upper) = minimise_log_spaced(r_lower, r_upper, 30, &mut residual_at);
    WilsonFit {
        radius: r_star,
        residual,
        at_upper_bound: at_upper,
    }
}

/// Fit a `dim`-dimensional hyperbolic model: find `r*` minimising the
/// signature residual `ρ(r)` (energy outside the
/// `(1 negative, dim positive)` subspace of `Z_hyperbolic(r)`).
///
/// Search bounds: r ≥ d_max/20 (keeps `cosh(d_max/r) ≤ cosh(20) ≈ 2.4·10⁸`,
/// safe from overflow) and r ≤ d_max.  Hyperbolic space is non-compact,
/// so the geodesic-fits-on-space lower bound from the spherical case
/// does not apply — but for r ≫ d_max, cosh(d/r) ≈ 1 and Z becomes
/// rank-1 dominated, producing a deep but artifactual "Euclidean limit"
/// minimum.  Capping at d_max excludes that regime and lets the genuine
/// hyperbolic minimum (at r comparable to the curvature radius) win.
pub fn fit_hyperbolic(distances: &[f64], n: usize, dim: usize) -> WilsonFit {
    let d_max = distances.iter().cloned().fold(0.0_f64, f64::max);
    let r_lower = d_max / 20.0;
    let r_upper = d_max;

    let mut residual_at = |r: f64| -> f64 { hyperbolic_residual(distances, n, dim, r) };

    let (r_star, residual, at_upper) = minimise_log_spaced(r_lower, r_upper, 30, &mut residual_at);
    WilsonFit {
        radius: r_star,
        residual,
        at_upper_bound: at_upper,
    }
}

// ── Detection ───────────────────────────────────────────────────────────────

/// Curvature-detection result combining the spherical and hyperbolic
/// fits with a coarse-geometric (Gromov) hyperbolicity check.
#[derive(Debug, Clone, Copy)]
pub struct GeometryDetection {
    pub spherical: WilsonFit,
    pub hyperbolic: WilsonFit,
    /// Normalised 4-point Gromov δ (90th percentile, divided by the
    /// median pairwise distance).  Small values (≲ `GROMOV_THRESHOLD`)
    /// indicate δ-hyperbolic / "negatively curved" metric structure;
    /// Euclidean data typically lands near 0.24.
    pub gromov_delta: f64,
    /// `"spherical"`, `"hyperbolic"`, or `"euclidean"`.
    pub best_geometry: &'static str,
    /// Signed sectional curvature estimate at the detected radius.
    /// `0.0` when the detection is `"euclidean"`.
    pub curvature: f64,
}

/// Diagnostic: spherical signature residual `ρ` at a single r for a
/// `dim`-dimensional model.  Exposed so tests can plot the residual curve.
pub fn spherical_residual_at(distances: &[f64], n: usize, dim: usize, r: f64) -> f64 {
    spherical_residual(distances, n, dim, r)
}

/// Diagnostic: hyperbolic signature residual `ρ` at a single r for a
/// `dim`-dimensional model.
pub fn hyperbolic_residual_at(distances: &[f64], n: usize, dim: usize, r: f64) -> f64 {
    hyperbolic_residual(distances, n, dim, r)
}

/// Minimum angular extent `d_max / r*` required for the spherical fit
/// to be treated as genuine.  A uniform sample on a sphere has
/// `d_max ≈ π · r` and `d/r` is literally the subtended angle in
/// radians, so 2.5 (≈ 0.8 π) accepts near-full coverage while rejecting
/// Euclidean data which fits a small cap on a very large sphere with
/// arbitrarily small residual.
pub const SPHERICAL_ANGULAR_MIN: f64 = 2.5;

/// Maximum normalised 4-point Gromov δ for the data to be treated as
/// δ-hyperbolic.  δ is the supremum over quadruples of
/// `(S_max − S_mid)/2` where the three S values are the pair-distance
/// sums; we use the 90th percentile of sampled quadruples, normalised
/// by the median pairwise distance.
///
/// Theoretical backing (unlike the prior `d_max / r` "angular extent"
/// heuristic on a non-compact space):
///
/// - `H²` with `K = −1` has 4-point hyperbolicity constant `log(2)/2 ≈
///   0.347`, and the sup of δ scales as `r` under `K → −1/r²`, so
///   `δ / d_max → 0` as the data covers many curvature radii.
/// - `ℝⁿ` is *not* δ-hyperbolic; the normalised δ on a uniform sample
///   stabilises around ~0.24 (empirically) and the underlying δ scales
///   with the diameter.
///
/// `0.18` sits in the (wide) gap between the two regimes observed in
/// the test fixtures (≲ 0.15 hyperbolic, ~0.24 Euclidean).
pub const GROMOV_THRESHOLD: f64 = 0.18;

/// Number of random 4-tuples sampled for the Gromov δ estimate.
const GROMOV_SAMPLES: usize = 5000;

/// Detect curvature from a distance matrix.
///
/// Decision rule:
///
/// 1. **Spherical** if `d_max / r_s* ≥ SPHERICAL_ANGULAR_MIN`.  On a
///    sphere `d/r` is the subtended angle, so this is a literal
///    coverage criterion.
/// 2. **Hyperbolic** if the normalised 4-point Gromov δ is below
///    `GROMOV_THRESHOLD` AND the Wilson hyperbolic fit did not pin at
///    its upper bound (which would mean Wilson's residual minimum is
///    the Euclidean-limit artifact, so the reported curvature
///    magnitude cannot be trusted).
/// 3. **Euclidean** otherwise.
///
/// `dim` is the target embedding dimension — the spherical and
/// hyperbolic models are fitted as `dim`-dimensional manifolds (rank
/// `dim+1` Gram matrices).
pub fn detect_geometry(distances: &[f64], n: usize, dim: usize) -> GeometryDetection {
    let spherical = fit_spherical(distances, n, dim);
    let hyperbolic = fit_hyperbolic(distances, n, dim);

    let d_max = distances.iter().cloned().fold(0.0_f64, f64::max);
    let s_angular = d_max / spherical.radius;
    let gromov_delta = gromov_hyperbolicity(distances, n, GROMOV_SAMPLES);

    let (best_geometry, curvature) = if s_angular >= SPHERICAL_ANGULAR_MIN {
        ("spherical", 1.0 / (spherical.radius * spherical.radius))
    } else if !hyperbolic.at_upper_bound && gromov_delta < GROMOV_THRESHOLD {
        ("hyperbolic", -1.0 / (hyperbolic.radius * hyperbolic.radius))
    } else {
        ("euclidean", 0.0)
    };

    GeometryDetection {
        spherical,
        hyperbolic,
        gromov_delta,
        best_geometry,
        curvature,
    }
}
