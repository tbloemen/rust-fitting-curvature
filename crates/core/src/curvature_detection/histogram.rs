//! Shell-density geometry detection from pairwise distances.
//!
//! **Purpose.** This is the deliberately *naive* baseline detector: it
//! fits each constant-curvature shell-density curve to the empirical
//! profile and picks whichever fits best (highest R²).  It exists to
//! demonstrate that curvature *sign* cannot be reliably read off the
//! residuals — Euclidean is the `c → 0` limit of both the spherical and
//! hyperbolic densities, so the three models are nearly degenerate on
//! finite samples and the best-R² winner is often the wrong sign.  The
//! growing-ball Gromov test ([`super::gromov_ball_curve::detect_hyperbolic`]) is the
//! principled alternative; the contrast with this detector is the point.
//!
//! **Mathematical basis.** In a d-dimensional Riemannian manifold of constant
//! curvature, the surface area of a geodesic sphere of radius r is proportional
//! to:
//!
//! | Geometry   | Surface area                            |
//! |------------|-----------------------------------------|
//! | Euclidean  | r^(d−1)                                 |
//! | Spherical  | (sin(√c · r) / √c)^(d−1)                |
//! | Hyperbolic | (sinh(√c · r) / √c)^(d−1)               |
//!
//! The 1/√c prefactor only contributes a constant to log density, so it is
//! absorbed into the regression intercept and the fit recovers d from the
//! slope and c from the curvature scale that maximises R².
//!
//! **Algorithm.**
//! 1. For each central point, collect its neighbour distances and build a
//!    shell-density profile (see [`shell_density_profile`]).
//! 2. Fit all three curves (Euclidean power-law, spherical sin, hyperbolic
//!    sinh) to the same sampled (radius, log-density) values via
//!    [`fit_geometries`].
//! 3. [`detect_geometry`] returns the [`GeometryVerdict`] for whichever
//!    curve has the highest R²; ties favour the simpler Euclidean model.
//!    No Gromov gate, no peak-shape heuristics — just the residuals.

use super::signature::GeometryVerdict;

/// Result of fitting one geometry model.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Estimated intrinsic dimension (slope + 1).
    pub dim: f64,
    /// Coefficient of determination R² ∈ [0, 1].
    pub r_squared: f64,
    /// Intercept in the log-space regression (log of scale constant).
    pub log_scale: f64,
    /// Estimated curvature magnitude c ≥ 0.  The signed sectional curvature
    /// is k = -c for hyperbolic fits, k = +c for spherical fits, and k = 0
    /// for Euclidean fits (in which case this field is 0).
    pub curvature_scale: f64,
}

/// The three constant-curvature fits to the shell-density profile.
/// Exposed so the near-degeneracy of the residuals can be inspected —
/// that the R² values are typically close is exactly why
/// [`detect_geometry`]'s best-fit pick is an unreliable curvature sign.
#[derive(Debug, Clone)]
pub struct GeometryFits {
    pub euclidean: FitResult,
    pub spherical: FitResult,
    pub hyperbolic: FitResult,
}

/// Full shell-density profile and the diagnostics derived from it.
#[derive(Debug, Clone)]
pub struct ShellProfile {
    /// Bin centres of the peak-truncated profile (rising regime only).
    pub bin_centers_trunc: Vec<f64>,
    /// Density of the peak-truncated profile.
    pub density_trunc: Vec<f64>,
    /// Bin centres of the full 95th-percentile range.
    pub bin_centers_full: Vec<f64>,
    /// Density of the full 95th-percentile range.
    pub density_full: Vec<f64>,
    /// Position of the (smoothed) density peak on the full profile, as a
    /// fraction of n_bins: 0 ≈ left edge, 1 ≈ right edge.
    pub peak_ratio: f64,
    /// Fraction of the full profile's density mass that sits past the
    /// peak bin.  Spherical sin^(d−1) densities are roughly symmetric
    /// around √c·r = π/2, so tail_mass ≈ 0.5.  Euclidean and hyperbolic
    /// profiles peak near the sample boundary (rising volume-growth law
    /// cut off by finite-N effects), so most of the mass sits *before*
    /// the peak and tail_mass is small (≲ 0.3).  This diagnostic is far
    /// more dimension-stable than `peak_ratio` because it captures the
    /// global *shape* (symmetric vs clipped) rather than the peak
    /// location, which drifts inward in high-d due to distance
    /// concentration.
    pub tail_mass: f64,
}

/// Compute the empirical shell density profile from a distance matrix.
///
/// Uses the most central ~10 % of points (smallest mean distance to the
/// rest) as references; central points have good boundary clearance so
/// their neighbour-distance profile extends into the large-r region where
/// curvature signals live.
///
/// Returns both the peak-truncated and full versions; see [`ShellProfile`].
///
/// * `distances`    — flat row-major n×n distance matrix.
/// * `n_points`     — n.
/// * `n_bins`       — histogram resolution (30–50 recommended).
pub fn shell_density_profile(distances: &[f64], n_points: usize, n_bins: usize) -> ShellProfile {
    let empty = ShellProfile {
        bin_centers_trunc: vec![0.0; n_bins],
        density_trunc: vec![0.0; n_bins],
        bin_centers_full: vec![0.0; n_bins],
        density_full: vec![0.0; n_bins],
        peak_ratio: 1.0,
        tail_mass: 0.0,
    };

    if n_points < 3 {
        return empty;
    }

    let n_ref = (n_points / 10).max(3);

    let mean_dist: Vec<f64> = (0..n_points)
        .map(|i| {
            distances[i * n_points..(i + 1) * n_points]
                .iter()
                .sum::<f64>()
                / (n_points as f64 - 1.0)
        })
        .collect();

    let mut order: Vec<usize> = (0..n_points).collect();
    order.sort_by(|&a, &b| mean_dist[a].partial_cmp(&mean_dist[b]).unwrap());

    let k = n_points - 1;

    let mut local_dists = Vec::with_capacity(n_ref * k);
    for &ri in &order[..n_ref] {
        let row = &distances[ri * n_points..(ri + 1) * n_points];
        let mut dists: Vec<f64> = row
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != ri)
            .map(|(_, d)| *d)
            .collect();
        dists.sort_by(|a: &f64, b| a.partial_cmp(b).unwrap());
        local_dists.extend_from_slice(&dists[..k.min(dists.len())]);
    }

    if local_dists.is_empty() {
        return empty;
    }

    // Full range: 95th percentile of neighbour distances.  This is the
    // working window for the spherical fit (which needs the post-peak
    // descent) and for diagnosing where the peak sits.
    local_dists.sort_by(|a: &f64, b| a.partial_cmp(b).unwrap());
    let idx95 = ((local_dists.len() as f64) * 0.95) as usize;
    let r_full = local_dists[idx95.min(local_dists.len() - 1)];

    if r_full < 1e-12 {
        return empty;
    }

    let (bin_centers_full, density_full) = histogram(&local_dists, r_full, n_bins);

    // Locate the smoothed peak on the full profile.  For E^d/H^d this
    // marks the onset of boundary clipping; for S^d it is the intrinsic
    // density peak near √c·r = π/2.
    let smoothed: Vec<f64> = (0..n_bins)
        .map(|i| {
            let lo = if i > 0 { i - 1 } else { 0 };
            let hi = (i + 1).min(n_bins - 1);
            density_full[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64
        })
        .collect();

    let peak_bin = smoothed
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(n_bins - 1);

    let peak_ratio = (peak_bin as f64 + 0.5) / n_bins as f64;

    // Tail mass past the peak.  density_full integrates to ~1 with bin
    // width r_full / n_bins, so this is simply the post-peak fraction of
    // the area under the curve.
    let bin_width_full = r_full / n_bins as f64;
    let tail_mass: f64 = density_full[(peak_bin + 1).min(n_bins)..]
        .iter()
        .sum::<f64>()
        * bin_width_full;

    // Truncated profile: re-bin out to the peak.  This is the working
    // window for the Euclidean and hyperbolic fits, where the signal is in
    // the rising regime and the tail would be dominated by sample-
    // boundary clipping.
    let r_trunc = (peak_bin as f64 + 1.0) * bin_width_full;

    let (bin_centers_trunc, density_trunc) = if r_trunc < 1e-12 {
        (vec![0.0; n_bins], vec![0.0; n_bins])
    } else {
        histogram(&local_dists, r_trunc, n_bins)
    };

    ShellProfile {
        bin_centers_trunc,
        density_trunc,
        bin_centers_full,
        density_full,
        peak_ratio,
        tail_mass,
    }
}

/// Build a normalised density histogram of `sorted_dists` over (0, r_max)
/// with `n_bins` equal-width bins.  Returns `(bin_centers, density)`; the
/// density integrates to ~1 so it is comparable across datasets.
fn histogram(sorted_dists: &[f64], r_max: f64, n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    let bin_width = r_max / n_bins as f64;
    let mut counts = vec![0.0f64; n_bins];
    for &d in sorted_dists {
        if d > 1e-12 && d < r_max {
            let bin = ((d / r_max) * n_bins as f64) as usize;
            counts[bin.min(n_bins - 1)] += 1.0;
        }
    }
    let total: f64 = counts.iter().sum::<f64>() * bin_width;
    let bin_centers: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * bin_width).collect();
    let density: Vec<f64> = if total > 1e-12 {
        counts.iter().map(|&c| c / total).collect()
    } else {
        vec![0.0; n_bins]
    };
    (bin_centers, density)
}

/// Ordinary least squares: y = slope·x + intercept.
/// Returns `(slope, intercept, r_squared)`.
fn ols(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len();
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    let n_f = n as f64;
    let mean_x = x.iter().sum::<f64>() / n_f;
    let mean_y = y.iter().sum::<f64>() / n_f;

    let ss_xx: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
    let ss_xy: f64 = x
        .iter()
        .zip(y)
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    if ss_xx.abs() < 1e-12 {
        return (0.0, mean_y, 0.0);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y)
        .map(|(&xi, &yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();

    let r_squared = if ss_tot < 1e-12 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };

    (slope, intercept, r_squared)
}

/// Fit one model: transform r → log(f(r)) and regress log density on it.
fn fit_model(r_vals: &[f64], log_density: &[f64]) -> FitResult {
    let pairs: Vec<(f64, f64)> = r_vals
        .iter()
        .zip(log_density)
        .filter_map(|(&r, &ld)| (if r > 1e-10 { Some(r.ln()) } else { None }).map(|tx| (tx, ld)))
        .collect();

    if pairs.len() < 3 {
        return FitResult {
            dim: 1.0,
            r_squared: 0.0,
            log_scale: 0.0,
            curvature_scale: 0.0,
        };
    }

    let xs: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
    let ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
    let (slope, intercept, r2) = ols(&xs, &ys);

    FitResult {
        dim: slope + 1.0,
        r_squared: r2.max(0.0),
        log_scale: intercept,
        curvature_scale: 0.0,
    }
}

/// Search a log-spaced grid of curvature scales c and pick the c maximising
/// R² of the OLS fit log density vs. log(transform(√c·r)).
fn fit_curved_model(
    transform: impl Fn(f64, f64) -> Option<f64>,
    r_vals: &[f64],
    log_density: &[f64],
    c_grid: &[f64],
) -> FitResult {
    let mut best = FitResult {
        dim: 1.0,
        r_squared: 0.0,
        log_scale: 0.0,
        curvature_scale: 0.0,
    };

    for &c in c_grid {
        let sqrt_c = c.sqrt();
        let pairs: Vec<(f64, f64)> = r_vals
            .iter()
            .zip(log_density)
            .filter_map(|(&r, &ld)| transform(r, sqrt_c).map(|tx| (tx, ld)))
            .collect();

        if pairs.len() < 3 {
            continue;
        }

        let xs: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
        let (slope, intercept, r2) = ols(&xs, &ys);

        if r2 > best.r_squared {
            best = FitResult {
                dim: slope + 1.0,
                r_squared: r2.max(0.0),
                log_scale: intercept,
                curvature_scale: c,
            };
        }
    }

    best
}

/// Build a log-spaced grid of curvature magnitudes such that √c·r_max spans
/// `[arg_min, arg_max]`.  This keeps the search adapted to the data scale.
fn curvature_grid(r_max: f64, arg_min: f64, arg_max: f64, n: usize) -> Vec<f64> {
    if r_max < 1e-10 || n < 2 {
        return Vec::new();
    }
    let c_min = (arg_min / r_max).powi(2);
    let c_max = (arg_max / r_max).powi(2);
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            c_min * (c_max / c_min).powf(t)
        })
        .collect()
}

/// Fit a hyperbolic model with a free curvature magnitude c.
/// Searches sqrt(c)·r_max ∈ [0.01, 20] on a 60-point log-spaced grid.
fn fit_hyperbolic_curved(r_vals: &[f64], log_density: &[f64]) -> FitResult {
    let r_max = r_vals.iter().cloned().fold(0.0_f64, f64::max);
    let c_grid = curvature_grid(r_max, 0.01, 20.0, 60);
    fit_curved_model(
        |r, sqrt_c| {
            let s = (sqrt_c * r).sinh();
            if s > 1e-10 && s.is_finite() {
                Some(s.ln())
            } else {
                None
            }
        },
        r_vals,
        log_density,
        &c_grid,
    )
}

/// Fit a spherical model with a free curvature magnitude c.
/// Searches sqrt(c)·r_max ∈ [0.01, 0.95π] on a 60-point log-spaced grid;
/// the upper cap keeps sin(√c·r) bounded away from its zero at π.
fn fit_spherical_curved(r_vals: &[f64], log_density: &[f64]) -> FitResult {
    let r_max = r_vals.iter().cloned().fold(0.0_f64, f64::max);
    let c_grid = curvature_grid(r_max, 0.01, 0.95 * std::f64::consts::PI, 60);
    fit_curved_model(
        |r, sqrt_c| {
            let arg = sqrt_c * r;
            if arg < std::f64::consts::PI {
                let s = arg.sin();
                if s > 1e-10 {
                    Some(s.ln())
                } else {
                    None
                }
            } else {
                None
            }
        },
        r_vals,
        log_density,
        &c_grid,
    )
}

/// Fit all three constant-curvature shell-density curves to the *same*
/// sampled (radius, log-density) values.
///
/// Uses the peak-truncated (rising-regime) profile, where the finite-sample
/// boundary clipping of the tail is excluded.  Curve transforms: Euclidean
/// `log r`, spherical `log sin(√c·r)`, hyperbolic `log sinh(√c·r)`, with `c`
/// grid-searched for the curved models (`curvature_scale = 0` for Euclidean).
/// Fitting all three to identical points makes their R² directly comparable
/// — and reveals the degeneracy that defeats the detector: in the rising
/// regime all three curves are nearly the same power law, so their R²
/// barely differ and the best-fit curvature sign is essentially noise.
///
/// * `distances`    — flat row-major n×n distance matrix.
/// * `n_points`     — n.
/// * `n_bins`       — histogram resolution (30–50 works well).
pub fn fit_geometries(distances: &[f64], n_points: usize, n_bins: usize) -> GeometryFits {
    let profile = shell_density_profile(distances, n_points, n_bins);

    let (r, log_d): (Vec<f64>, Vec<f64>) = profile
        .bin_centers_trunc
        .iter()
        .copied()
        .zip(profile.density_trunc.iter().copied())
        .filter(|&(r, d)| r > 1e-10 && d > 1e-10)
        .map(|(r, d)| (r, d.ln()))
        .unzip();

    let zero = FitResult {
        dim: 1.0,
        r_squared: 0.0,
        log_scale: 0.0,
        curvature_scale: 0.0,
    };
    if r.len() < 3 {
        return GeometryFits {
            euclidean: zero.clone(),
            spherical: zero.clone(),
            hyperbolic: zero,
        };
    }

    GeometryFits {
        euclidean: fit_model(&r, &log_d),
        spherical: fit_spherical_curved(&r, &log_d),
        hyperbolic: fit_hyperbolic_curved(&r, &log_d),
    }
}

/// Detect the underlying geometry by picking the best-fitting shell-density
/// curve.
///
/// Deliberately naive: fit all three curves ([`fit_geometries`]) and
/// return the [`GeometryVerdict`] for whichever has the highest R²; ties
/// favour the simpler Euclidean model.  Because Euclidean is the `c → 0`
/// limit of both curved densities, the residuals barely separate the
/// models on finite samples, so the returned curvature *sign* is often
/// wrong — that is the intended demonstration, motivating the principled
/// growing-ball test in [`super::gromov_ball_curve::detect_hyperbolic`].
///
/// * `distances`    — flat row-major n×n distance matrix.
/// * `n_points`     — n.
/// * `n_bins`       — histogram resolution (30–50 works well).
pub fn detect_geometry(distances: &[f64], n_points: usize, n_bins: usize) -> GeometryVerdict {
    let fits = fit_geometries(distances, n_points, n_bins);

    // (label, signed curvature, R²).  Euclidean first so it wins ties.
    let candidates = [
        ("euclidean", 0.0, fits.euclidean.r_squared),
        (
            "spherical",
            fits.spherical.curvature_scale,
            fits.spherical.r_squared,
        ),
        (
            "hyperbolic",
            -fits.hyperbolic.curvature_scale,
            fits.hyperbolic.r_squared,
        ),
    ];

    let (best_geometry, curvature, _) = candidates
        .into_iter()
        .reduce(|best, c| if c.2 > best.2 { c } else { best })
        .unwrap();

    GeometryVerdict {
        best_geometry,
        curvature,
    }
}
