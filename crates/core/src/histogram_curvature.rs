//! Geometry detection from pairwise distances.
//!
//! Given a distance matrix, the algorithm detects whether the underlying
//! geometry is Euclidean, spherical, or hyperbolic, estimates the intrinsic
//! dimension d, and (for non-Euclidean cases) estimates the curvature
//! magnitude c, where the sectional curvature is k = -c (hyperbolic) or
//! k = +c (spherical).
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
//! 1. For each central point, collect its neighbour distances.
//! 2. Build TWO shell density histograms:
//!    - a **full** profile over the 95th-percentile range, and
//!    - a **peak-truncated** profile (rising regime only) that excludes
//!      boundary-clipping artefacts.
//! 3. Euclidean and hyperbolic fits use the truncated profile (their
//!    signature is the rising regime; the tail is contaminated by the
//!    finite-sample boundary).
//! 4. Spherical fit uses the **full** profile.  In S^d the density
//!    sin^(d−1)(√c·r) genuinely peaks at √c·r = π/2 and descends to 0 at
//!    √c·r = π; truncating at the peak removes the disambiguating signal.
//! 5. Model selection:
//!    - Strong spherical signal: density peak well-interior to the data
//!      range (`peak_ratio < PEAK_RATIO_SPHERICAL`) and spherical R² above
//!      `SPHERICAL_R2_MIN` → spherical.
//!    - Strong hyperbolic signal: normalised Gromov δ below
//!      `GROMOV_THRESHOLD` → hyperbolic.
//!    - Otherwise compare R² on the truncated profile with `R2_MARGIN`.

pub const GROMOV_THRESHOLD: f64 = 0.15;

/// Loose peak-position ratio: above this threshold the density's peak
/// sits too close to the sample boundary to be a real spherical peak at
/// √c·r = π/2.  Used as a necessary (not sufficient) condition combined
/// with the spherical-vs-Euclidean R² gap.  The 0.72 value clears the
/// histogram-quantisation boundary at 0.700 (peak_bin=24/n_bins=35) so
/// high-d sphere realisations (S⁵ at SEED=42 lands at 0.671) whose peak
/// drifts toward the right edge are not silently excluded.  The price
/// is roughly one Euclidean false positive per ten seeds.
pub const PEAK_RATIO_SPHERICAL: f64 = 0.72;

/// Tight peak-position ratio: when the peak sits this far into the
/// profile, it is the dominant signal on its own.  In S^d the intrinsic
/// peak is at √c·r = π/2 (peak_ratio ≈ 0.5); peak_ratio < 0.6 is reached
/// reliably on S² and frequently on S³ at N=400.  Below this threshold
/// no additional confirmation is required.
pub const PEAK_RATIO_SPHERICAL_STRONG: f64 = 0.60;

/// Minimum (spherical_R² − Euclidean_R²) gap on the full profile.  Sin
/// can model the post-peak descent; r^(d−1) cannot.  On genuine S^d the
/// gap is typically 0.15–0.65 across seeds; on Euclidean data it stays
/// below ~0.14 on most seeds, with rare excursions to ~0.15 on
/// adversarial realisations — those are the dominant residual failure
/// mode of the spherical-vs-Euclidean discrimination at this N.
pub const SPHERICAL_R2_GAP: f64 = 0.15;

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

/// Full result of geometry detection.
#[derive(Debug, Clone)]
pub struct GeometryDetection {
    pub euclidean: FitResult,
    pub spherical: FitResult,
    pub hyperbolic: FitResult,
    /// `"euclidean"`, `"spherical"`, `"hyperbolic"`, or `"unknown"`.
    pub best_geometry: &'static str,
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
                if s > 1e-10 { Some(s.ln()) } else { None }
            } else {
                None
            }
        },
        r_vals,
        log_density,
        &c_grid,
    )
}

/// Detect the underlying geometry and estimate the intrinsic dimension.
///
/// * `distances`    — flat row-major n×n distance matrix.
/// * `n_points`     — n.
/// * `n_bins`       — histogram resolution (30–50 works well).
pub fn detect_geometry(distances: &[f64], n_points: usize, n_bins: usize) -> GeometryDetection {
    let profile = shell_density_profile(distances, n_points, n_bins);

    // Truncated profile: Euclidean and hyperbolic fits.  Their signature
    // lives in the rising regime; the tail of the full profile is
    // dominated by sample-boundary clipping.
    let (r_trunc, log_d_trunc): (Vec<f64>, Vec<f64>) = profile
        .bin_centers_trunc
        .iter()
        .copied()
        .zip(profile.density_trunc.iter().copied())
        .filter(|&(r, d)| r > 1e-10 && d > 1e-10)
        .map(|(r, d)| (r, d.ln()))
        .unzip();

    // Full profile: spherical fit.  The disambiguating signal for S^d is
    // the post-peak descent of sin^(d−1)(√c·r), which the truncated
    // profile discards.
    let (r_full, log_d_full): (Vec<f64>, Vec<f64>) = profile
        .bin_centers_full
        .iter()
        .copied()
        .zip(profile.density_full.iter().copied())
        .filter(|&(r, d)| r > 1e-10 && d > 1e-10)
        .map(|(r, d)| (r, d.ln()))
        .unzip();

    if r_trunc.len() < 3 || r_full.len() < 3 {
        let zero = FitResult {
            dim: 1.0,
            r_squared: 0.0,
            log_scale: 0.0,
            curvature_scale: 0.0,
        };
        return GeometryDetection {
            euclidean: zero.clone(),
            spherical: zero.clone(),
            hyperbolic: zero.clone(),
            best_geometry: "unknown",
        };
    }

    // Euclidean on truncated profile (rising regime).  Reported as the
    // canonical Euclidean fit, also used for the curvature-scale=0 case.
    let euclidean = fit_model(&r_trunc, &log_d_trunc);

    // Hyperbolic: log ρ ~ (d−1) log sinh(√c r), c searched.
    let hyperbolic = fit_hyperbolic_curved(&r_trunc, &log_d_trunc);

    // Spherical: log ρ ~ (d−1) log sin(√c r), c searched, on the full
    // profile so the post-peak sin descent participates in the fit.
    let spherical = fit_spherical_curved(&r_full, &log_d_full);

    // Auxiliary: Euclidean fit on the **full** profile, only used as a
    // spherical-vs-Euclidean discriminator (not reported).  Sin can model
    // a rise-and-descent profile; r^(d−1) cannot — so on S^d the gap
    // (S_full − E_full) is sizeable, while on E^d sin with c → 0
    // degenerates to a power law and the gap is small.
    let euclidean_full = fit_model(&r_full, &log_d_full);
    let r2_gap = spherical.r_squared - euclidean_full.r_squared;

    // --- Model selection ---
    //
    // 1. Spherical: tiered.  Either the density peak is firmly interior
    //    (peak_ratio < tight threshold) — a near-unambiguous signature
    //    of √c·r = π/2 — OR the peak is moderately interior AND the
    //    spherical fit beats the Euclidean fit by a wide R² margin on
    //    the full profile.  The tiered structure is needed because
    //    high-d spheres concentrate distances near π/2 and finite-N
    //    noise shifts the empirical peak rightward; in those cases the
    //    R² gap carries the signal.
    //
    // 2. Hyperbolic: tree-like 4-point Gromov δ below threshold.
    //    Euclidean is the c → 0 limit of sinh, so log-space R² differs
    //    by only ~1e-3 between the two on Euclidean data — too small to
    //    discriminate reliably across seeds.  Gromov δ, by contrast,
    //    diverges between hyperbolic (≲ 0.15) and Euclidean (~0.24).
    //
    // 3. Euclidean: default.
    let gromov = gromov_hyperbolicity(distances, n_points, 5000);

    let strong_spherical = profile.peak_ratio < PEAK_RATIO_SPHERICAL_STRONG
        || (profile.peak_ratio < PEAK_RATIO_SPHERICAL && r2_gap > SPHERICAL_R2_GAP);

    let best_geometry = if strong_spherical {
        "spherical"
    } else if gromov < GROMOV_THRESHOLD {
        "hyperbolic"
    } else {
        "euclidean"
    };

    GeometryDetection {
        euclidean,
        spherical,
        hyperbolic,
        best_geometry,
    }
}

/// Estimate Gromov 4-point hyperbolicity from the distance matrix.
///
/// Samples random 4-tuples, computes δ = (S_max − S_mid) / 2 for each
/// (where S are the three distance-pair sums), and returns the 90th
/// percentile of δ normalised by the median pairwise distance.
///
/// Hyperbolic spaces have small normalised δ (bounded by log(2)/R for
/// curvature −1 and typical distance R), while Euclidean/spherical
/// spaces produce larger values.
pub fn gromov_hyperbolicity(distances: &[f64], n: usize, n_samples: usize) -> f64 {
    if n < 4 {
        return 0.0;
    }

    // Simple deterministic PRNG for reproducible sampling.
    let mut state: u64 = 0xdeadbeef;
    let mut rng = || -> usize {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as usize) % n
    };

    let mut deltas = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let a = rng();
        let mut b = rng();
        while b == a {
            b = rng();
        }
        let mut c = rng();
        while c == a || c == b {
            c = rng();
        }
        let mut d = rng();
        while d == a || d == b || d == c {
            d = rng();
        }

        let dab = distances[a * n + b];
        let dcd = distances[c * n + d];
        let dac = distances[a * n + c];
        let dbd = distances[b * n + d];
        let dad = distances[a * n + d];
        let dbc = distances[b * n + c];

        let s1 = dab + dcd;
        let s2 = dac + dbd;
        let s3 = dad + dbc;

        let mut sums = [s1, s2, s3];
        sums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let delta = (sums[2] - sums[1]) / 2.0;
        deltas.push(delta);
    }

    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // 90th percentile δ.
    let p90_idx = (deltas.len() as f64 * 0.90) as usize;
    let delta_90 = deltas[p90_idx.min(deltas.len() - 1)];

    // Normalise by median pairwise distance.
    let mut all_dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            all_dists.push(distances[i * n + j]);
        }
    }
    all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_d = all_dists[all_dists.len() / 2];

    if median_d < 1e-12 {
        return 0.0;
    }

    delta_90 / median_d
}
