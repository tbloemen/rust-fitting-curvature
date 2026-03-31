//! Geometry detection from pairwise distances.
//!
//! Given a distance matrix, the algorithm detects whether the underlying
//! geometry is Euclidean, spherical, or hyperbolic and estimates the intrinsic
//! dimension d.
//!
//! **Mathematical basis.** In a d-dimensional Riemannian manifold of constant
//! curvature, the surface area of a geodesic sphere of radius r is proportional
//! to:
//!
//! | Geometry   | Surface area            |
//! |------------|-------------------------|
//! | Euclidean  | r^(d−1)                |
//! | Spherical  | sin(r)^(d−1)           |
//! | Hyperbolic | sinh(r)^(d−1)          |
//!
//! **Algorithm.**
//! 1. For each point, collect its K nearest-neighbour distances.
//! 2. Build a shell density histogram, truncated at the density peak
//!    (to exclude boundary-clipping artefacts).
//! 3. For each model, fit log density = (d−1) · log f(r) + C via OLS.
//! 4. R² selects between Euclidean and spherical.  For Euclidean vs
//!    hyperbolic, a residual-curvature test detects whether the density
//!    grows faster than any power law (the hallmark of exponential/sinh
//!    growth).

/// Result of fitting one geometry model.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Estimated intrinsic dimension (slope + 1).
    pub dim: f64,
    /// Coefficient of determination R² ∈ [0, 1].
    pub r_squared: f64,
    /// Intercept in the log-space regression (log of scale constant).
    pub log_scale: f64,
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

/// Compute the empirical shell density profile using K nearest neighbours.
///
/// The colleague's algorithm: "Pick a point and start growing a ball; note
/// at which radius you collect the 1st, 2nd, …, nth neighbour."
///
/// Returns `(bin_centers, density)`.  The density integrates to ~1 so it is
/// comparable across datasets with different point counts.
///
/// * `distances`    — flat row-major n×n distance matrix.
/// * `n_points`     — n.
/// * `n_bins`       — histogram resolution (30–50 recommended).
/// * `k_neighbors`  — how many nearest neighbours per reference point
///   (0 ⇒ auto: n/4).
pub fn shell_density_profile(
    distances: &[f64],
    n_points: usize,
    n_bins: usize,
    k_neighbors: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_points < 3 {
        return (vec![0.0; n_bins], vec![0.0; n_bins]);
    }

    // Use the most central ~10% of points as references (smallest mean
    // distance to all others).  Central points have good boundary clearance,
    // so their neighbour-distance profile extends to the large-r region
    // where curvature signals are strongest.  We use ALL of each ref's
    // neighbours (no KNN cap) and rely on peak-truncation to handle
    // boundary clipping.
    let n_ref = if k_neighbors != 0 {
        n_points // caller-specified K: use all points
    } else {
        (n_points / 10).max(3)
    };

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

    let k = if k_neighbors == 0 {
        n_points - 1 // all neighbours
    } else {
        k_neighbors.min(n_points - 1)
    };

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
        return (vec![0.0; n_bins], vec![0.0; n_bins]);
    }

    // Build a preliminary histogram using 95th-percentile as r_max.
    local_dists.sort_by(|a: &f64, b| a.partial_cmp(b).unwrap());
    let idx95 = ((local_dists.len() as f64) * 0.95) as usize;
    let r_raw = local_dists[idx95.min(local_dists.len() - 1)];

    if r_raw < 1e-12 {
        return (vec![0.0; n_bins], vec![0.0; n_bins]);
    }

    let bin_width_raw = r_raw / n_bins as f64;
    let mut counts_raw = vec![0.0f64; n_bins];
    for &d in &local_dists {
        if d > 1e-12 && d < r_raw {
            let bin = ((d / r_raw) * n_bins as f64) as usize;
            counts_raw[bin.min(n_bins - 1)] += 1.0;
        }
    }

    // Find the peak of the density profile using a 3-bin moving average.
    // In a bounded sample the peak marks the onset of boundary clipping.
    let smoothed: Vec<f64> = (0..n_bins)
        .map(|i| {
            let lo = if i > 0 { i - 1 } else { 0 };
            let hi = (i + 1).min(n_bins - 1);
            counts_raw[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64
        })
        .collect();

    let peak_bin = smoothed
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(n_bins - 1);

    // Truncate at the peak to use only the increasing part.
    let r_max = (peak_bin as f64 + 1.0) * bin_width_raw;

    if r_max < 1e-12 {
        return (vec![0.0; n_bins], vec![0.0; n_bins]);
    }

    // Re-bin with the truncated r_max.
    let bin_width = r_max / n_bins as f64;
    let mut counts = vec![0.0f64; n_bins];

    for &d in &local_dists {
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
fn fit_model(
    r_transform: impl Fn(f64) -> Option<f64>,
    r_vals: &[f64],
    log_density: &[f64],
) -> FitResult {
    let pairs: Vec<(f64, f64)> = r_vals
        .iter()
        .zip(log_density)
        .filter_map(|(&r, &ld)| r_transform(r).map(|tx| (tx, ld)))
        .collect();

    if pairs.len() < 3 {
        return FitResult {
            dim: 1.0,
            r_squared: 0.0,
            log_scale: 0.0,
        };
    }

    let xs: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
    let ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
    let (slope, intercept, r2) = ols(&xs, &ys);

    FitResult {
        dim: slope + 1.0,
        r_squared: r2.max(0.0),
        log_scale: intercept,
    }
}

/// Detect the underlying geometry and estimate the intrinsic dimension.
///
/// * `distances`    — flat row-major n×n distance matrix.
/// * `n_points`     — n.
/// * `n_bins`       — histogram resolution (30–50 works well).
/// * `k_neighbors`  — nearest neighbours per reference (0 ⇒ auto: n/4).
pub fn detect_geometry(
    distances: &[f64],
    n_points: usize,
    n_bins: usize,
    k_neighbors: usize,
) -> GeometryDetection {
    let (r_vals, density) = shell_density_profile(distances, n_points, n_bins, k_neighbors);

    let (r_filt, log_d_filt): (Vec<f64>, Vec<f64>) = r_vals
        .iter()
        .copied()
        .zip(density.iter().copied())
        .filter(|&(r, d)| r > 1e-10 && d > 1e-10)
        .map(|(r, d)| (r, d.ln()))
        .unzip();

    if r_filt.len() < 3 {
        let zero = FitResult {
            dim: 1.0,
            r_squared: 0.0,
            log_scale: 0.0,
        };
        return GeometryDetection {
            euclidean: zero.clone(),
            spherical: zero.clone(),
            hyperbolic: zero.clone(),
            best_geometry: "unknown",
        };
    }

    // Euclidean: log ρ ~ (d−1) log r
    let euclidean = fit_model(
        |r| if r > 1e-10 { Some(r.ln()) } else { None },
        &r_filt,
        &log_d_filt,
    );

    // Spherical: log ρ ~ (d−1) log sin r  (only for r < π)
    let spherical = fit_model(
        |r| {
            let s = r.sin();
            if r < std::f64::consts::PI && s > 1e-10 {
                Some(s.ln())
            } else {
                None
            }
        },
        &r_filt,
        &log_d_filt,
    );

    // Hyperbolic: log ρ ~ (d−1) log sinh r
    let hyperbolic = fit_model(
        |r| {
            let s = r.sinh();
            if s > 1e-10 { Some(s.ln()) } else { None }
        },
        &r_filt,
        &log_d_filt,
    );

    // --- Model selection ---
    // R² works well for spherical vs others.  For Euclidean vs hyperbolic,
    // log-space R² often fails because the Euclidean model adjusts its
    // exponent to approximate moderate sinh growth.  We supplement with
    // the Gromov 4-point hyperbolicity: in hyperbolic spaces the
    // normalised δ is small (bounded), while in Euclidean/spherical
    // spaces it grows with the sample diameter.
    let gromov = gromov_hyperbolicity(distances, n_points, 5000);

    let best_geometry = if hyperbolic.r_squared > euclidean.r_squared
        && hyperbolic.r_squared > spherical.r_squared
    {
        "hyperbolic"
    } else if spherical.r_squared > euclidean.r_squared
        && spherical.r_squared > hyperbolic.r_squared
    {
        "spherical"
    } else if gromov < 0.15 {
        // Low Gromov δ → metric space is tree-like → hyperbolic.
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
fn gromov_hyperbolicity(distances: &[f64], n: usize, n_samples: usize) -> f64 {
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
