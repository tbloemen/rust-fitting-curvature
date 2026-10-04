//! Synthetic dataset generators with known intrinsic curvature.
//!
//! Each generator returns `DataPoints` with:
//! - `x`: ambient coordinates (flat row-major, shape n × `ambient_dim`)
//! - `labels`: integer labels (length n)
//! - `distances`: precomputed intrinsic distance matrix (flat n × n)

use crate::cast::{count_to_f64, to_u32, to_usize};
use std::f64::consts::PI;

pub use crate::rng::Rng;

/// Result of a data generator or real dataset loader.
pub struct DataPoints {
    /// Flat row-major coordinates, shape (`n_points`, `ambient_dim`)
    pub x: Vec<f64>,
    pub n_points: usize,
    pub ambient_dim: usize,
    /// Integer labels per point
    pub labels: Vec<u32>,
    /// Precomputed distance matrix (flat n × n, row-major)
    pub distances: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Private helper functions
// ---------------------------------------------------------------------------

/// Compute pairwise Euclidean distances.
fn euclidean_distances(x: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0;
            for k in 0..dim {
                let diff = x[i * dim + k] - x[j * dim + k];
                sq += diff * diff;
            }
            let dist = sq.sqrt();
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }
    d
}

/// Compute pairwise great-circle distances on S^(dim-1).
fn spherical_distances_nd(x: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = (0..dim).map(|k| x[i * dim + k] * x[j * dim + k]).sum();
            let dist = dot.clamp(-1.0, 1.0).acos();
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }
    d
}

/// Compute pairwise hyperboloid distances on H^(dim-1) embedded in R^dim.
/// Lorentzian inner product: -x[0]*y[0] + x[1]*y[1] + ... + x[dim-1]*y[dim-1]
fn hyperboloid_distances_nd(x: &[f64], n: usize, dim: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let inner = -x[i * dim] * x[j * dim]
                + (1..dim)
                    .map(|k| x[i * dim + k] * x[j * dim + k])
                    .sum::<f64>();
            let dist = (-inner).max(1.0).acosh();
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }
    d
}

/// Convert Poincaré ball coordinates (dim `poincare_dim`) to hyperboloid model (dim `poincare_dim+1`).
fn poincare_to_hyperboloid_nd(p: &[f64], n: usize, poincare_dim: usize) -> Vec<f64> {
    let ambient = poincare_dim + 1;
    let mut x = Vec::with_capacity(n * ambient);
    for i in 0..n {
        let sq_norm: f64 = (0..poincare_dim)
            .map(|k| p[i * poincare_dim + k].powi(2))
            .sum();
        let denom = 1.0 - sq_norm;
        x.push((1.0 + sq_norm) / denom);
        for k in 0..poincare_dim {
            x.push(2.0 * p[i * poincare_dim + k] / denom);
        }
    }
    x
}

/// Sample a uniformly random unit vector on S^(dim-1).
fn sample_unit_sphere(rng: &mut Rng, dim: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Lay out tree *levels* as concentric rings in the 2D Poincaré disk.
///
/// This is a radial hyperbolic point distribution keyed by depth, **not** a
/// tree metric: no parent/child relation enters the distances, which are the
/// continuous hyperbolic ones between ring positions.
///
/// Returns (`poincaré_coords` [n×2], labels [n]).
fn poincare_tree_2d(n_samples: usize, rng: &mut Rng) -> (Vec<f64>, Vec<u32>) {
    let max_depth = to_usize(count_to_f64(n_samples).log2().ceil());
    let max_depth = max_depth.max(2);

    let mut poincare = Vec::new();
    let mut labels = Vec::new();

    // Root at origin
    poincare.push(0.0_f64);
    poincare.push(0.0_f64);
    labels.push(0u32);

    'outer: for depth in 1..=max_depth {
        let n_at_depth = 1 << depth; // 2^depth
        let r = (count_to_f64(depth) * 0.8 / 2.0).tanh();
        for i in 0..n_at_depth {
            let angle = 2.0 * PI * f64::from(i) / f64::from(n_at_depth) + count_to_f64(depth) * 0.3;
            poincare.push(r * angle.cos());
            poincare.push(r * angle.sin());
            labels.push(u32::try_from(depth.min(4)).expect("depth is a log2 of n_samples"));
            if labels.len() >= n_samples {
                break 'outer;
            }
        }
    }

    while labels.len() < n_samples {
        let depth = to_usize(rng.uniform() * count_to_f64(max_depth)) + 1;
        let r = (count_to_f64(depth) * 0.8 / 2.0).tanh();
        let angle = rng.uniform() * 2.0 * PI;
        poincare.push(r * angle.cos());
        poincare.push(r * angle.sin());
        labels.push(u32::try_from(depth.min(4)).expect("depth is a log2 of n_samples"));
    }

    poincare.truncate(n_samples * 2);
    labels.truncate(n_samples);

    // Clamp to stay strictly inside disk
    for i in 0..n_samples {
        let p1 = poincare[i * 2];
        let p2 = poincare[i * 2 + 1];
        let norm = (p1 * p1 + p2 * p2).sqrt();
        if norm >= 1.0 {
            poincare[i * 2] = p1 / norm * 0.99;
            poincare[i * 2 + 1] = p2 / norm * 0.99;
        }
    }

    (poincare, labels)
}

// ---------------------------------------------------------------------------
// Euclidean generators
// ---------------------------------------------------------------------------

/// Uniform random samples in [-1,1]^2, labels by quadrant (0-3).
#[must_use]
pub fn generate_uniform_grid(n_samples: usize, seed: u64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * 2);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x0 = rng.uniform() * 2.0 - 1.0;
        let x1 = rng.uniform() * 2.0 - 1.0;
        x.push(x0);
        x.push(x1);
        let label = if x0 >= 0.0 { 2 } else { 0 } + u32::from(x1 >= 0.0);
        labels.push(label);
    }

    let distances = euclidean_distances(&x, n_samples, 2);

    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 2,
        labels,
        distances,
    }
}

// ---------------------------------------------------------------------------
// Spherical generators (D = great-circle distances)
// ---------------------------------------------------------------------------

/// Uniform on S^2 via Marsaglia method, labels by hemisphere (0=south, 1=north).
#[must_use]
pub fn generate_uniform_sphere(n_samples: usize, seed: u64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * 3);
    let mut labels = Vec::with_capacity(n_samples);

    let mut count = 0;
    while count < n_samples {
        let u1 = rng.uniform() * 2.0 - 1.0;
        let u2 = rng.uniform() * 2.0 - 1.0;
        let s = u1 * u1 + u2 * u2;
        if s >= 1.0 {
            continue;
        }
        let sqrt_term = (1.0 - s).sqrt();
        let px = 2.0 * u1 * sqrt_term;
        let py = 2.0 * u2 * sqrt_term;
        let pz = 1.0 - 2.0 * s;

        let norm = (px * px + py * py + pz * pz).sqrt();
        x.push(px / norm);
        x.push(py / norm);
        x.push(pz / norm);
        labels.push(u32::from(pz >= 0.0));
        count += 1;
    }

    let distances = spherical_distances_nd(&x, n_samples, 3);

    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

// ---------------------------------------------------------------------------
// Hyperbolic generators (D = hyperboloid distances)
// ---------------------------------------------------------------------------

/// Proper sinh-weighted radial sampling in Poincaré disk, labels by radius bins.
///
/// `max_rho` controls the sampling radius in the hyperbolic metric.
/// Use `max_rho = 3.0` for t-SNE embedding data; use `max_rho ≥ 5.0` for
/// curvature detection, where longer distances make H² distinguishable from E².
#[must_use]
pub fn generate_uniform_hyperbolic(n_samples: usize, seed: u64, max_rho: f64) -> DataPoints {
    let mut rng = Rng::new(seed);

    let mut poincare = Vec::with_capacity(n_samples * 2);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let u = rng.uniform();
        let cosh_rho = 1.0 + u * (max_rho.cosh() - 1.0);
        let rho = cosh_rho.acosh();

        let poincare_r = (rho / 2.0).tanh();
        let angle = rng.uniform() * 2.0 * PI;

        poincare.push(poincare_r * angle.cos());
        poincare.push(poincare_r * angle.sin());

        let label = to_u32(rho / max_rho * 3.0).min(2);
        labels.push(label);
    }

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, 2);
    let distances = hyperboloid_distances_nd(&x, n_samples, 3);

    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

/// Radial hyperbolic layout by tree level, labels by depth (0-4).
///
/// Despite the name this measures continuous H² distances between points
/// placed on concentric rings — it tests a radial distribution, not hierarchy
/// preservation.
#[must_use]
pub fn generate_tree_structured(n_samples: usize, seed: u64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let (poincare, labels) = poincare_tree_2d(n_samples, &mut rng);

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, 2);
    let distances = hyperboloid_distances_nd(&x, n_samples, 3);

    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

/// Concentric rings at fixed hyperbolic radii, labels by shell (0, 1, 2).
///
/// # Panics
///
/// Panics if `shell_idx` exceeds u32 range.
#[must_use]
pub fn generate_hyperbolic_shells(n_samples: usize, seed: u64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let n_per_shell = n_samples / 3;
    let n_last = n_samples - 2 * n_per_shell;

    let shell_params = [(n_per_shell, 0.5), (n_per_shell, 1.5), (n_last, 2.5)];

    let mut poincare = Vec::with_capacity(n_samples * 2);
    let mut labels = Vec::with_capacity(n_samples);

    for (shell_idx, &(n_pts, rho)) in shell_params.iter().enumerate() {
        let poincare_r = (rho / 2.0_f64).tanh();
        for _ in 0..n_pts {
            let noise = 0.05 * rng.normal();
            let r = (poincare_r + noise).clamp(0.01, 0.99);
            let angle = rng.uniform() * 2.0 * PI;
            poincare.push(r * angle.cos());
            poincare.push(r * angle.sin());
            labels.push(u32::try_from(shell_idx).expect("the number of shells is a small count"));
        }
    }

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, 2);
    let distances = hyperboloid_distances_nd(&x, n_samples, 3);

    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

// ---------------------------------------------------------------------------
// High-dimensional curved geometry generators
//
// These embed curved manifolds in `dim`-dimensional space for use as
// high-dimensional input data to the t-SNE optimizer. With dim=3 they
// reduce to the same manifolds as the generators above.
// ---------------------------------------------------------------------------

/// Uniform random samples in [-1,1]^dim (flat Euclidean reference dataset).
/// Labels by quadrant of the first two coordinates (0-3), matching the 2D
/// generator: using all 2^dim orthants would give one label per handful of
/// points at dim=10 and make the label-based metrics meaningless.
///
/// # Panics
///
/// Panics if `dim < 2`.
#[must_use]
pub fn generate_hd_uniform_grid(n_samples: usize, dim: usize, seed: u64) -> DataPoints {
    assert!(dim >= 2, "dim must be at least 2");
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * dim);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let coords: Vec<f64> = (0..dim).map(|_| rng.uniform() * 2.0 - 1.0).collect();
        let label = if coords[0] >= 0.0 { 2 } else { 0 } + u32::from(coords[1] >= 0.0);
        x.extend_from_slice(&coords);
        labels.push(label);
    }

    let distances = euclidean_distances(&x, n_samples, dim);
    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: dim,
        labels,
        distances,
    }
}

/// Uniform on S^(dim-1): sample dim normals and normalize.
/// Labels by sign of first coordinate (two hemispheres).
///
/// # Panics
///
/// Panics if `dim < 2`.
#[must_use]
pub fn generate_hd_sphere(n_samples: usize, dim: usize, seed: u64) -> DataPoints {
    assert!(dim >= 2, "dim must be at least 2");
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * dim);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let coords = sample_unit_sphere(&mut rng, dim);
        labels.push(u32::from(coords[0] >= 0.0));
        x.extend_from_slice(&coords);
    }

    let distances = spherical_distances_nd(&x, n_samples, dim);
    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: dim,
        labels,
        distances,
    }
}

/// Radial hyperbolic layout by tree level on H^(dim-1), embedded in R^dim.
///
/// Like [`generate_tree_structured`], this is a point distribution rather than
/// a hierarchy. The layout is
/// built in a 2D Poincaré disk and the extra Poincaré dimensions receive
/// `0.05·N(0,1)` noise so the data is non-degenerate in all ambient dimensions;
/// that noise pushes most points outside the unit ball at the deeper levels, so
/// the rescale below fires for the majority of them and the boundary shell it
/// produces is largely an artefact of this generator. Labels by depth (0-4).
///
/// # Panics
///
/// Panics if `dim < 3`.
#[must_use]
pub fn generate_hd_tree(n_samples: usize, dim: usize, seed: u64) -> DataPoints {
    assert!(dim >= 3, "dim must be at least 3 for hd_tree");
    let mut rng = Rng::new(seed);
    let poincare_dim = dim - 1;

    let (poincare2d, labels) = poincare_tree_2d(n_samples, &mut rng);

    // Embed 2D Poincaré disk in (dim-1)-dimensional Poincaré ball.
    // Extra dimensions get small Gaussian noise so the embedding is non-trivial.
    let noise_scale = 0.05;
    let mut poincare = Vec::with_capacity(n_samples * poincare_dim);
    for i in 0..n_samples {
        poincare.push(poincare2d[i * 2]);
        poincare.push(poincare2d[i * 2 + 1]);
        for _ in 2..poincare_dim {
            poincare.push(rng.normal() * noise_scale);
        }
        // Ensure the point is strictly inside the Poincaré ball
        let norm_sq: f64 = poincare[i * poincare_dim..(i + 1) * poincare_dim]
            .iter()
            .map(|v| v * v)
            .sum();
        if norm_sq >= 1.0 {
            let norm = norm_sq.sqrt();
            for k in 0..poincare_dim {
                poincare[i * poincare_dim + k] /= norm * (1.0 / 0.99);
            }
        }
    }

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, poincare_dim);
    let distances = hyperboloid_distances_nd(&x, n_samples, dim);
    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: dim,
        labels,
        distances,
    }
}

/// Concentric hyperbolic shells in H^(dim-1) embedded in R^dim.
/// Each shell is a (dim-2)-sphere in the Poincaré ball at a fixed hyperbolic radius.
/// Labels by shell (0, 1, 2).
///
/// # Panics
///
/// Panics if `dim < 3` or `shell_idx` exceeds u32.
#[must_use]
pub fn generate_hd_hyperbolic_shells(n_samples: usize, dim: usize, seed: u64) -> DataPoints {
    assert!(dim >= 3, "dim must be at least 3 for hd_hyperbolic_shells");
    let mut rng = Rng::new(seed);
    let poincare_dim = dim - 1;

    let n_per_shell = n_samples / 3;
    let n_last = n_samples - 2 * n_per_shell;
    let shell_params = [(n_per_shell, 0.5_f64), (n_per_shell, 1.5), (n_last, 2.5)];

    let mut poincare = Vec::with_capacity(n_samples * poincare_dim);
    let mut labels = Vec::with_capacity(n_samples);

    for (shell_idx, &(n_pts, rho)) in shell_params.iter().enumerate() {
        let poincare_r = (rho / 2.0).tanh();
        for _ in 0..n_pts {
            let noise = 0.05 * rng.normal();
            let r = (poincare_r + noise).clamp(0.01, 0.99);
            // Sample direction uniformly on S^(poincare_dim-1)
            let dir = sample_unit_sphere(&mut rng, poincare_dim);
            for dir_k in dir.iter().take(poincare_dim) {
                poincare.push(r * dir_k);
            }
            labels.push(u32::try_from(shell_idx).expect("the number of shells is a small count"));
        }
    }

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, poincare_dim);
    let distances = hyperboloid_distances_nd(&x, n_samples, dim);
    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: dim,
        labels,
        distances,
    }
}

// ---------------------------------------------------------------------------
// Euclidean balls (flat controls for curvature detection)
// ---------------------------------------------------------------------------

/// Uniform random samples inside a 2D ball of the given radius (Euclidean plane).
///
/// Use `radius ≈ 3` to match the natural scale of H² and S² generators.
#[must_use]
pub fn generate_uniform_ball_2d(n_samples: usize, seed: u64, radius: f64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * 2);
    let mut count = 0;
    while count < n_samples {
        let x0 = (rng.uniform() * 2.0 - 1.0) * radius;
        let x1 = (rng.uniform() * 2.0 - 1.0) * radius;
        if x0 * x0 + x1 * x1 <= radius * radius {
            x.push(x0);
            x.push(x1);
            count += 1;
        }
    }
    let distances = euclidean_distances(&x, n_samples, 2);
    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 2,
        labels: vec![0; n_samples],
        distances,
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Ambient dimension of the `hd_*` generators in [`load_synthetic`]: the
/// sweep datasets `optimizer::Dataset::load_synthetic` builds (as `sphere`,
/// `tree`, `hyperbolic_shells`, `grid`) live in R^10.
pub const HD_AMBIENT_DIM: usize = 10;

/// Available synthetic dataset names (for the frontend generators).
///
/// The `hd_*` entries are the exact datasets the optimizer sweeps run on
/// (see [`HD_AMBIENT_DIM`]); the unprefixed 2-D generators are the toy
/// versions that embed trivially.
pub const DATASET_NAMES: &[&str] = &[
    "uniform_grid",
    "uniform_sphere",
    "tree_structured",
    "hyperbolic_shells",
    "hd_uniform_grid",
    "hd_sphere",
    "hd_tree",
    "hd_hyperbolic_shells",
];

/// Load a synthetic dataset by name (frontend generators).
///
/// # Errors
///
/// Returns `Err` for unknown dataset name.
pub fn load_synthetic(name: &str, n_samples: usize, seed: u64) -> Result<DataPoints, String> {
    match name {
        // The sweep datasets, same generator and ambient dimension as
        // `optimizer::Dataset::load_synthetic` uses under the unprefixed name.
        "hd_uniform_grid" => Ok(generate_hd_uniform_grid(n_samples, HD_AMBIENT_DIM, seed)),
        "hd_sphere" => Ok(generate_hd_sphere(n_samples, HD_AMBIENT_DIM, seed)),
        "hd_tree" => Ok(generate_hd_tree(n_samples, HD_AMBIENT_DIM, seed)),
        "hd_hyperbolic_shells" => Ok(generate_hd_hyperbolic_shells(
            n_samples,
            HD_AMBIENT_DIM,
            seed,
        )),
        "uniform_grid" => Ok(generate_uniform_grid(n_samples, seed)),
        "uniform_sphere" => Ok(generate_uniform_sphere(n_samples, seed)),
        "tree_structured" => Ok(generate_tree_structured(n_samples, seed)),
        "hyperbolic_shells" => Ok(generate_hyperbolic_shells(n_samples, seed)),
        _ => Err(format!(
            "Unknown synthetic dataset: {name}. Available: {DATASET_NAMES:?}"
        )),
    }
}
