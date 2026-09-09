//! Synthetic dataset generators with known intrinsic curvature.
//!
//! Each generator returns `DataPoints` with:
//! - `x`: ambient coordinates (flat row-major, shape n × `ambient_dim`)
//! - `labels`: integer labels (length n)
//! - `distances`: precomputed intrinsic distance matrix (flat n × n)

use crate::cast::{count_to_f64, to_u32, to_usize};
use crate::graph::{all_pairs_bfs_distances, RootedTree};
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
/// continuous hyperbolic ones between ring positions. For the hierarchy
/// itself see [`generate_tree_graph`].
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

/// N(0, I) in R^2, labels by median radius (0=inner, 1=outer).
///
/// # Panics
///
/// Panics if `n_samples == 0` or if any radius is NaN.
#[must_use]
pub fn generate_gaussian_blob(n_samples: usize, seed: u64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * 2);
    let mut radii = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x0 = rng.normal();
        let x1 = rng.normal();
        x.push(x0);
        x.push(x1);
        radii.push((x0 * x0 + x1 * x1).sqrt());
    }

    // Find median radius
    let mut sorted_radii = radii.clone();
    sorted_radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_radii[n_samples / 2];

    let labels: Vec<u32> = radii.iter().map(|&r| u32::from(r >= median)).collect();

    let distances = euclidean_distances(&x, n_samples, 2);

    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 2,
        labels,
        distances,
    }
}

/// Two concentric rings at r=1, r=2 with noise, labels by ring (0, 1).
#[must_use]
pub fn generate_concentric_circles(n_samples: usize, seed: u64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let n_inner = n_samples / 2;
    let n_outer = n_samples - n_inner;

    let mut x = Vec::with_capacity(n_samples * 2);
    let mut labels = Vec::with_capacity(n_samples);

    // Inner ring
    for _ in 0..n_inner {
        let angle = rng.uniform() * 2.0 * PI;
        let r = 1.0 + 0.1 * rng.normal();
        x.push(r * angle.cos());
        x.push(r * angle.sin());
        labels.push(0);
    }

    // Outer ring
    for _ in 0..n_outer {
        let angle = rng.uniform() * 2.0 * PI;
        let r = 2.0 + 0.1 * rng.normal();
        x.push(r * angle.cos());
        x.push(r * angle.sin());
        labels.push(1);
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

/// Von Mises-Fisher distribution (kappa=10) around north pole.
///
/// # Panics
///
/// Panics if `n_samples == 0` or if any distance is NaN.
#[expect(
    clippy::many_single_char_names,
    reason = "m,b,c,z,w,u,x,y are the standard Wood (1994) vMF rejection sampling variables: m=dim-1, b=rejection bound, c=acceptance threshold"
)]
#[must_use]
pub fn generate_von_mises_fisher(n_samples: usize, seed: u64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let kappa = 10.0_f64;

    // Wood (1994) rejection sampling for vMF on S^2
    let m = 2.0_f64; // dim - 1
    let b = (-2.0 * kappa + (4.0 * kappa * kappa + m * m).sqrt()) / m;
    let x0 = (1.0 - b) / (1.0 + b);
    let c = kappa * x0 + m * (1.0 - x0 * x0).ln();

    let mut points = Vec::with_capacity(n_samples * 3);
    let mut labels = Vec::with_capacity(n_samples);

    let mut count = 0;
    while count < n_samples {
        // Sample w from marginal using rejection
        let z = rng.uniform(); // Beta(1,1) = uniform for m=2
        let w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);
        let u = rng.uniform();
        if (kappa * w + m * (1.0 - x0 * w).ln() - c) < u.ln() {
            continue;
        }

        // Sample direction on S^1
        let angle = rng.uniform() * 2.0 * PI;
        let sqrt_term = (1.0 - w * w).max(0.0).sqrt();

        let px = sqrt_term * angle.cos();
        let py = sqrt_term * angle.sin();
        let pz = w;

        let norm = (px * px + py * py + pz * pz).sqrt();
        points.push(px / norm);
        points.push(py / norm);
        points.push(pz / norm);

        labels.push(u32::from(count >= n_samples / 2));
        count += 1;
    }

    // Relabel by median distance from north pole
    let dists: Vec<f64> = (0..n_samples)
        .map(|i| points[i * 3 + 2].clamp(-1.0, 1.0).acos())
        .collect();
    let mut sorted_dists = dists.clone();
    sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_dists[n_samples / 2];
    labels = dists.iter().map(|&d| u32::from(d >= median)).collect();

    let distances = spherical_distances_nd(&points, n_samples, 3);

    DataPoints {
        x: points,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

/// Two vMF clusters at north and south poles (kappa=10), labels by cluster.
#[must_use]
pub fn generate_antipodal_clusters(n_samples: usize, seed: u64) -> DataPoints {
    let n_north = n_samples / 2;
    let n_south = n_samples - n_north;

    let north = generate_von_mises_fisher(n_north, seed);
    let south_raw = generate_von_mises_fisher(n_south, seed.wrapping_add(1));

    // Flip south points: negate z coordinate
    let mut x = north.x.clone();
    for i in 0..n_south {
        x.push(south_raw.x[i * 3]);
        x.push(south_raw.x[i * 3 + 1]);
        x.push(-south_raw.x[i * 3 + 2]);
    }

    let mut labels = vec![0u32; n_north];
    labels.extend(vec![1u32; n_south]);

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
/// preservation. [`generate_tree_graph`] is the tree-metric benchmark.
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

/// Two concentrated clusters at antipodal poles on S^(dim-1).
/// Uses shift-and-normalize: add κ * `pole_direction` to a random normal, then normalize.
/// Labels by cluster (0=north, 1=south).
///
/// # Panics
///
/// Panics if `dim < 2`.
#[must_use]
pub fn generate_hd_antipodal_clusters(n_samples: usize, dim: usize, seed: u64) -> DataPoints {
    assert!(dim >= 2, "dim must be at least 2");
    let mut rng = Rng::new(seed);
    let kappa = 5.0_f64; // concentration toward poles
    let n_north = n_samples / 2;
    let mut x = Vec::with_capacity(n_samples * dim);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let pole_sign = if i < n_north { 1.0 } else { -1.0 };
        let mut coords: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
        coords[0] += kappa * pole_sign; // shift first coordinate toward pole
        let norm = coords.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);
        for v in &mut coords {
            *v /= norm;
        }
        x.extend_from_slice(&coords);
        labels.push(u32::from(i >= n_north));
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
/// a hierarchy — see [`generate_tree_graph`] for the tree metric. The layout is
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
// Hierarchy benchmark: a real tree metric
// ---------------------------------------------------------------------------

/// Rooted `branching`-ary tree carrying its unweighted shortest-path metric.
///
/// Unlike [`generate_tree_structured`], the distances here *are* the hierarchy:
/// `d(u,v) = depth(u) + depth(v) − 2·depth(lca(u,v))`, the number of edges
/// between two nodes. No manifold is involved and no Riemannian curvature is
/// claimed — a tree is not a constant-curvature space. Hyperbolic space is a
/// candidate *representation* of this metric, which is the same footing
/// `wordnet_mammals` sits on.
///
/// `detect_geometry` calls this hyperbolic, as it should: measured at n = 1000
/// the δ(k) tail slope is 0.000 — a tree metric is exactly 0-hyperbolic. Two
/// caveats on the numbers beside that verdict. Its `κ_data` reads `0.0100`,
/// which is `HYPERBOLIC_KAPPA_MIN` exactly and so is the search window's bound
/// rather than a measurement; and both Wilson arms fit badly (residual ~2e-1),
/// because a tree is not a constant-curvature manifold and neither model
/// describes it. The verdict is sound; the curvature magnitude is not a reading.
///
/// Shaped like the `WordNet` loader: empty `x`, `ambient_dim: 0`, populated
/// `distances` and `labels`. Labels are branch membership at depth
/// `label_level` (see [`RootedTree::branch_labels`]); depth is deliberately not
/// folded into them and is recoverable with
/// `RootedTree::complete(n, branching).depths()`.
///
/// # Panics
///
/// Panics if `n_samples == 0` or `branching < 2`.
#[must_use]
pub fn generate_tree_graph(n_samples: usize, branching: usize, label_level: u32) -> DataPoints {
    let tree = RootedTree::complete(n_samples, branching);
    let distances = all_pairs_bfs_distances(&tree.adjacency(), n_samples);
    let labels = tree.branch_labels(label_level);

    DataPoints {
        // No feature representation: the tree metric drives affinities and
        // evaluation through `distances`, exactly as for `wordnet_mammals`.
        x: Vec::new(),
        n_points: n_samples,
        ambient_dim: 0,
        labels,
        distances,
    }
}

// ---------------------------------------------------------------------------
// Matched geodesic balls: one sampling scheme, three geometries
// ---------------------------------------------------------------------------

/// The shared draw behind [`generate_matched_ball`]: radial quantiles and unit
/// directions, in one fixed order so every curvature sign consumes the same
/// random stream.
fn matched_ball_draw(n_samples: usize, m: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut u = Vec::with_capacity(n_samples);
    let mut dirs = Vec::with_capacity(n_samples * m);
    for _ in 0..n_samples {
        u.push(rng.uniform());
        dirs.extend_from_slice(&sample_unit_sphere(&mut rng, m));
    }
    (u, dirs)
}

/// A geodesic ball of radius `extent` in the constant-curvature space of the
/// given sign, sampled so that **geometry is the only difference** between the
/// three signs.
///
/// Given the same `seed` and `intrinsic_dim`, all three signs draw the same
/// radial quantiles `u_i` and the same unit directions `d_i` — one `Rng`, one
/// draw order — and then map them into their own manifold. The radial law is
/// `r_i = extent · u_i^(1/m)`, the Euclidean uniform-in-ball law, used
/// unchanged in all three. These are *matched sampling distributions*, and
/// deliberately **not** uniform with respect to each manifold's volume:
/// sampling each manifold uniformly would put a distributional difference back
/// alongside the geometric one, which is the confound this family exists to
/// remove.
///
/// Curvature is fixed at `+1 / 0 / −1`, so `extent` is directly in
/// curvature-radius units and means the same thing in all three.
///
/// Coordinate layout, with `m = intrinsic_dim`:
///
/// | sign | ambient | coordinates                       | origin        |
/// |------|---------|-----------------------------------|---------------|
/// | `0`  | `m`     | `r·d`                             | `0`           |
/// | `+1` | `m+1`   | `(sin r · d, cos r)` (pole last)  | `e_m`         |
/// | `−1` | `m+1`   | `(cosh r, sinh r · d)` (time first) | `(1, 0, …)` |
///
/// Each satisfies `d(origin, x_i) = r_i` exactly. The pole-last convention on
/// the sphere matches what `embedding::lift_pca_to_manifold` produces for a
/// fitted embedding; time-first on the hyperboloid matches
/// [`poincare_to_hyperboloid_nd`] and the `Hyperboloid` manifold.
///
/// Labels combine the direction sector with the radial band, so they are
/// bit-identical across the three signs.
///
/// # Panics
///
/// Panics if `intrinsic_dim < 2`, if `extent <= 0`, or if `curvature_sign > 0`
/// and `extent > PI` (past the antipode the ball stops being a ball).
#[expect(
    clippy::many_single_char_names,
    reason = "m = intrinsic dimension, u = radial quantile, d = unit direction, r = geodesic radius, x = ambient coordinates: the notation of the table above"
)]
#[must_use]
pub fn generate_matched_ball(
    n_samples: usize,
    intrinsic_dim: usize,
    curvature_sign: f64,
    extent: f64,
    seed: u64,
) -> DataPoints {
    assert!(intrinsic_dim >= 2, "intrinsic_dim must be at least 2");
    assert!(extent > 0.0, "extent must be positive");
    assert!(
        curvature_sign <= 0.0 || extent <= PI,
        "a spherical ball cannot reach past the antipode: extent must be <= PI"
    );

    let m = intrinsic_dim;
    let (u, dirs) = matched_ball_draw(n_samples, m, seed);

    let ambient = if curvature_sign == 0.0 { m } else { m + 1 };
    let mut x = Vec::with_capacity(n_samples * ambient);
    let mut labels = Vec::with_capacity(n_samples);

    let inv_m = 1.0 / count_to_f64(m);
    for i in 0..n_samples {
        let d = &dirs[i * m..(i + 1) * m];
        let r = extent * u[i].powf(inv_m);

        if curvature_sign > 0.0 {
            // Sphere: the constrained coordinate goes in slot 0 as `-cos r`, so
            // the ball is centred on the SOUTH pole `(-1, 0, …)` — the origin
            // `Sphere::distances_from_origin` actually measures from.
            let (sin_r, cos_r) = r.sin_cos();
            x.push(-cos_r);
            x.extend(d.iter().map(|dk| sin_r * dk));
        } else if curvature_sign < 0.0 {
            // Hyperboloid upper sheet: time component first.
            x.push(r.cosh());
            let sinh_r = r.sinh();
            x.extend(d.iter().map(|dk| sinh_r * dk));
        } else {
            x.extend(d.iter().map(|dk| r * dk));
        }

        // Four direction sectors x two radial bands. Derived from the shared
        // draw alone, so the three geometries get identical label vectors.
        let sector = 2 * u32::from(d[0] >= 0.0) + u32::from(d[1] >= 0.0);
        let band = 4 * u32::from(u[i] >= 0.5);
        labels.push(band + sector);
    }

    let distances = if curvature_sign > 0.0 {
        spherical_distances_nd(&x, n_samples, ambient)
    } else if curvature_sign < 0.0 {
        hyperboloid_distances_nd(&x, n_samples, ambient)
    } else {
        euclidean_distances(&x, n_samples, ambient)
    };

    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: ambient,
        labels,
        distances,
    }
}

/// The ball radius, in curvature-radius units, the matched family is generated
/// at.
///
/// **This is a compromise, and it is not the value that maximises detectability
/// on both curved arms — no single value does.** Measured with `detect_geometry`
/// / `detect_hyperbolic` at n = 500, m = 2, over extents 1.0 … 6.0:
///
/// | extent | spherical verdict | fitted `r*` | hyperbolic δ tail slope |
/// |--------|-------------------|-------------|-------------------------|
/// | 1.0    | euclidean (pinned)| 0.795       | 0.571                   |
/// | 1.3    | spherical         | 1.000       | 0.407                   |
/// | 1.5    | spherical         | 1.000       | 0.377                   |
/// | 2.5    | spherical         | 1.000       | 0.232                   |
/// | 4.0    | —  (past π)       | —           | 0.085 → hyperbolic      |
/// | 6.0    | —  (past π)       | —           | 0.020 → hyperbolic      |
///
/// The spherical arm needs `extent > 1.25` for the Wilson radius search to clear
/// its flat-ward bound (`d_max/`[`SPHERICAL_ANGULAR_MIN`]-equivalent), and wants
/// `extent < π/2` so that `d_max ≈ 2·extent` stays below π: past that `d_max`
/// saturates at π, the search window's lower edge climbs to `d_max/π ≈ 1`, and
/// the true `r* = 1` ends up sitting on it with a margin of ~0.002. The δ(k)
/// saturation gate, meanwhile, only fires below a slope of 0.15, which needs
/// `extent ≥ 4`. The two windows do not overlap.
///
/// Matching wins, because a shared radius is what the family is *for*: the three
/// arms must differ in curvature and nothing else. So the value is chosen inside
/// the spherical window, and `detect_geometry`'s *verdict* on the hyperbolic arm
/// is then `"euclidean"` — as it already is for `hyperbolic_shells`.
///
/// That verdict understates what the detector actually recovers, and the
/// distinction matters. Measured at n = 1000, extent 1.5, m = 2:
///
/// - `ball2_spherical`: verdict spherical, `r* = 1.000`, residual 1.4e-7.
/// - `ball2_hyperbolic`: verdict euclidean, but the **hyperbolic Wilson arm
///   fits `r* = 1.000` with residual 2.3e-8** — the signature test identifies
///   the curvature radius exactly. Only the δ(k) saturation gate declines
///   (slope 0.396), because a radius-1.5 ball is not tree-like.
/// - `ball2_euclidean`: verdict euclidean, spherical arm pinned, δ slope 0.508.
///
/// So the data *is* exactly H² and the signature residual says so to eight
/// digits; `detect_geometry` gates hyperbolicity on δ-saturation rather than on
/// that residual, and saturation needs a radius the sphere cannot match. At this
/// extent κ = |K|·`R_rms`² ≈ 1.1 (m = 2), well above every real dataset in the
/// thesis. Detection proper is served by the unmatched controls
/// (`generate_uniform_hyperbolic(n, seed, 5.0)`), which are free to use a radius
/// the sphere cannot reach.
///
/// Do not raise this to satisfy the hyperbolic gate: it would move all three
/// arms and break the match.
///
/// # The 9-D tier is not a detection fixture, for two independent reasons
///
/// Measured at n = 1000, extent 1.5, m = 9 (and reproduced at n = 500 across
/// every extent from 1.0 to 6.0):
///
/// | dataset | verdict | sph `r*` | sph residual | δ slope |
/// |---------|---------|----------|--------------|---------|
/// | `ball9_euclidean`  | **hyperbolic** | 1.169 (pinned) | 1.4e-1 | 0.100 |
/// | `ball9_spherical`  | euclidean      | 0.988          | 8.6e-2 | 0.195 |
/// | `ball9_hyperbolic` | hyperbolic     | 1.177 (pinned) | 2.0e-1 | 0.047 |
///
/// The flat ball, which has exactly zero curvature by construction, is called
/// hyperbolic. Two separate things break, and neither is a defect in the
/// generator:
///
/// 1. **The signature arms fit a 2-dimensional model.** `detect_geometry` is
///    called with `dim = 2`, so a 9-dimensional sample cannot conform to it
///    whatever its curvature: `ball9_spherical`'s radius search lands on
///    `r* = 0.988`, within 1.2% of the truth, yet its residual is 8.6e-2 —
///    almost two orders of magnitude above the threshold a spherical verdict
///    needs. The radius is right and the model is still rejected.
/// 2. **The δ(k) gate false-positives under concentration.** Uniform directions
///    on S⁸ pull the pairwise distances together: the coefficient of variation
///    falls from 0.47 at m = 2 to 0.19 at m = 9. A δ curve that is flat because
///    every distance is nearly equal is indistinguishable, to the saturation
///    test, from one that is flat because the space is tree-like. (`grid`
///    escapes this only because a *cube* has corners, keeping its spread wide
///    enough for a tail slope of 0.53.)
///
/// Both hyperbolic verdicts above also report `κ_data = 0.0100`, which is
/// `HYPERBOLIC_KAPPA_MIN` exactly — the flat-ward edge of the search window,
/// i.e. a bound rather than a measurement.
///
/// The consequence is a scope limit, not a fix: the 9-D tier is a
/// **dimension-reduction** fixture for the geometry-matching experiment, where
/// the truth is known by construction, and it must not be read as a curvature
/// detection benchmark. The 2-D tier is the one whose geometry the detector
/// recovers.
pub const MATCHED_BALL_EXTENT: f64 = 1.5;

// ---------------------------------------------------------------------------
// Higher-dimensional generators (for curvature detection experiments)
// ---------------------------------------------------------------------------

/// Geodesic distances on the unit (d-1)-sphere embedded in R^d.
/// `x` is flat row-major, shape n × d.
fn sphere_distances(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = (0..d).map(|k| x[i * d + k] * x[j * d + k]).sum();
            let v = dot.clamp(-1.0, 1.0).acos();
            dist[i * n + j] = v;
            dist[j * n + i] = v;
        }
    }
    dist
}

/// Geodesic distances on the hyperboloid model H^d in R^{d+1}.
/// `x` is flat row-major, shape n × (d+1).
/// Lorentzian inner product: −x₀y₀ + x₁y₁ + … + xᵈyᵈ.
fn hyperboloid_distances_generic(x: &[f64], n: usize, ambient_dim: usize) -> Vec<f64> {
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let inner = -x[i * ambient_dim] * x[j * ambient_dim]
                + (1..ambient_dim)
                    .map(|k| x[i * ambient_dim + k] * x[j * ambient_dim + k])
                    .sum::<f64>();
            let v = (-inner).max(1.0).acosh();
            dist[i * n + j] = v;
            dist[j * n + i] = v;
        }
    }
    dist
}

/// Convert d-dimensional Poincaré ball coordinates to hyperboloid model in R^{d+1}.
/// `p` is flat row-major, shape n × d.
fn poincare_to_hyperboloid_generic(p: &[f64], n: usize, d: usize) -> Vec<f64> {
    let ambient = d + 1;
    let mut x = Vec::with_capacity(n * ambient);
    for i in 0..n {
        let sq_norm: f64 = (0..d).map(|k| p[i * d + k].powi(2)).sum();
        let denom = 1.0 - sq_norm;
        x.push((1.0 + sq_norm) / denom);
        for k in 0..d {
            x.push(2.0 * p[i * d + k] / denom);
        }
    }
    x
}

/// Uniform random samples inside a 2D ball of the given radius (Euclidean plane).
///
/// Use `radius ≈ 3` to match the natural scale of H² and S² generators,
/// which is important for curvature detection based on the density profile.
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

/// Uniform random samples inside a 3D ball of the given radius (Euclidean 3-space).
///
/// Use `radius ≈ 3` to match the natural scale of H³ and S³ generators.
#[must_use]
pub fn generate_uniform_ball_3d(n_samples: usize, seed: u64, radius: f64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * 3);
    let mut count = 0;
    while count < n_samples {
        let x0 = (rng.uniform() * 2.0 - 1.0) * radius;
        let x1 = (rng.uniform() * 2.0 - 1.0) * radius;
        let x2 = (rng.uniform() * 2.0 - 1.0) * radius;
        if x0 * x0 + x1 * x1 + x2 * x2 <= radius * radius {
            x.push(x0);
            x.push(x1);
            x.push(x2);
            count += 1;
        }
    }
    let distances = euclidean_distances(&x, n_samples, 3);
    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels: vec![0; n_samples],
        distances,
    }
}

/// Uniform random samples on the unit 3-sphere S³ ⊂ R⁴.
/// Distances are geodesic (great-circle) distances.
#[must_use]
pub fn generate_uniform_sphere3(n_samples: usize, seed: u64) -> DataPoints {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * 4);
    for _ in 0..n_samples {
        let components: [f64; 4] = [rng.normal(), rng.normal(), rng.normal(), rng.normal()];
        let norm = components
            .iter()
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt()
            .max(1e-15);
        for c in &components {
            x.push(c / norm);
        }
    }
    let distances = sphere_distances(&x, n_samples, 4);
    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 4,
        labels: vec![0; n_samples],
        distances,
    }
}

/// Inverse-CDF for the radial distribution on H³: CDF ∝ sinh(r)cosh(r) − r.
fn h3_inverse_cdf(u: f64, max_r: f64) -> f64 {
    let cdf_max = max_r.sinh() * max_r.cosh() - max_r;
    let target = u * cdf_max;
    let mut lo = 0.0f64;
    let mut hi = max_r;
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        if mid.sinh() * mid.cosh() - mid < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Uniform random samples in a hyperbolic ball of the given radius in H³.
/// Stored in the hyperboloid model in R⁴; distances are geodesic.
///
/// Use `max_r ≥ 5.0` for curvature detection experiments.
#[must_use]
pub fn generate_uniform_hyperbolic3(n_samples: usize, seed: u64, max_r: f64) -> DataPoints {
    let mut rng = Rng::new(seed);

    let mut poincare = Vec::with_capacity(n_samples * 3);

    for _ in 0..n_samples {
        let r = h3_inverse_cdf(rng.uniform(), max_r);
        // Uniform direction on S²: cos θ uniform in [−1, 1]
        let cos_theta = rng.uniform() * 2.0 - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = rng.uniform() * 2.0 * PI;
        let poincare_r = (r / 2.0).tanh();
        poincare.push(poincare_r * sin_theta * phi.cos());
        poincare.push(poincare_r * sin_theta * phi.sin());
        poincare.push(poincare_r * cos_theta);
    }

    let x = poincare_to_hyperboloid_generic(&poincare, n_samples, 3);
    let distances = hyperboloid_distances_generic(&x, n_samples, 4);
    DataPoints {
        x,
        n_points: n_samples,
        ambient_dim: 4,
        labels: vec![0; n_samples],
        distances,
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Available synthetic dataset names (for the frontend/2D generators).
pub const DATASET_NAMES: &[&str] = &[
    "uniform_grid",
    "gaussian_blob",
    "concentric_circles",
    "uniform_sphere",
    "von_mises_fisher",
    "antipodal_clusters",
    "uniform_hyperbolic",
    "tree_structured",
    "hyperbolic_shells",
    "tree_graph",
    "ball2_euclidean",
    "ball2_spherical",
    "ball2_hyperbolic",
    "ball9_euclidean",
    "ball9_spherical",
    "ball9_hyperbolic",
];

/// Load a synthetic dataset by name (2D/3D frontend generators).
///
/// # Errors
///
/// Returns `Err` for unknown dataset name.
pub fn load_synthetic(name: &str, n_samples: usize, seed: u64) -> Result<DataPoints, String> {
    match name {
        "uniform_grid" => Ok(generate_uniform_grid(n_samples, seed)),
        "gaussian_blob" => Ok(generate_gaussian_blob(n_samples, seed)),
        "concentric_circles" => Ok(generate_concentric_circles(n_samples, seed)),
        "uniform_sphere" => Ok(generate_uniform_sphere(n_samples, seed)),
        "von_mises_fisher" => Ok(generate_von_mises_fisher(n_samples, seed)),
        "antipodal_clusters" => Ok(generate_antipodal_clusters(n_samples, seed)),
        "uniform_hyperbolic" => Ok(generate_uniform_hyperbolic(n_samples, seed, 3.0)),
        "tree_structured" => Ok(generate_tree_structured(n_samples, seed)),
        "hyperbolic_shells" => Ok(generate_hyperbolic_shells(n_samples, seed)),
        "tree_graph" => Ok(generate_tree_graph(n_samples, 2, 3)),
        "ball2_euclidean" => Ok(generate_matched_ball(
            n_samples,
            2,
            0.0,
            MATCHED_BALL_EXTENT,
            seed,
        )),
        "ball2_spherical" => Ok(generate_matched_ball(
            n_samples,
            2,
            1.0,
            MATCHED_BALL_EXTENT,
            seed,
        )),
        "ball2_hyperbolic" => Ok(generate_matched_ball(
            n_samples,
            2,
            -1.0,
            MATCHED_BALL_EXTENT,
            seed,
        )),
        "ball9_euclidean" => Ok(generate_matched_ball(
            n_samples,
            9,
            0.0,
            MATCHED_BALL_EXTENT,
            seed,
        )),
        "ball9_spherical" => Ok(generate_matched_ball(
            n_samples,
            9,
            1.0,
            MATCHED_BALL_EXTENT,
            seed,
        )),
        "ball9_hyperbolic" => Ok(generate_matched_ball(
            n_samples,
            9,
            -1.0,
            MATCHED_BALL_EXTENT,
            seed,
        )),
        _ => Err(format!(
            "Unknown synthetic dataset: {name}. Available: {DATASET_NAMES:?}"
        )),
    }
}
