//! Synthetic dataset generators with known intrinsic curvature.
//!
//! Each generator returns `SyntheticData` with:
//! - `x`: ambient coordinates (flat row-major, shape n × ambient_dim)
//! - `labels`: integer labels (length n)
//! - `distances`: precomputed intrinsic distance matrix (flat n × n)

use std::f64::consts::PI;

/// Simple seeded PRNG (xoshiro256**) for reproducible data generation.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // SplitMix64 to initialise state from a single seed
        let mut state = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate normal via Box-Muller
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Result of a data generator.
pub struct SyntheticData {
    /// Flat row-major coordinates, shape (n_points, ambient_dim)
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
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// Generate a tree structure in the 2D Poincaré disk.
/// Returns (poincaré_coords [n×2], labels [n]).
fn poincare_tree_2d(n_samples: usize, rng: &mut Rng) -> (Vec<f64>, Vec<u32>) {
    let max_depth = (n_samples as f64).log2().ceil() as usize;
    let max_depth = max_depth.max(2);

    let mut poincare = Vec::new();
    let mut labels = Vec::new();

    // Root at origin
    poincare.push(0.0_f64);
    poincare.push(0.0_f64);
    labels.push(0u32);

    'outer: for depth in 1..=max_depth {
        let n_at_depth = 1 << depth; // 2^depth
        let r = (depth as f64 * 0.8 / 2.0).tanh();
        for i in 0..n_at_depth {
            let angle = 2.0 * PI * i as f64 / n_at_depth as f64 + depth as f64 * 0.3;
            poincare.push(r * angle.cos());
            poincare.push(r * angle.sin());
            labels.push((depth as u32).min(4));
            if labels.len() >= n_samples {
                break 'outer;
            }
        }
    }

    while labels.len() < n_samples {
        let depth = (rng.uniform() * max_depth as f64) as usize + 1;
        let r = (depth as f64 * 0.8 / 2.0).tanh();
        let angle = rng.uniform() * 2.0 * PI;
        poincare.push(r * angle.cos());
        poincare.push(r * angle.sin());
        labels.push((depth as u32).min(4));
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
pub fn generate_uniform_grid(n_samples: usize, seed: u64) -> SyntheticData {
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * 2);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x0 = rng.uniform() * 2.0 - 1.0;
        let x1 = rng.uniform() * 2.0 - 1.0;
        x.push(x0);
        x.push(x1);
        let label = if x0 >= 0.0 { 2 } else { 0 } + if x1 >= 0.0 { 1 } else { 0 };
        labels.push(label);
    }

    let distances = euclidean_distances(&x, n_samples, 2);

    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: 2,
        labels,
        distances,
    }
}

/// N(0, I) in R^2, labels by median radius (0=inner, 1=outer).
pub fn generate_gaussian_blob(n_samples: usize, seed: u64) -> SyntheticData {
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

    let labels: Vec<u32> = radii
        .iter()
        .map(|&r| if r >= median { 1 } else { 0 })
        .collect();

    let distances = euclidean_distances(&x, n_samples, 2);

    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: 2,
        labels,
        distances,
    }
}

/// Two concentric rings at r=1, r=2 with noise, labels by ring (0, 1).
pub fn generate_concentric_circles(n_samples: usize, seed: u64) -> SyntheticData {
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

    SyntheticData {
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
pub fn generate_uniform_sphere(n_samples: usize, seed: u64) -> SyntheticData {
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
        labels.push(if pz >= 0.0 { 1 } else { 0 });
        count += 1;
    }

    let distances = spherical_distances_nd(&x, n_samples, 3);

    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

/// Von Mises-Fisher distribution (kappa=10) around north pole.
pub fn generate_von_mises_fisher(n_samples: usize, seed: u64) -> SyntheticData {
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

        labels.push(if count < n_samples / 2 { 0 } else { 1 });
        count += 1;
    }

    // Relabel by median distance from north pole
    let dists: Vec<f64> = (0..n_samples)
        .map(|i| points[i * 3 + 2].clamp(-1.0, 1.0).acos())
        .collect();
    let mut sorted_dists = dists.clone();
    sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_dists[n_samples / 2];
    labels = dists
        .iter()
        .map(|&d| if d >= median { 1 } else { 0 })
        .collect();

    let distances = spherical_distances_nd(&points, n_samples, 3);

    SyntheticData {
        x: points,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

/// Two vMF clusters at north and south poles (kappa=10), labels by cluster.
pub fn generate_antipodal_clusters(n_samples: usize, seed: u64) -> SyntheticData {
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

    SyntheticData {
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
pub fn generate_uniform_hyperbolic(n_samples: usize, seed: u64, max_rho: f64) -> SyntheticData {
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

        let label = ((rho / max_rho * 3.0) as u32).min(2);
        labels.push(label);
    }

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, 2);
    let distances = hyperboloid_distances_nd(&x, n_samples, 3);

    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

/// Regular branching tree embedded in hyperbolic space, labels by depth.
pub fn generate_tree_structured(n_samples: usize, seed: u64) -> SyntheticData {
    let mut rng = Rng::new(seed);
    let (poincare, labels) = poincare_tree_2d(n_samples, &mut rng);

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, 2);
    let distances = hyperboloid_distances_nd(&x, n_samples, 3);

    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

/// Concentric rings at fixed hyperbolic radii, labels by shell (0, 1, 2).
pub fn generate_hyperbolic_shells(n_samples: usize, seed: u64) -> SyntheticData {
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
            labels.push(shell_idx as u32);
        }
    }

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, 2);
    let distances = hyperboloid_distances_nd(&x, n_samples, 3);

    SyntheticData {
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

/// Uniform on S^(dim-1): sample dim normals and normalize.
/// Labels by sign of first coordinate (two hemispheres).
pub fn generate_hd_sphere(n_samples: usize, dim: usize, seed: u64) -> SyntheticData {
    assert!(dim >= 2, "dim must be at least 2");
    let mut rng = Rng::new(seed);
    let mut x = Vec::with_capacity(n_samples * dim);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let coords = sample_unit_sphere(&mut rng, dim);
        labels.push(if coords[0] >= 0.0 { 1 } else { 0 });
        x.extend_from_slice(&coords);
    }

    let distances = spherical_distances_nd(&x, n_samples, dim);
    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: dim,
        labels,
        distances,
    }
}

/// Two concentrated clusters at antipodal poles on S^(dim-1).
/// Uses shift-and-normalize: add κ * pole_direction to a random normal, then normalize.
/// Labels by cluster (0=north, 1=south).
pub fn generate_hd_antipodal_clusters(n_samples: usize, dim: usize, seed: u64) -> SyntheticData {
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
        labels.push(if i < n_north { 0 } else { 1 });
    }

    let distances = spherical_distances_nd(&x, n_samples, dim);
    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: dim,
        labels,
        distances,
    }
}

/// Branching tree on H^(dim-1) embedded in R^dim.
/// The tree structure is generated in a 2D Poincaré disk; extra Poincaré dimensions
/// receive small noise so the data is non-degenerate in all ambient dimensions.
/// Labels by depth (0-4).
pub fn generate_hd_tree(n_samples: usize, dim: usize, seed: u64) -> SyntheticData {
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
    SyntheticData {
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
pub fn generate_hd_hyperbolic_shells(n_samples: usize, dim: usize, seed: u64) -> SyntheticData {
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
            labels.push(shell_idx as u32);
        }
    }

    let x = poincare_to_hyperboloid_nd(&poincare, n_samples, poincare_dim);
    let distances = hyperboloid_distances_nd(&x, n_samples, dim);
    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: dim,
        labels,
        distances,
    }
}

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
pub fn generate_uniform_ball_2d(n_samples: usize, seed: u64, radius: f64) -> SyntheticData {
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
    SyntheticData {
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
pub fn generate_uniform_ball_3d(n_samples: usize, seed: u64, radius: f64) -> SyntheticData {
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
    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels: vec![0; n_samples],
        distances,
    }
}

/// Uniform random samples on the unit 3-sphere S³ ⊂ R⁴.
/// Distances are geodesic (great-circle) distances.
pub fn generate_uniform_sphere3(n_samples: usize, seed: u64) -> SyntheticData {
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
    SyntheticData {
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
pub fn generate_uniform_hyperbolic3(n_samples: usize, seed: u64, max_r: f64) -> SyntheticData {
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
    SyntheticData {
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
];

/// Load a synthetic dataset by name (2D/3D frontend generators).
pub fn load_synthetic(name: &str, n_samples: usize, seed: u64) -> Result<SyntheticData, String> {
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
        _ => Err(format!(
            "Unknown synthetic dataset: {name}. Available: {DATASET_NAMES:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_uniform_range() {
        let mut rng = Rng::new(42);
        for _ in 0..10000 {
            let v = rng.uniform();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
