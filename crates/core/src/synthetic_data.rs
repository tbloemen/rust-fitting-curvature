//! Synthetic dataset generators with known intrinsic curvature.
//!
//! Each generator returns `(X, labels, D)` where:
//! - X: ambient coordinates (flat row-major, shape n × d)
//! - labels: integer labels (length n)
//! - D: precomputed distance matrix (flat n × n)

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
// Euclidean generators
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

/// Compute pairwise great-circle distances on the unit sphere.
fn spherical_distances(x: &[f64], n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dot = 0.0;
            for k in 0..3 {
                dot += x[i * 3 + k] * x[j * 3 + k];
            }
            let dist = dot.clamp(-1.0, 1.0).acos();
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }
    d
}

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

        // Normalize for safety
        let norm = (px * px + py * py + pz * pz).sqrt();
        x.push(px / norm);
        x.push(py / norm);
        x.push(pz / norm);
        labels.push(if pz >= 0.0 { 1 } else { 0 });
        count += 1;
    }

    let distances = spherical_distances(&x, n_samples);

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

        // Point in coordinate system where mu = [0, 0, 1]
        let px = sqrt_term * angle.cos();
        let py = sqrt_term * angle.sin();
        let pz = w;

        let norm = (px * px + py * py + pz * pz).sqrt();
        points.push(px / norm);
        points.push(py / norm);
        points.push(pz / norm);

        // Label by distance from north pole
        labels.push(if count < n_samples / 2 { 0 } else { 1 });
        count += 1;
    }

    // Relabel by median distance from pole
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

    let distances = spherical_distances(&points, n_samples);

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
        x.push(-south_raw.x[i * 3 + 2]); // flip z
    }

    let mut labels = vec![0u32; n_north];
    labels.extend(vec![1u32; n_south]);

    let distances = spherical_distances(&x, n_samples);

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

/// Convert Poincaré disk coordinates to hyperboloid model.
fn poincare_to_hyperboloid(p: &[f64], n: usize) -> Vec<f64> {
    let mut x = Vec::with_capacity(n * 3);
    for i in 0..n {
        let p1 = p[i * 2];
        let p2 = p[i * 2 + 1];
        let sq_norm = p1 * p1 + p2 * p2;
        let denom = 1.0 - sq_norm;
        let x0 = (1.0 + sq_norm) / denom;
        let x1 = 2.0 * p1 / denom;
        let x2 = 2.0 * p2 / denom;
        x.push(x0);
        x.push(x1);
        x.push(x2);
    }
    x
}

/// Compute pairwise hyperboloid distances.
fn hyperboloid_distances(x: &[f64], n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            // Lorentzian inner product: -x0*y0 + x1*y1 + x2*y2
            let inner =
                -x[i * 3] * x[j * 3] + x[i * 3 + 1] * x[j * 3 + 1] + x[i * 3 + 2] * x[j * 3 + 2];
            let minus_inner = (-inner).max(1.0);
            let dist = minus_inner.acosh();
            d[i * n + j] = dist;
            d[j * n + i] = dist;
        }
    }
    d
}

/// Proper sinh-weighted radial sampling in Poincaré disk, labels by radius bins.
pub fn generate_uniform_hyperbolic(n_samples: usize, seed: u64) -> SyntheticData {
    let mut rng = Rng::new(seed);
    let max_rho = 3.0_f64;

    let mut poincare = Vec::with_capacity(n_samples * 2);
    let mut labels = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let u = rng.uniform();
        // Inverse CDF: cosh(rho) = 1 + u*(cosh(max_rho) - 1)
        let cosh_rho = 1.0 + u * (max_rho.cosh() - 1.0);
        let rho = cosh_rho.acosh();

        let poincare_r = (rho / 2.0).tanh();
        let angle = rng.uniform() * 2.0 * PI;

        poincare.push(poincare_r * angle.cos());
        poincare.push(poincare_r * angle.sin());

        // Labels by radius bins (3 bins)
        let label = ((rho / max_rho * 3.0) as u32).min(2);
        labels.push(label);
    }

    let x = poincare_to_hyperboloid(&poincare, n_samples);
    let distances = hyperboloid_distances(&x, n_samples);

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
    let max_depth = (n_samples as f64).log2().ceil() as usize;
    let max_depth = max_depth.max(2);

    let mut poincare = Vec::new();
    let mut labels = Vec::new();

    // Root at origin
    poincare.push(0.0);
    poincare.push(0.0);
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

    // Fill remaining with random points
    while labels.len() < n_samples {
        let depth = (rng.uniform() * max_depth as f64) as usize + 1;
        let r = (depth as f64 * 0.8 / 2.0).tanh();
        let angle = rng.uniform() * 2.0 * PI;
        poincare.push(r * angle.cos());
        poincare.push(r * angle.sin());
        labels.push((depth as u32).min(4));
    }

    // Truncate if needed
    poincare.truncate(n_samples * 2);
    labels.truncate(n_samples);

    // Clamp to stay inside disk
    for i in 0..n_samples {
        let p1 = poincare[i * 2];
        let p2 = poincare[i * 2 + 1];
        let norm = (p1 * p1 + p2 * p2).sqrt();
        if norm >= 1.0 {
            poincare[i * 2] = p1 / norm * 0.99;
            poincare[i * 2 + 1] = p2 / norm * 0.99;
        }
    }

    let x = poincare_to_hyperboloid(&poincare, n_samples);
    let distances = hyperboloid_distances(&x, n_samples);

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

    let x = poincare_to_hyperboloid(&poincare, n_samples);
    let distances = hyperboloid_distances(&x, n_samples);

    SyntheticData {
        x,
        n_points: n_samples,
        ambient_dim: 3,
        labels,
        distances,
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Available synthetic dataset names.
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

/// Load a synthetic dataset by name.
pub fn load_synthetic(name: &str, n_samples: usize, seed: u64) -> Result<SyntheticData, String> {
    match name {
        "uniform_grid" => Ok(generate_uniform_grid(n_samples, seed)),
        "gaussian_blob" => Ok(generate_gaussian_blob(n_samples, seed)),
        "concentric_circles" => Ok(generate_concentric_circles(n_samples, seed)),
        "uniform_sphere" => Ok(generate_uniform_sphere(n_samples, seed)),
        "von_mises_fisher" => Ok(generate_von_mises_fisher(n_samples, seed)),
        "antipodal_clusters" => Ok(generate_antipodal_clusters(n_samples, seed)),
        "uniform_hyperbolic" => Ok(generate_uniform_hyperbolic(n_samples, seed)),
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
