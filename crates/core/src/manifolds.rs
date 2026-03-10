use crate::config::ScalingLossType;
use crate::synthetic_data::Rng;

/// A point stored as a flat Vec<f64> of length n_points * ambient_dim (row-major).
/// This avoids nested allocations and is cache-friendly.
pub type Points = Vec<f64>;

/// Trait for constant curvature manifolds.
pub trait Manifold {
    fn curvature(&self) -> f64;
    fn radius(&self) -> f64;
    fn ambient_dim(&self, embed_dim: usize) -> usize;

    /// Initialize `n_points` on the manifold. Returns flat row-major array.
    fn init_points(&self, n_points: usize, embed_dim: usize, init_scale: f64, seed: u64) -> Points;

    /// Compute full pairwise distance matrix (n_points x n_points, row-major).
    fn pairwise_distances(&self, points: &[f64], n_points: usize, ambient_dim: usize) -> Vec<f64>;

    /// Project Euclidean gradient to tangent space (in-place on `grad`).
    fn project_to_tangent(
        &self,
        points: &[f64],
        grad: &mut [f64],
        n_points: usize,
        ambient_dim: usize,
    );

    /// Exponential map: move points along tangent vectors (updates `points` in-place).
    fn exp_map(&self, points: &mut [f64], tangent: &[f64], n_points: usize, ambient_dim: usize);

    /// Center points so Fréchet mean is at origin (in-place).
    fn center(&self, points: &mut [f64], n_points: usize, ambient_dim: usize);

    /// Differentiable scaling loss. Returns (loss_value, gradient wrt points).
    fn scaling_loss(&self, points: &[f64], n_points: usize, ambient_dim: usize) -> (f64, Vec<f64>);
}

// ---------------------------------------------------------------------------
// Euclidean
// ---------------------------------------------------------------------------

pub struct Euclidean;

impl Manifold for Euclidean {
    fn curvature(&self) -> f64 {
        0.0
    }
    fn radius(&self) -> f64 {
        1.0
    }
    fn ambient_dim(&self, embed_dim: usize) -> usize {
        embed_dim
    }

    fn init_points(&self, n_points: usize, embed_dim: usize, init_scale: f64, seed: u64) -> Points {
        let len = n_points * embed_dim;
        let mut rng = Rng::new(seed);
        let mut pts = Vec::with_capacity(len);
        for _ in 0..n_points {
            for _ in 0..embed_dim {
                pts.push(rng.normal() * init_scale);
            }
        }
        pts
    }

    fn pairwise_distances(&self, points: &[f64], n_points: usize, ambient_dim: usize) -> Vec<f64> {
        let mut dist = vec![0.0; n_points * n_points];
        for i in 0..n_points {
            for j in (i + 1)..n_points {
                let mut sq = 0.0;
                for d in 0..ambient_dim {
                    let diff = points[i * ambient_dim + d] - points[j * ambient_dim + d];
                    sq += diff * diff;
                }
                let d = sq.sqrt();
                dist[i * n_points + j] = d;
                dist[j * n_points + i] = d;
            }
        }
        dist
    }

    fn project_to_tangent(
        &self,
        _points: &[f64],
        _grad: &mut [f64],
        _n_points: usize,
        _ambient_dim: usize,
    ) {
        // Identity — tangent space is the whole space.
    }

    fn exp_map(&self, points: &mut [f64], tangent: &[f64], n_points: usize, ambient_dim: usize) {
        let len = n_points * ambient_dim;
        for k in 0..len {
            points[k] += tangent[k];
        }
    }

    fn center(&self, points: &mut [f64], n_points: usize, ambient_dim: usize) {
        for d in 0..ambient_dim {
            let mean: f64 = (0..n_points)
                .map(|i| points[i * ambient_dim + d])
                .sum::<f64>()
                / n_points as f64;
            for i in 0..n_points {
                points[i * ambient_dim + d] -= mean;
            }
        }
    }

    fn scaling_loss(&self, _points: &[f64], n_points: usize, ambient_dim: usize) -> (f64, Vec<f64>) {
        // No-op for Euclidean — scaling loss only applies to hyperbolic.
        (0.0, vec![0.0; n_points * ambient_dim])
    }
}

// ---------------------------------------------------------------------------
// Hyperboloid
// ---------------------------------------------------------------------------

pub struct Hyperboloid {
    curvature: f64,
    radius: f64,
    radius_sq: f64,
    scaling_loss_type: ScalingLossType,
}

impl Hyperboloid {
    pub fn new(curvature: f64, scaling_loss_type: ScalingLossType) -> Self {
        assert!(curvature < 0.0, "Hyperboloid requires negative curvature");
        let radius = 1.0 / (-curvature).sqrt();
        Self {
            curvature,
            radius,
            radius_sq: radius * radius,
            scaling_loss_type,
        }
    }

    /// Lorentzian inner product: -a0*b0 + a1*b1 + ... + ad*bd
    fn lorentz_inner(a: &[f64], b: &[f64]) -> f64 {
        let mut val = -a[0] * b[0];
        for i in 1..a.len() {
            val += a[i] * b[i];
        }
        val
    }
}

impl Manifold for Hyperboloid {
    fn curvature(&self) -> f64 {
        self.curvature
    }
    fn radius(&self) -> f64 {
        self.radius
    }
    fn ambient_dim(&self, embed_dim: usize) -> usize {
        embed_dim + 1
    }

    fn init_points(&self, n_points: usize, embed_dim: usize, init_scale: f64, seed: u64) -> Points {
        let ambient = embed_dim + 1;
        let mut rng = Rng::new(seed);
        let mut pts = vec![0.0; n_points * ambient];
        for i in 0..n_points {
            let mut spatial_sq = 0.0;
            for d in 0..embed_dim {
                let val = rng.normal() * init_scale;
                pts[i * ambient + 1 + d] = val;
                spatial_sq += val * val;
            }
            // Time component from constraint: -x0^2 + ||spatial||^2 = -r^2
            pts[i * ambient] = (self.radius_sq + spatial_sq).sqrt();
        }
        pts
    }

    fn pairwise_distances(&self, points: &[f64], n_points: usize, ambient_dim: usize) -> Vec<f64> {
        let r = self.radius;
        let r_sq = self.radius_sq;
        let eps = 1e-7;

        let mut dist = vec![0.0; n_points * n_points];
        for i in 0..n_points {
            let pi = &points[i * ambient_dim..(i + 1) * ambient_dim];
            for j in (i + 1)..n_points {
                let pj = &points[j * ambient_dim..(j + 1) * ambient_dim];
                let lorentz = Self::lorentz_inner(pi, pj);
                let arg = (-lorentz / r_sq).max(1.0 + eps);
                let d = r * arg.acosh();
                dist[i * n_points + j] = d;
                dist[j * n_points + i] = d;
            }
        }
        dist
    }

    fn project_to_tangent(
        &self,
        points: &[f64],
        grad: &mut [f64],
        n_points: usize,
        ambient_dim: usize,
    ) {
        let r_sq = self.radius_sq;
        for i in 0..n_points {
            let offset = i * ambient_dim;
            let x = &points[offset..offset + ambient_dim];

            // Apply inverse metric: negate time component
            grad[offset] = -grad[offset];

            // Lorentzian inner product <x, h>_L
            let h = &grad[offset..offset + ambient_dim];
            let inner = Self::lorentz_inner(x, h);

            // Project: v = h + <x,h>_L / r^2 * x
            let coeff = inner / r_sq;
            for d in 0..ambient_dim {
                grad[offset + d] += coeff * x[d];
            }
        }
    }

    fn exp_map(&self, points: &mut [f64], tangent: &[f64], n_points: usize, ambient_dim: usize) {
        let r = self.radius;
        let r_sq = self.radius_sq;

        for i in 0..n_points {
            let offset = i * ambient_dim;
            let v = &tangent[offset..offset + ambient_dim];

            // Lorentzian norm of tangent vector
            let v_sq = Self::lorentz_inner(v, v).max(1e-15);
            let v_norm = v_sq.sqrt();

            // Clip norm
            let max_norm = 10.0 * r;
            let scale = if v_norm > max_norm {
                max_norm / v_norm
            } else {
                1.0
            };
            let v_norm_clipped = v_norm * scale;

            let cosh_val = (v_norm_clipped / r).cosh();
            let sinh_val = (v_norm_clipped / r).sinh();

            // exp_x(v) = cosh(||v||/r)*x + sinh(||v||/r)*(r*v/||v||)
            for d in 0..ambient_dim {
                let x_d = points[offset + d];
                let v_d = tangent[offset + d] * scale;
                points[offset + d] = cosh_val * x_d + sinh_val * v_d / v_norm_clipped * r;
            }

            // Re-project to hyperboloid: x0 = sqrt(r^2 + ||spatial||^2)
            let mut spatial_sq = 0.0;
            for d in 1..ambient_dim {
                spatial_sq += points[offset + d].powi(2);
            }
            points[offset] = (r_sq + spatial_sq).sqrt();
        }
    }

    fn center(&self, points: &mut [f64], n_points: usize, ambient_dim: usize) {
        let r = self.radius;

        // Compute extrinsic mean
        let mut mean = vec![0.0; ambient_dim];
        for i in 0..n_points {
            for d in 0..ambient_dim {
                mean[d] += points[i * ambient_dim + d];
            }
        }
        for d in mean.iter_mut().take(ambient_dim) {
            *d /= n_points as f64;
        }

        // Normalize to hyperboloid
        let lorentz_sq = Self::lorentz_inner(&mean, &mean);
        if lorentz_sq >= 0.0 {
            return; // Can't center, degenerate case
        }
        let scale = r / (-lorentz_sq).sqrt();
        for d in mean.iter_mut().take(ambient_dim) {
            *d *= scale;
        }

        // Compute spatial norm of Fréchet mean
        let mut mu_spatial_sq = 0.0;
        for d in mean.iter_mut().take(ambient_dim).skip(1) {
            mu_spatial_sq += d.powi(2);
        }
        let mu_spatial_norm = mu_spatial_sq.sqrt();

        if mu_spatial_norm < 1e-10 {
            return; // Already centered
        }

        // Boost to center
        let cosh_alpha = mean[0] / r;
        let sinh_alpha = mu_spatial_norm / r;

        // Direction vector
        let mut n_dir = vec![0.0; ambient_dim - 1];
        for d in 0..(ambient_dim - 1) {
            n_dir[d] = mean[d + 1] / mu_spatial_norm;
        }

        for i in 0..n_points {
            let offset = i * ambient_dim;
            let x0 = points[offset];

            // x_par = spatial · n_dir
            let mut x_par = 0.0;
            for d in 0..(ambient_dim - 1) {
                x_par += points[offset + 1 + d] * n_dir[d];
            }

            // x_perp = spatial - x_par * n_dir
            let mut x_perp = vec![0.0; ambient_dim - 1];
            for d in 0..(ambient_dim - 1) {
                x_perp[d] = points[offset + 1 + d] - x_par * n_dir[d];
            }

            // Lorentz boost
            let new_x0 = cosh_alpha * x0 - sinh_alpha * x_par;
            let new_x_par = -sinh_alpha * x0 + cosh_alpha * x_par;

            points[offset] = new_x0;
            for d in 0..(ambient_dim - 1) {
                points[offset + 1 + d] = x_perp[d] + new_x_par * n_dir[d];
            }
        }
    }

    fn scaling_loss(&self, points: &[f64], n_points: usize, ambient_dim: usize) -> (f64, Vec<f64>) {
        let r = self.radius;
        let mut grad = vec![0.0; n_points * ambient_dim];

        match self.scaling_loss_type {
            ScalingLossType::HardBarrier => {
                let d_max = 3.0 * r;
                let mut loss = 0.0;
                for i in 0..n_points {
                    let x0 = points[i * ambient_dim];
                    let arg = (x0 / r).max(1.0 + 1e-7);
                    let geo_dist = r * arg.acosh();
                    let excess = (geo_dist - d_max).max(0.0);
                    loss += excess * excess;

                    if excess > 0.0 {
                        // d(geo_dist)/d(x0) = 1 / sqrt(x0^2/r^2 - 1)
                        let dg_dx0 = 1.0 / (arg * arg - 1.0).max(1e-12).sqrt();
                        grad[i * ambient_dim] = 2.0 * excess * dg_dx0 / n_points as f64;
                    }
                }
                loss /= n_points as f64;
                (loss, grad)
            }
            _ => (0.0, grad),
        }
    }
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

pub struct Sphere {
    curvature: f64,
    radius: f64,
    radius_sq: f64,
}

impl Sphere {
    pub fn new(curvature: f64) -> Self {
        assert!(curvature > 0.0, "Sphere requires positive curvature");
        let radius = 1.0 / curvature.sqrt();
        Self {
            curvature,
            radius,
            radius_sq: radius * radius,
        }
    }
}

impl Manifold for Sphere {
    fn curvature(&self) -> f64 {
        self.curvature
    }
    fn radius(&self) -> f64 {
        self.radius
    }
    fn ambient_dim(&self, embed_dim: usize) -> usize {
        embed_dim + 1
    }

    fn init_points(&self, n_points: usize, embed_dim: usize, init_scale: f64, seed: u64) -> Points {
        let ambient = embed_dim + 1;
        let r = self.radius;
        let mut rng = Rng::new(seed);
        let mut pts = vec![0.0; n_points * ambient];
        for i in 0..n_points {
            // Generate random point on sphere: normal vector, then normalize to radius
            let mut norm_sq = 0.0;
            for d in 0..ambient {
                let val = rng.normal();
                pts[i * ambient + d] = val;
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt().max(1e-10);
            // Scale: place near south pole (x0 ≈ -r) with spread controlled by init_scale
            // First normalize to sphere
            for d in 0..ambient {
                pts[i * ambient + d] *= r / norm;
            }
            // Apply init_scale: interpolate between south pole and random point
            // Small init_scale → clustered near south pole, large → spread across sphere
            let scale = init_scale.min(1.0);
            // Use exponential map from south pole with scaled tangent
            // Simpler: generate random direction, scale the angular distance
            let x0 = pts[i * ambient];
            let mut spatial_sq = 0.0;
            for d in 1..ambient {
                spatial_sq += pts[i * ambient + d].powi(2);
            }
            let spatial_norm = spatial_sq.sqrt();
            // Current angular distance from south pole (-r, 0, ..., 0)
            let theta = ((-x0 / r).clamp(-1.0, 1.0)).acos(); // angle from south pole
            let new_theta = theta * scale;
            // Reconstruct with new angle
            if spatial_norm > 1e-10 {
                pts[i * ambient] = -r * new_theta.cos();
                let sin_factor = r * new_theta.sin() / spatial_norm;
                for d in 1..ambient {
                    pts[i * ambient + d] *= sin_factor;
                }
            }
        }
        pts
    }

    fn pairwise_distances(&self, points: &[f64], n_points: usize, ambient_dim: usize) -> Vec<f64> {
        let r = self.radius;
        let r_sq = self.radius_sq;
        let eps = 1e-7;

        let mut dist = vec![0.0; n_points * n_points];
        for i in 0..n_points {
            let pi = &points[i * ambient_dim..(i + 1) * ambient_dim];
            for j in (i + 1)..n_points {
                let pj = &points[j * ambient_dim..(j + 1) * ambient_dim];
                let mut dot = 0.0;
                for d in 0..ambient_dim {
                    dot += pi[d] * pj[d];
                }
                let arg = (dot / r_sq).clamp(-1.0 + eps, 1.0 - eps);
                let d = r * arg.acos();
                dist[i * n_points + j] = d;
                dist[j * n_points + i] = d;
            }
        }
        dist
    }

    fn project_to_tangent(
        &self,
        points: &[f64],
        grad: &mut [f64],
        n_points: usize,
        ambient_dim: usize,
    ) {
        let r_sq = self.radius_sq;
        for i in 0..n_points {
            let offset = i * ambient_dim;
            let x = &points[offset..offset + ambient_dim];

            // <grad, x>
            let mut inner = 0.0;
            for d in 0..ambient_dim {
                inner += grad[offset + d] * x[d];
            }

            // proj_x(h) = h - <h,x>/r^2 * x
            let coeff = inner / r_sq;
            for d in 0..ambient_dim {
                grad[offset + d] -= coeff * x[d];
            }
        }
    }

    fn exp_map(&self, points: &mut [f64], tangent: &[f64], n_points: usize, ambient_dim: usize) {
        let r = self.radius;
        let r_sq = self.radius_sq;

        for i in 0..n_points {
            let offset = i * ambient_dim;

            // ||v||
            let mut v_sq = 0.0;
            for d in 0..ambient_dim {
                v_sq += tangent[offset + d].powi(2);
            }
            let v_norm = v_sq.sqrt().max(1e-10);

            let cos_val = (v_norm / r).cos();
            let sin_val = (v_norm / r).sin();

            for d in 0..ambient_dim {
                let x_d = points[offset + d];
                let v_d = tangent[offset + d];
                points[offset + d] = cos_val * x_d + sin_val * v_d / v_norm * r;
            }

            // Re-project to sphere
            let mut norm_sq = 0.0;
            for d in 0..ambient_dim {
                norm_sq += points[offset + d].powi(2);
            }
            let norm = norm_sq.sqrt();
            let target = r_sq.sqrt();
            for d in 0..ambient_dim {
                points[offset + d] *= target / norm;
            }
        }
    }

    fn center(&self, _points: &mut [f64], _n_points: usize, _ambient_dim: usize) {
        // No-op for sphere (centering is a rotation, doesn't affect KL loss).
    }

    fn scaling_loss(
        &self,
        _points: &[f64],
        n_points: usize,
        ambient_dim: usize,
    ) -> (f64, Vec<f64>) {
        (0.0, vec![0.0; n_points * ambient_dim])
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

pub fn create_manifold(curvature: f64, scaling_loss_type: ScalingLossType) -> Box<dyn Manifold> {
    if curvature > 0.0 {
        Box::new(Sphere::new(curvature))
    } else if curvature == 0.0 {
        Box::new(Euclidean)
    } else {
        Box::new(Hyperboloid::new(curvature, scaling_loss_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperboloid_constraint() {
        let h = Hyperboloid::new(-1.0, ScalingLossType::HardBarrier);
        let pts = h.init_points(10, 2, 0.01, 42);
        let ambient = 3;
        for i in 0..10 {
            let x0 = pts[i * ambient];
            let x1 = pts[i * ambient + 1];
            let x2 = pts[i * ambient + 2];
            let lorentz = -x0 * x0 + x1 * x1 + x2 * x2;
            assert!((lorentz + 1.0).abs() < 1e-6, "Point {i} lorentz={lorentz}");
        }
    }

    #[test]
    fn test_sphere_constraint() {
        let s = Sphere::new(1.0);
        let pts = s.init_points(10, 2, 0.01, 42);
        let ambient = 3;
        for i in 0..10 {
            let mut norm_sq = 0.0;
            for d in 0..ambient {
                norm_sq += pts[i * ambient + d].powi(2);
            }
            assert!((norm_sq - 1.0).abs() < 1e-6, "Point {i} norm_sq={norm_sq}");
        }
    }

    #[test]
    fn test_euclidean_distances() {
        let e = Euclidean;
        // Two points in 2D: (0, 0) and (3, 4) -> distance = 5
        let pts = vec![0.0, 0.0, 3.0, 4.0];
        let dist = e.pairwise_distances(&pts, 2, 2);
        assert!((dist[1] - 5.0).abs() < 1e-10);
    }
}
