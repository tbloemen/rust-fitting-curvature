use crate::manifolds::Manifold;

/// Riemannian SGD with momentum optimizer.
///
/// Implements Algorithm 1 from Gu et al. (2019) with parallel transport for momentum.
pub struct RiemannianSGDMomentum {
    pub lr: f64,
    pub momentum: f64,
    velocity: Vec<f64>,
    prev_points: Vec<f64>,
    initialized: bool,
}

impl RiemannianSGDMomentum {
    pub fn new(lr: f64, momentum: f64, n_points: usize, ambient_dim: usize) -> Self {
        let len = n_points * ambient_dim;
        Self {
            lr,
            momentum,
            velocity: vec![0.0; len],
            prev_points: vec![0.0; len],
            initialized: false,
        }
    }

    pub fn set_momentum(&mut self, momentum: f64) {
        self.momentum = momentum;
    }

    /// Perform one optimization step given Riemannian gradients (tangent vectors).
    ///
    /// The gradient must already be a tangent vector at each point.
    /// For the KL gradient, this is computed directly via log maps.
    /// For other gradient terms (e.g. scaling loss), project to tangent space
    /// before passing to this method.
    pub fn step(
        &mut self,
        manifold: &dyn Manifold,
        points: &mut [f64],
        grad: &[f64],
        n_points: usize,
        ambient_dim: usize,
    ) {
        let len = n_points * ambient_dim;

        // Parallel transport velocity from previous tangent space
        if self.initialized {
            self.parallel_transport(manifold, points, n_points, ambient_dim);
        }

        // Update velocity: v = momentum * v - lr * grad
        for (k, grad_k) in grad.iter().enumerate().take(len) {
            self.velocity[k] = self.momentum * self.velocity[k] - self.lr * grad_k;
        }

        // Clip velocity magnitude
        let mut vel_norm_sq = 0.0;
        for k in 0..len {
            vel_norm_sq += self.velocity[k].powi(2);
        }
        let vel_norm = vel_norm_sq.sqrt();
        let max_vel = 10.0;
        if vel_norm > max_vel {
            let scale = max_vel / vel_norm;
            for k in 0..len {
                self.velocity[k] *= scale;
            }
        }

        // Store current points before update
        self.prev_points[..len].copy_from_slice(&points[..len]);

        // Update via exponential map
        manifold.exp_map(points, &self.velocity, n_points, ambient_dim);

        self.initialized = true;
    }

    fn parallel_transport(
        &mut self,
        manifold: &dyn Manifold,
        points: &[f64],
        n_points: usize,
        ambient_dim: usize,
    ) {
        let k = manifold.curvature();
        if k == 0.0 {
            return; // Euclidean: identity transport
        }

        for i in 0..n_points {
            let offset = i * ambient_dim;
            let x_old = &self.prev_points[offset..offset + ambient_dim];
            let x_new = &points[offset..offset + ambient_dim];
            let v = &self.velocity[offset..offset + ambient_dim];

            let transported = if k < 0.0 {
                transport_hyperboloid(x_old, x_new, v, manifold.radius())
            } else {
                transport_sphere(x_old, x_new, v, manifold.radius())
            };

            self.velocity[offset..offset + ambient_dim].copy_from_slice(&transported);
        }
    }
}

/// Parallel transport on hyperboloid (He 2024, Eq. 5).
fn transport_hyperboloid(x_old: &[f64], x_new: &[f64], v: &[f64], radius: f64) -> Vec<f64> {
    let r_sq = radius * radius;
    let dim = x_old.len();

    // <x_new, v>_L
    let mut inner_new_v = -x_new[0] * v[0];
    for d in 1..dim {
        inner_new_v += x_new[d] * v[d];
    }

    // <x_old, x_new>_L
    let mut inner_old_new = -x_old[0] * x_new[0];
    for d in 1..dim {
        inner_old_new += x_old[d] * x_new[d];
    }

    let denom = (r_sq - inner_old_new).max(1e-8);
    let coeff = inner_new_v / denom;

    let mut transported = vec![0.0; dim];
    for d in 0..dim {
        transported[d] = v[d] + coeff * (x_old[d] + x_new[d]);
    }

    // Re-project to tangent space at x_new
    let mut inner = -x_new[0] * transported[0];
    for d in 1..dim {
        inner += x_new[d] * transported[d];
    }
    let proj_coeff = inner / r_sq;
    for d in 0..dim {
        transported[d] += proj_coeff * x_new[d];
    }

    transported
}

/// Parallel transport on sphere.
fn transport_sphere(x_old: &[f64], x_new: &[f64], v: &[f64], radius: f64) -> Vec<f64> {
    let r_sq = radius * radius;
    let dim = x_old.len();

    let mut inner_old_new = 0.0;
    let mut inner_new_v = 0.0;
    for d in 0..dim {
        inner_old_new += x_old[d] * x_new[d];
        inner_new_v += x_new[d] * v[d];
    }

    let denom = (r_sq + inner_old_new).max(1e-8);
    let coeff = inner_new_v / denom;

    let mut transported = vec![0.0; dim];
    for d in 0..dim {
        transported[d] = v[d] - coeff * (x_old[d] + x_new[d]);
    }

    // Re-project to tangent space
    let mut inner = 0.0;
    for d in 0..dim {
        inner += x_new[d] * transported[d];
    }
    let proj_coeff = inner / r_sq;
    for d in 0..dim {
        transported[d] -= proj_coeff * x_new[d];
    }

    transported
}
