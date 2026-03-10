use crate::manifolds::Manifold;

/// KL divergence loss: -sum(P * log(Q + eps)), excluding diagonal.
pub fn kl_loss(q: &[f64], p: &[f64], n_points: usize) -> f64 {
    let eps = 1e-12;
    let mut loss = 0.0;
    for i in 0..n_points {
        for j in 0..n_points {
            if i != j {
                let idx = i * n_points + j;
                loss -= p[idx] * (q[idx] + eps).ln();
            }
        }
    }
    loss
}

/// Compute the Riemannian gradient of KL(P || Q) with respect to embedding points.
///
/// Returns a tangent vector at each point (no further projection needed).
///
/// The general formula is:
///   grad_{y_i} C = 4 * sum_j (p_ij - q_ij) * w_ij * (-log_{y_i}(y_j))
///
/// where w_ij = (1 + d_ij^2)^{-1} and log_{y_i}(y_j) is the Riemannian log map.
///
/// - Euclidean: -log_{y_i}(y_j) = y_i - y_j
/// - Hyperboloid: uses Lorentzian log map
/// - Sphere: uses spherical log map
pub fn kl_gradient(
    manifold: &dyn Manifold,
    points: &[f64],
    q: &[f64],
    p: &[f64],
    distances: &[f64],
    n_points: usize,
    ambient_dim: usize,
) -> Vec<f64> {
    let k = manifold.curvature();
    if k < 0.0 {
        kl_gradient_hyperboloid(
            manifold.radius(),
            points,
            q,
            p,
            distances,
            n_points,
            ambient_dim,
        )
    } else if k > 0.0 {
        kl_gradient_sphere(
            manifold.radius(),
            points,
            q,
            p,
            distances,
            n_points,
            ambient_dim,
        )
    } else {
        kl_gradient_euclidean(points, q, p, distances, n_points, ambient_dim)
    }
}

/// Euclidean KL gradient: -log_{y_i}(y_j) = y_i - y_j.
fn kl_gradient_euclidean(
    points: &[f64],
    q: &[f64],
    p: &[f64],
    distances: &[f64],
    n_points: usize,
    ambient_dim: usize,
) -> Vec<f64> {
    let mut grad = vec![0.0; n_points * ambient_dim];

    for i in 0..n_points {
        for j in 0..n_points {
            if i == j {
                continue;
            }
            let idx = i * n_points + j;
            let d = distances[idx];
            let factor = 4.0 * (p[idx] - q[idx]) / (1.0 + d * d);

            for dim in 0..ambient_dim {
                let diff = points[i * ambient_dim + dim] - points[j * ambient_dim + dim];
                grad[i * ambient_dim + dim] += factor * diff;
            }
        }
    }

    grad
}

/// Hyperboloid KL gradient using the Lorentzian log map.
///
/// For points x, y on the hyperboloid of radius r:
///   -log_x(y) = -(acosh(alpha) / sqrt(alpha^2 - 1)) * (y - alpha * x)
/// where alpha = -<x, y>_L / r^2  and  <.,.>_L is the Lorentz inner product.
fn kl_gradient_hyperboloid(
    radius: f64,
    points: &[f64],
    q: &[f64],
    p: &[f64],
    distances: &[f64],
    n_points: usize,
    ambient_dim: usize,
) -> Vec<f64> {
    let mut grad = vec![0.0; n_points * ambient_dim];
    let r_sq = radius * radius;

    for i in 0..n_points {
        let oi = i * ambient_dim;
        for j in 0..n_points {
            if i == j {
                continue;
            }
            let idx = i * n_points + j;
            let d = distances[idx];
            let w_ij = 1.0 / (1.0 + d * d);
            let coeff = 4.0 * (p[idx] - q[idx]) * w_ij;

            let oj = j * ambient_dim;

            // Lorentz inner product <y_i, y_j>_L = -y0*y0 + y1*y1 + ...
            let mut lorentz = -points[oi] * points[oj];
            for dim in 1..ambient_dim {
                lorentz += points[oi + dim] * points[oj + dim];
            }

            let alpha = (-lorentz / r_sq).max(1.0);

            // Scale factor: acosh(alpha) / sqrt(alpha^2 - 1)
            // For alpha -> 1, this ratio -> 1 (L'Hopital).
            let scale = if alpha < 1.0 + 1e-10 {
                1.0
            } else {
                alpha.acosh() / (alpha * alpha - 1.0).sqrt()
            };

            // -log_{y_i}(y_j) = -scale * (y_j - alpha * y_i)
            for dim in 0..ambient_dim {
                let u = points[oj + dim] - alpha * points[oi + dim];
                grad[oi + dim] += coeff * (-scale * u);
            }
        }
    }

    grad
}

/// Sphere KL gradient using the spherical log map.
///
/// For points x, y on the sphere of radius r:
///   -log_x(y) = -(theta / sin(theta)) * (y - cos(theta) * x)
/// where cos(theta) = <x, y> / r^2.
fn kl_gradient_sphere(
    radius: f64,
    points: &[f64],
    q: &[f64],
    p: &[f64],
    distances: &[f64],
    n_points: usize,
    ambient_dim: usize,
) -> Vec<f64> {
    let mut grad = vec![0.0; n_points * ambient_dim];
    let r_sq = radius * radius;

    for i in 0..n_points {
        let oi = i * ambient_dim;
        for j in 0..n_points {
            if i == j {
                continue;
            }
            let idx = i * n_points + j;
            let d = distances[idx];
            let w_ij = 1.0 / (1.0 + d * d);
            let coeff = 4.0 * (p[idx] - q[idx]) * w_ij;

            let oj = j * ambient_dim;

            // Euclidean inner product <y_i, y_j>
            let mut inner = 0.0;
            for dim in 0..ambient_dim {
                inner += points[oi + dim] * points[oj + dim];
            }

            let cos_theta = (inner / r_sq).clamp(-1.0 + 1e-7, 1.0 - 1e-7);
            let theta = cos_theta.acos();
            let sin_theta = theta.sin();

            // Scale factor: theta / sin(theta). For theta -> 0, this -> 1.
            let scale = if sin_theta < 1e-10 {
                1.0
            } else {
                theta / sin_theta
            };

            // -log_{y_i}(y_j) = -scale * (y_j - cos_theta * y_i)
            for dim in 0..ambient_dim {
                let u = points[oj + dim] - cos_theta * points[oi + dim];
                grad[oi + dim] += coeff * (-scale * u);
            }
        }
    }

    grad
}
