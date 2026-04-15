use crate::manifolds::Manifold;

/// Compute globally-normalized similarity matrix from pairwise distances.
///
/// Used for the Zhou & Sharpee global t-SNE loss. Unlike the standard Q matrix
/// which uses `(1 + d²)^{-1}` (emphasizing small distances), this uses `(1 + d²)`
/// (emphasizing large distances), making it sensitive to global structure.
///
/// `result_ij = (1 + d²_ij) / Σ_{m≠n}(1 + d²_mn)`
pub fn compute_global_similarities(distances: &[f64], n_points: usize) -> Vec<f64> {
    let mut kernel = vec![0.0; n_points * n_points];
    let mut total = 0.0;
    for i in 0..n_points {
        for j in 0..n_points {
            if i != j {
                let d = distances[i * n_points + j];
                let v = 1.0 + d * d;
                kernel[i * n_points + j] = v;
                total += v;
            }
        }
    }
    let total = if total == 0.0 { 1e-10 } else { total };
    for v in &mut kernel {
        *v /= total;
    }
    kernel
}

/// Norm loss: H = (1/m) * Σ_i (||x_i||² - ||y_i||²)²
///
/// Tries to match the squared Euclidean norm of each input point to its
/// embedding. For tree-structured hyperbolic data this preserves the hierarchy
/// level (root ~= origin, leaves ~= boundary of the Poincaré ball).
///
/// Returns `(loss, gradient_in_ambient_space)`. The gradient is in ambient
/// coordinates; the caller must project it to the tangent space before adding
/// it to the main gradient.
pub fn norm_loss_gradient(
    input_data: &[f64],
    points: &[f64],
    n_points: usize,
    input_dim: usize,
    ambient_dim: usize,
) -> (f64, Vec<f64>) {
    let mut loss = 0.0;
    let mut grad = vec![0.0; n_points * ambient_dim];

    for i in 0..n_points {
        let x_norm_sq: f64 = input_data[i * input_dim..(i + 1) * input_dim]
            .iter()
            .map(|&v| v * v)
            .sum();
        let y_norm_sq: f64 = points[i * ambient_dim..(i + 1) * ambient_dim]
            .iter()
            .map(|&v| v * v)
            .sum();

        let diff = y_norm_sq - x_norm_sq;
        loss += diff * diff;

        // ∂(diff²)/∂y_i = 2·diff · 2·y_i = 4·diff·y_i
        let coeff = 4.0 * diff;
        for d in 0..ambient_dim {
            grad[i * ambient_dim + d] = coeff * points[i * ambient_dim + d];
        }
    }

    let m = n_points as f64;
    loss /= m;
    for g in &mut grad {
        *g /= m;
    }

    (loss, grad)
}

/// Depth norm loss for distance-based (graph/tree) data.
///
/// Compares each embedding point's "depth" (distance from the embedding origin)
/// to a pre-computed target depth derived from the point's graph distance to the
/// root node.  This is the distance-based analogue of `norm_loss_gradient`, which
/// requires feature vectors and is used for Euclidean input data.
///
/// For **hyperbolic** embeddings (k < 0) the depth of a point on the hyperboloid
/// is measured as its Poincaré ball radius after stereographic projection:
///   r = ||spatial|| / (t + R)
/// where `t = y[0]` (time component) and `spatial = y[1..]`.
/// The target Poincaré radius is `tanh(d_root / (2R))`, which maps hop-distances
/// to the expected Poincaré radius under an ideal hyperbolic tree embedding.
///
/// For **Euclidean** embeddings (k = 0) the depth is the plain Euclidean norm
/// and the target is the raw hop-distance (no scaling needed).
///
/// For **spherical** embeddings (k > 0) all points share the same radius, so
/// depth is not meaningful; this function returns zero loss and gradient.
///
/// Returns `(loss, gradient_in_ambient_space)`. The gradient is in ambient
/// coordinates; the caller must project it to the tangent space before use.
pub fn depth_norm_loss_gradient(
    points: &[f64],
    target_norms: &[f64],
    n_points: usize,
    ambient_dim: usize,
    curvature: f64,
    radius: f64,
) -> (f64, Vec<f64>) {
    let mut loss = 0.0;
    let mut grad = vec![0.0f64; n_points * ambient_dim];

    for (i, target) in target_norms.iter().enumerate().take(n_points) {
        let oi = i * ambient_dim;

        if curvature < 0.0 {
            // Hyperboloid: Poincaré radius r = ||spatial|| / (t + R)
            let t = points[oi];
            let norm_s = (1..ambient_dim)
                .map(|d| points[oi + d] * points[oi + d])
                .sum::<f64>()
                .sqrt();

            let denom = t + radius;
            if denom < 1e-12 || norm_s < 1e-12 {
                continue;
            }
            let r = norm_s / denom;
            let diff = r - target;
            loss += diff * diff;

            let coeff = 2.0 * diff;
            // ∂r/∂t = -norm_s / denom²
            grad[oi] = coeff * (-norm_s / (denom * denom));
            // ∂r/∂s_d = s_d / (norm_s * denom)
            for d in 1..ambient_dim {
                grad[oi + d] = coeff * points[oi + d] / (norm_s * denom);
            }
        } else if curvature == 0.0 {
            // Euclidean: depth = ||y||, target = raw hop-distance
            let norm_y = (0..ambient_dim)
                .map(|d| points[oi + d] * points[oi + d])
                .sum::<f64>()
                .sqrt();
            if norm_y < 1e-12 {
                continue;
            }
            let diff = norm_y - target;
            loss += diff * diff;
            let coeff = 2.0 * diff / norm_y;
            for d in 0..ambient_dim {
                grad[oi + d] = coeff * points[oi + d];
            }
        }
        // k > 0 (sphere): depth not meaningful, contribute zero.
    }

    // Intentionally NOT divided by m (unlike `norm_loss_gradient`).
    //
    // The KL gradient accumulates ~perplexity pairwise terms per point, giving
    // it an effective scale of O(perplexity).  The depth-norm gradient has only
    // one term per point and Poincaré residuals bounded in [0, 1], so dividing
    // by m would make it O(1/n) — roughly perplexity × n times smaller than the
    // KL gradient, forcing norm_loss_weight into the [10, 100] range to compensate.
    //
    // Using a sum (no /m) gives O(1) scale, making norm_loss_weight ≈ 0.01–1.0
    // sufficient to balance the KL loss — consistent with CO-SNE's λ₂ = 0.01.
    (loss, grad)
}

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
