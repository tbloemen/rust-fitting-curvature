use crate::manifolds::Manifold;

/// Student-t kernel: k(d) = (1 + d^2/dof)^(-(dof+1)/2)
pub fn t_distribution_kernel(distances: &[f64], dof: f64) -> Vec<f64> {
    let exponent = -(dof + 1.0) / 2.0;
    distances
        .iter()
        .map(|&d| (1.0 + d * d / dof).powf(exponent))
        .collect()
}

/// Compute normalized Q matrix for t-SNE.
///
/// Returns flat n_points x n_points row-major matrix.
pub fn compute_q_matrix(
    manifold: &dyn Manifold,
    points: &[f64],
    n_points: usize,
    ambient_dim: usize,
    dof: f64,
) -> Vec<f64> {
    let dist = manifold.pairwise_distances(points, n_points, ambient_dim);
    let mut kernel = t_distribution_kernel(&dist, dof);

    // Zero diagonal
    for i in 0..n_points {
        kernel[i * n_points + i] = 0.0;
    }

    // Normalize
    let total: f64 = kernel.iter().sum();
    let total = if total == 0.0 { 1e-10 } else { total };
    for val in &mut kernel {
        *val /= total;
    }

    kernel
}

/// Compute Q matrix and also return the distances (avoids recomputation for gradient).
pub fn compute_q_matrix_with_distances(
    manifold: &dyn Manifold,
    points: &[f64],
    n_points: usize,
    ambient_dim: usize,
    dof: f64,
) -> (Vec<f64>, Vec<f64>) {
    let dist = manifold.pairwise_distances(points, n_points, ambient_dim);
    let mut kernel = t_distribution_kernel(&dist, dof);

    for i in 0..n_points {
        kernel[i * n_points + i] = 0.0;
    }

    let total: f64 = kernel.iter().sum();
    let total = if total == 0.0 { 1e-10 } else { total };
    for val in &mut kernel {
        *val /= total;
    }

    (kernel, dist)
}
