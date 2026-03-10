use fitting_core::config::ScalingLossType;
use fitting_core::kernels::compute_q_matrix_with_distances;
use fitting_core::kl_divergence::kl_gradient;
use fitting_core::manifolds::{self, Manifold};

/// Helper: build a small set of points on the manifold, a fake P matrix,
/// compute Q and distances, then return everything needed for gradient tests.
struct GradientTestSetup {
    manifold: Box<dyn Manifold>,
    points: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
    distances: Vec<f64>,
    n_points: usize,
    ambient_dim: usize,
}

impl GradientTestSetup {
    fn new(curvature: f64, n_points: usize, embed_dim: usize) -> Self {
        let manifold = manifolds::create_manifold(curvature, ScalingLossType::None);
        let ambient_dim = manifold.ambient_dim(embed_dim);
        let points = manifold.init_points(n_points, embed_dim, 0.1, 42);

        // Build a fake symmetric P matrix
        let mut p = vec![0.0; n_points * n_points];
        for i in 0..n_points {
            for j in 0..n_points {
                if i != j {
                    // Non-uniform P: use (i+j+1) as weight
                    p[i * n_points + j] = (i + j + 1) as f64;
                }
            }
        }
        // Normalize
        let sum: f64 = p.iter().sum();
        for v in &mut p {
            *v /= sum;
        }
        // Symmetrize
        for i in 0..n_points {
            for j in (i + 1)..n_points {
                let avg = (p[i * n_points + j] + p[j * n_points + i]) / 2.0;
                p[i * n_points + j] = avg;
                p[j * n_points + i] = avg;
            }
        }
        // Re-normalize after symmetrization
        let sum: f64 = p.iter().sum();
        for v in &mut p {
            *v /= sum;
        }

        let (q, distances) =
            compute_q_matrix_with_distances(manifold.as_ref(), &points, n_points, ambient_dim, 1.0);

        Self {
            manifold,
            points,
            p,
            q,
            distances,
            n_points,
            ambient_dim,
        }
    }
}

/// KL loss computation (mirrors embedding.rs kl_loss).
fn kl_loss(q: &[f64], p: &[f64], n_points: usize) -> f64 {
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

// -----------------------------------------------------------------------
// Test: gradient is a tangent vector
// -----------------------------------------------------------------------

#[test]
fn test_hyperboloid_gradient_is_tangent() {
    let s = GradientTestSetup::new(-1.0, 5, 2);
    let grad = kl_gradient(
        s.manifold.as_ref(),
        &s.points,
        &s.q,
        &s.p,
        &s.distances,
        s.n_points,
        s.ambient_dim,
    );

    // For each point, check <grad_i, x_i>_L = 0
    for i in 0..s.n_points {
        let o = i * s.ambient_dim;
        let x = &s.points[o..o + s.ambient_dim];
        let g = &grad[o..o + s.ambient_dim];

        // Lorentz inner product
        let inner = -x[0] * g[0] + x[1] * g[1] + x[2] * g[2];
        assert!(
            inner.abs() < 1e-10,
            "Point {i}: <grad, x>_L = {inner} (should be 0)"
        );
    }
}

#[test]
fn test_sphere_gradient_is_tangent() {
    let s = GradientTestSetup::new(1.0, 5, 2);
    let grad = kl_gradient(
        s.manifold.as_ref(),
        &s.points,
        &s.q,
        &s.p,
        &s.distances,
        s.n_points,
        s.ambient_dim,
    );

    // For each point, check <grad_i, x_i> = 0
    for i in 0..s.n_points {
        let o = i * s.ambient_dim;
        let x = &s.points[o..o + s.ambient_dim];
        let g = &grad[o..o + s.ambient_dim];

        let inner: f64 = x.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
        assert!(
            inner.abs() < 1e-10,
            "Point {i}: <grad, x> = {inner} (should be 0)"
        );
    }
}

#[test]
fn test_hyperboloid_gradient_is_tangent_different_curvatures() {
    for &k in &[-0.5, -2.0, -0.1] {
        let s = GradientTestSetup::new(k, 4, 2);
        let grad = kl_gradient(
            s.manifold.as_ref(),
            &s.points,
            &s.q,
            &s.p,
            &s.distances,
            s.n_points,
            s.ambient_dim,
        );

        for i in 0..s.n_points {
            let o = i * s.ambient_dim;
            let x = &s.points[o..o + s.ambient_dim];
            let g = &grad[o..o + s.ambient_dim];

            let mut inner = -x[0] * g[0];
            for d in 1..s.ambient_dim {
                inner += x[d] * g[d];
            }
            assert!(
                inner.abs() < 1e-9,
                "k={k}, point {i}: <grad, x>_L = {inner}"
            );
        }
    }
}

#[test]
fn test_sphere_gradient_is_tangent_different_curvatures() {
    for &k in &[0.5, 2.0, 0.1] {
        let s = GradientTestSetup::new(k, 4, 2);
        let grad = kl_gradient(
            s.manifold.as_ref(),
            &s.points,
            &s.q,
            &s.p,
            &s.distances,
            s.n_points,
            s.ambient_dim,
        );

        for i in 0..s.n_points {
            let o = i * s.ambient_dim;
            let x = &s.points[o..o + s.ambient_dim];
            let g = &grad[o..o + s.ambient_dim];

            let inner: f64 = x.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
            assert!(inner.abs() < 1e-9, "k={k}, point {i}: <grad, x> = {inner}");
        }
    }
}

// -----------------------------------------------------------------------
// Test: finite-difference validation of the Riemannian gradient
//
// For a function f on a manifold, the Riemannian gradient satisfies:
//   <grad_f, v>_R = d/dt f(exp_x(tv)) |_{t=0}
// We check this via central differences.
// -----------------------------------------------------------------------

/// Compute the KL loss at a given set of points.
fn loss_at(
    manifold: &dyn Manifold,
    points: &[f64],
    p: &[f64],
    n_points: usize,
    ambient_dim: usize,
) -> f64 {
    let (q, _) = compute_q_matrix_with_distances(manifold, points, n_points, ambient_dim, 1.0);
    kl_loss(&q, p, n_points)
}

/// Generate a random-ish tangent vector at point i.
fn make_tangent_vector(
    manifold: &dyn Manifold,
    points: &[f64],
    n_points: usize,
    ambient_dim: usize,
    point_idx: usize,
    seed: usize,
) -> Vec<f64> {
    // Create a full-size gradient-like vector (zero everywhere except at point_idx)
    let mut v = vec![0.0; n_points * ambient_dim];
    let o = point_idx * ambient_dim;

    // Fill with pseudo-random values
    for d in 0..ambient_dim {
        v[o + d] = ((seed * 7 + d * 13 + 37) as f64 % 97.0 - 48.5) / 48.5 * 0.01;
    }

    // Project to tangent space
    manifold.project_to_tangent(points, &mut v, n_points, ambient_dim);

    v
}

/// Riemannian inner product of two tangent vectors at a point.
fn riemannian_inner(
    curvature: f64,
    _points: &[f64],
    u: &[f64],
    v: &[f64],
    point_idx: usize,
    ambient_dim: usize,
) -> f64 {
    let o = point_idx * ambient_dim;
    if curvature < 0.0 {
        // Lorentzian inner product
        let mut inner = -u[o] * v[o];
        for d in 1..ambient_dim {
            inner += u[o + d] * v[o + d];
        }
        inner
    } else {
        // Euclidean inner product (for sphere and Euclidean)
        let mut inner = 0.0;
        for d in 0..ambient_dim {
            inner += u[o + d] * v[o + d];
        }
        inner
    }
}

fn finite_difference_check(curvature: f64) {
    let n_points = 4;
    let embed_dim = 2;
    let s = GradientTestSetup::new(curvature, n_points, embed_dim);
    let ambient_dim = s.ambient_dim;

    let grad = kl_gradient(
        s.manifold.as_ref(),
        &s.points,
        &s.q,
        &s.p,
        &s.distances,
        n_points,
        ambient_dim,
    );

    let eps = 1e-5;

    for i in 0..n_points {
        // Test with a few different tangent vectors
        for seed in 0..3 {
            let v = make_tangent_vector(
                s.manifold.as_ref(),
                &s.points,
                n_points,
                ambient_dim,
                i,
                seed + i * 10,
            );

            // f(exp_x(eps*v))
            let mut pts_plus = s.points.clone();
            let v_scaled_plus: Vec<f64> = v.iter().map(|&x| x * eps).collect();
            s.manifold
                .exp_map(&mut pts_plus, &v_scaled_plus, n_points, ambient_dim);
            let loss_plus = loss_at(s.manifold.as_ref(), &pts_plus, &s.p, n_points, ambient_dim);

            // f(exp_x(-eps*v))
            let mut pts_minus = s.points.clone();
            let v_scaled_minus: Vec<f64> = v.iter().map(|&x| -x * eps).collect();
            s.manifold
                .exp_map(&mut pts_minus, &v_scaled_minus, n_points, ambient_dim);
            let loss_minus = loss_at(s.manifold.as_ref(), &pts_minus, &s.p, n_points, ambient_dim);

            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            let analytic = riemannian_inner(curvature, &s.points, &grad, &v, i, ambient_dim);

            let abs_err = (fd - analytic).abs();
            let rel_err = abs_err / (analytic.abs().max(fd.abs()).max(1e-10));

            assert!(
                rel_err < 1e-4 || abs_err < 1e-8,
                "k={curvature}, point {i}, seed {seed}: fd={fd:.8e}, analytic={analytic:.8e}, rel_err={rel_err:.4e}"
            );
        }
    }
}

#[test]
fn test_gradient_finite_diff_euclidean() {
    finite_difference_check(0.0);
}

#[test]
fn test_gradient_finite_diff_hyperboloid() {
    finite_difference_check(-1.0);
}

#[test]
fn test_gradient_finite_diff_hyperboloid_k05() {
    finite_difference_check(-0.5);
}

#[test]
fn test_gradient_finite_diff_sphere() {
    finite_difference_check(1.0);
}

#[test]
fn test_gradient_finite_diff_sphere_k05() {
    finite_difference_check(0.5);
}

// -----------------------------------------------------------------------
// Test: Euclidean gradient is unchanged from original formula
// -----------------------------------------------------------------------

#[test]
fn test_euclidean_gradient_matches_original() {
    let s = GradientTestSetup::new(0.0, 5, 2);

    let grad = kl_gradient(
        s.manifold.as_ref(),
        &s.points,
        &s.q,
        &s.p,
        &s.distances,
        s.n_points,
        s.ambient_dim,
    );

    // Compute using the original formula directly
    let mut expected = vec![0.0; s.n_points * s.ambient_dim];
    for i in 0..s.n_points {
        for j in 0..s.n_points {
            if i == j {
                continue;
            }
            let idx = i * s.n_points + j;
            let d = s.distances[idx];
            let factor = 4.0 * (s.p[idx] - s.q[idx]) / (1.0 + d * d);
            for dim in 0..s.ambient_dim {
                let diff = s.points[i * s.ambient_dim + dim] - s.points[j * s.ambient_dim + dim];
                expected[i * s.ambient_dim + dim] += factor * diff;
            }
        }
    }

    for k in 0..grad.len() {
        assert!(
            (grad[k] - expected[k]).abs() < 1e-12,
            "Component {k}: got {}, expected {}",
            grad[k],
            expected[k]
        );
    }
}

// -----------------------------------------------------------------------
// Test: gradient is non-zero for non-trivial P != Q
// -----------------------------------------------------------------------

#[test]
fn test_gradient_nonzero_hyperboloid() {
    let s = GradientTestSetup::new(-1.0, 4, 2);
    let grad = kl_gradient(
        s.manifold.as_ref(),
        &s.points,
        &s.q,
        &s.p,
        &s.distances,
        s.n_points,
        s.ambient_dim,
    );

    let norm_sq: f64 = grad.iter().map(|x| x * x).sum();
    assert!(norm_sq > 1e-15, "Gradient should be non-zero for P != Q");
}

#[test]
fn test_gradient_nonzero_sphere() {
    let s = GradientTestSetup::new(1.0, 4, 2);
    let grad = kl_gradient(
        s.manifold.as_ref(),
        &s.points,
        &s.q,
        &s.p,
        &s.distances,
        s.n_points,
        s.ambient_dim,
    );

    let norm_sq: f64 = grad.iter().map(|x| x * x).sum();
    assert!(norm_sq > 1e-15, "Gradient should be non-zero for P != Q");
}

// -----------------------------------------------------------------------
// Test: higher embed_dim (3D embedding)
// -----------------------------------------------------------------------

#[test]
fn test_gradient_finite_diff_hyperboloid_3d() {
    finite_difference_check_dim(-1.0, 3);
}

#[test]
fn test_gradient_finite_diff_sphere_3d() {
    finite_difference_check_dim(1.0, 3);
}

fn finite_difference_check_dim(curvature: f64, embed_dim: usize) {
    let n_points = 4;
    let s = GradientTestSetup::new(curvature, n_points, embed_dim);
    let ambient_dim = s.ambient_dim;

    let grad = kl_gradient(
        s.manifold.as_ref(),
        &s.points,
        &s.q,
        &s.p,
        &s.distances,
        n_points,
        ambient_dim,
    );

    let eps = 1e-5;

    for i in 0..n_points {
        let v = make_tangent_vector(
            s.manifold.as_ref(),
            &s.points,
            n_points,
            ambient_dim,
            i,
            i * 7,
        );

        let mut pts_plus = s.points.clone();
        let v_plus: Vec<f64> = v.iter().map(|&x| x * eps).collect();
        s.manifold
            .exp_map(&mut pts_plus, &v_plus, n_points, ambient_dim);
        let loss_plus = loss_at(s.manifold.as_ref(), &pts_plus, &s.p, n_points, ambient_dim);

        let mut pts_minus = s.points.clone();
        let v_minus: Vec<f64> = v.iter().map(|&x| -x * eps).collect();
        s.manifold
            .exp_map(&mut pts_minus, &v_minus, n_points, ambient_dim);
        let loss_minus = loss_at(s.manifold.as_ref(), &pts_minus, &s.p, n_points, ambient_dim);

        let fd = (loss_plus - loss_minus) / (2.0 * eps);
        let analytic = riemannian_inner(curvature, &s.points, &grad, &v, i, ambient_dim);

        let abs_err = (fd - analytic).abs();
        let rel_err = abs_err / (analytic.abs().max(fd.abs()).max(1e-10));

        assert!(
            rel_err < 1e-4 || abs_err < 1e-8,
            "k={curvature}, dim={embed_dim}, point {i}: fd={fd:.8e}, analytic={analytic:.8e}, rel_err={rel_err:.4e}"
        );
    }
}
