use crate::config::ScalingLossType;

/// Compute the radial scaling loss for hyperbolic embeddings.
///
/// This regularizes the radial spread of points on the hyperboloid by penalizing
/// geodesic distances from the origin. Only meaningful for curvature < 0; returns
/// zeros for other geometries.
///
/// Returns `(loss_value, ambient_space_gradient)`. The gradient must be projected
/// to the tangent space by the caller before adding to the Riemannian gradient.
pub fn compute(
    loss_type: ScalingLossType,
    points: &[f64],
    n_points: usize,
    ambient_dim: usize,
    radius: f64,
    curvature: f64,
) -> (f64, Vec<f64>) {
    if curvature >= 0.0 || matches!(loss_type, ScalingLossType::None) {
        return (0.0, vec![0.0; n_points * ambient_dim]);
    }

    let r = radius;
    let mut grad = vec![0.0; n_points * ambient_dim];

    match loss_type {
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
                    let dg_dx0 = 1.0 / (arg * arg - 1.0).max(1e-12).sqrt();
                    grad[i * ambient_dim] = 2.0 * excess * dg_dx0 / n_points as f64;
                }
            }
            loss /= n_points as f64;
            (loss, grad)
        }
        ScalingLossType::MeanDistance => {
            let n = n_points as f64;
            let mut loss = 0.0;
            for i in 0..n_points {
                let x0 = points[i * ambient_dim];
                let arg = (x0 / r).max(1.0 + 1e-7);
                let geo_dist = r * arg.acosh();
                loss += geo_dist;

                let dg_dx0 = 1.0 / (arg * arg - 1.0).max(1e-12).sqrt();
                grad[i * ambient_dim] = dg_dx0 / n;
            }
            loss /= n;
            (loss, grad)
        }
        ScalingLossType::Rms => {
            let n = n_points as f64;
            let mut sum_sq = 0.0;
            let mut geo_dists = Vec::with_capacity(n_points);
            let mut dg_dx0s = Vec::with_capacity(n_points);
            for i in 0..n_points {
                let x0 = points[i * ambient_dim];
                let arg = (x0 / r).max(1.0 + 1e-7);
                let geo_dist = r * arg.acosh();
                sum_sq += geo_dist * geo_dist;
                geo_dists.push(geo_dist);
                dg_dx0s.push(1.0 / (arg * arg - 1.0).max(1e-12).sqrt());
            }
            let rms = (sum_sq / n).sqrt();
            let loss = (rms - 1.0) * (rms - 1.0);
            if rms > 1e-12 {
                let coeff = 2.0 * (rms - 1.0) / (n * rms);
                for i in 0..n_points {
                    grad[i * ambient_dim] = coeff * geo_dists[i] * dg_dx0s[i];
                }
            }
            (loss, grad)
        }
        ScalingLossType::SoftplusBarrier => {
            let n = n_points as f64;
            let d_max = 3.0 * r;
            let mut loss = 0.0;
            for i in 0..n_points {
                let x0 = points[i * ambient_dim];
                let arg = (x0 / r).max(1.0 + 1e-7);
                let geo_dist = r * arg.acosh();
                let z = geo_dist - d_max;
                let sp = if z > 20.0 { z } else { (1.0 + z.exp()).ln() };
                loss += sp;

                let sig = 1.0 / (1.0 + (-z).exp());
                let dg_dx0 = 1.0 / (arg * arg - 1.0).max(1e-12).sqrt();
                grad[i * ambient_dim] = sig * dg_dx0 / n;
            }
            loss /= n;
            (loss, grad)
        }
        ScalingLossType::None => unreachable!(),
    }
}
