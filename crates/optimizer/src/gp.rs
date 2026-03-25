use std::f64::consts::PI;

use fitting_core::synthetic_data::Rng;

use crate::search_space::{OptimizeDirection, SearchSpace, TrialConfig};

// ─── Public API ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Trial {
    pub config: TrialConfig,
    pub metric: f64,
}

pub struct GpOptimizer {
    trials: Vec<Trial>,
    direction: OptimizeDirection,
    n_ei_candidates: usize,
}

impl GpOptimizer {
    pub fn new(space: SearchSpace) -> Self {
        Self {
            trials: Vec::new(),
            direction: space.direction,
            n_ei_candidates: 1000,
        }
    }

    pub fn observe(&mut self, config: TrialConfig, metric: f64) {
        self.trials.push(Trial { config, metric });
    }

    /// Returns the next config to evaluate.
    /// Uses random search for the first 5 trials, then switches to GP-guided EI.
    pub fn suggest(&self, rng: &mut Rng) -> TrialConfig {
        const N_INIT: usize = 5;
        if self.trials.len() < N_INIT {
            return TrialConfig::random(rng);
        }

        let gp = GpModel::fit(&self.trials, self.direction);

        let mut best_config = TrialConfig::random(rng);
        let mut best_ei = f64::MIN;

        for _ in 0..self.n_ei_candidates {
            let candidate = if rng.uniform() < 0.3 {
                let idx =
                    (rng.uniform() * self.trials.len() as f64) as usize % self.trials.len();
                self.trials[idx].config.mutate(rng)
            } else {
                TrialConfig::random(rng)
            };

            let ei = gp.ei(&candidate);
            if ei > best_ei {
                best_ei = ei;
                best_config = candidate;
            }
        }

        self.local_search(best_config, rng, &gp)
    }

    pub fn best_trial(&self) -> f64 {
        if self.trials.is_empty() {
            return match self.direction {
                OptimizeDirection::Maximize => f64::MIN,
                OptimizeDirection::Minimize => f64::MAX,
            };
        }
        match self.direction {
            OptimizeDirection::Maximize => {
                self.trials.iter().map(|t| t.metric).fold(f64::MIN, f64::max)
            }
            OptimizeDirection::Minimize => {
                self.trials.iter().map(|t| t.metric).fold(f64::MAX, f64::min)
            }
        }
    }

    pub fn best_config(&self) -> Option<&TrialConfig> {
        if self.trials.is_empty() {
            return None;
        }
        match self.direction {
            OptimizeDirection::Maximize => self
                .trials
                .iter()
                .max_by(|a, b| a.metric.partial_cmp(&b.metric).unwrap())
                .map(|t| &t.config),
            OptimizeDirection::Minimize => self
                .trials
                .iter()
                .min_by(|a, b| a.metric.partial_cmp(&b.metric).unwrap())
                .map(|t| &t.config),
        }
    }

    fn local_search(&self, initial: TrialConfig, rng: &mut Rng, gp: &GpModel) -> TrialConfig {
        let mut current = initial;
        let mut current_ei = gp.ei(&current);

        for _ in 0..50 {
            let mutated = current.mutate(rng);
            let ei = gp.ei(&mutated);
            if ei > current_ei {
                current = mutated;
                current_ei = ei;
            }
        }
        current
    }
}

// ─── GP model ─────────────────────────────────────────────────────────────────

/// Gaussian process surrogate fitted to the observed trials.
///
/// Uses an RBF kernel with length-scale set by the median heuristic.
/// Inputs are log-transformed where appropriate and then standardized.
/// Outputs are standardized and flipped so we always maximize.
///
/// Note: fitting costs O(n³) via Cholesky, so this scales to a few hundred
/// trials comfortably but will slow down for very long runs.
struct GpModel {
    xs_norm: Vec<Vec<f64>>,
    x_means: Vec<f64>,
    x_stds: Vec<f64>,
    chol: Vec<f64>, // flat row-major lower-triangular Cholesky factor
    alpha: Vec<f64>, // K⁻¹ y
    f_best_norm: f64,
    length_scale: f64,
    n: usize,
}

// Observation noise added to the kernel diagonal for numerical stability.
const NOISE: f64 = 1e-4;
// Exploration bonus in the EI formula.
const XI: f64 = 0.01;

impl GpModel {
    fn fit(trials: &[Trial], direction: OptimizeDirection) -> Self {
        // Map configs to a GP-friendly input space (log-scale for log-uniform params).
        let raw_xs: Vec<Vec<f64>> =
            trials.iter().map(|t| config_to_gp_input(&t.config)).collect();

        // Flip sign for minimization so we always maximize in GP space.
        let ys: Vec<f64> = trials
            .iter()
            .map(|t| match direction {
                OptimizeDirection::Maximize => t.metric,
                OptimizeDirection::Minimize => -t.metric,
            })
            .collect();

        // Standardize inputs.
        let (x_means, x_stds) = compute_normalization(&raw_xs);
        let xs_norm: Vec<Vec<f64>> =
            raw_xs.iter().map(|x| standardize(x, &x_means, &x_stds)).collect();

        // Standardize outputs.
        let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let y_var = ys.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / ys.len() as f64;
        let y_std = y_var.sqrt().max(1e-8);
        let ys_norm: Vec<f64> = ys.iter().map(|&y| (y - y_mean) / y_std).collect();

        // Length-scale: median of pairwise distances (median heuristic).
        let length_scale = median_heuristic(&xs_norm);

        let n = xs_norm.len();
        let k_mat = build_kernel_matrix(&xs_norm, length_scale, NOISE);
        let chol = cholesky(&k_mat, n);
        let alpha = chol_solve(&chol, &ys_norm, n);
        let f_best_norm = ys_norm.iter().cloned().fold(f64::MIN, f64::max);

        Self { xs_norm, x_means, x_stds, chol, alpha, f_best_norm, length_scale, n }
    }

    /// Predict GP posterior mean and standard deviation at a normalized input.
    fn predict(&self, x_norm: &[f64]) -> (f64, f64) {
        // k_* = [k(x_i, x*)] for all observed x_i
        let k_star: Vec<f64> = self
            .xs_norm
            .iter()
            .map(|xi| rbf(xi, x_norm, self.length_scale))
            .collect();

        // Posterior mean: k_*ᵀ α
        let mu: f64 = k_star.iter().zip(self.alpha.iter()).map(|(k, a)| k * a).sum();

        // Posterior variance: k(x*,x*) - k_*ᵀ K⁻¹ k_*
        // Since k(x*,x*) = 1 for RBF, variance = 1 - ‖L⁻¹ k_*‖²
        let v = forward_sub(&self.chol, &k_star, self.n);
        let v_sq: f64 = v.iter().map(|x| x * x).sum();
        let sigma = (1.0 - v_sq).max(0.0).sqrt();

        (mu, sigma)
    }

    /// Expected Improvement for a given (unnormalized) candidate config.
    fn ei(&self, config: &TrialConfig) -> f64 {
        let x_raw = config_to_gp_input(config);
        let x_norm = standardize(&x_raw, &self.x_means, &self.x_stds);
        let (mu, sigma) = self.predict(&x_norm);
        expected_improvement(mu, sigma, self.f_best_norm, XI)
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Map a config to a GP input vector.
/// Log-transform lr and perp (log-uniform params) so the GP kernel sees a
/// roughly uniform input distribution on all dimensions.
fn config_to_gp_input(config: &TrialConfig) -> Vec<f64> {
    vec![
        config.learning_rate.ln(),
        config.perplexity.ln(),
        config.momentum_main,
        config.n_iterations as f64,
        config.early_exaggeration_iterations as f64,
        config.scaling_loss as f64,
        config.centering_weight,
        config.global_loss_weight,
        config.norm_loss_weight,
    ]
}

fn compute_normalization(xs: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let dim = xs[0].len();
    let n = xs.len() as f64;
    let mut means = vec![0.0; dim];
    for x in xs {
        for (d, &v) in x.iter().enumerate() {
            means[d] += v;
        }
    }
    for m in &mut means {
        *m /= n;
    }
    let mut stds = vec![0.0; dim];
    for x in xs {
        for (d, &v) in x.iter().enumerate() {
            stds[d] += (v - means[d]).powi(2);
        }
    }
    for s in &mut stds {
        *s = (*s / n).sqrt().max(1e-8);
    }
    (means, stds)
}

fn standardize(x: &[f64], means: &[f64], stds: &[f64]) -> Vec<f64> {
    x.iter().enumerate().map(|(d, &v)| (v - means[d]) / stds[d]).collect()
}

/// Median of pairwise squared distances, then l = sqrt(median / 2).
fn median_heuristic(xs: &[Vec<f64>]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 1.0;
    }
    let mut sq_dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = xs[i].iter().zip(xs[j].iter()).map(|(a, b)| (a - b).powi(2)).sum();
            sq_dists.push(d);
        }
    }
    sq_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sq_dists[sq_dists.len() / 2];
    if median < 1e-10 { 1.0 } else { (median / 2.0).sqrt() }
}

/// RBF (squared exponential) kernel: exp(−‖x − y‖² / 2l²).
fn rbf(x: &[f64], y: &[f64], length_scale: f64) -> f64 {
    let sq_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    (-sq_dist / (2.0 * length_scale * length_scale)).exp()
}

fn build_kernel_matrix(xs: &[Vec<f64>], length_scale: f64, noise: f64) -> Vec<f64> {
    let n = xs.len();
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] = rbf(&xs[i], &xs[j], length_scale);
        }
        k[i * n + i] += noise;
    }
    k
}

/// Cholesky decomposition: returns lower-triangular L (flat row-major) with A = L Lᵀ.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                // Clamp to prevent sqrt(negative) from floating-point drift.
                s.max(1e-12).sqrt()
            } else {
                s / l[j * n + j]
            };
        }
    }
    l
}

/// Forward substitution: solve L y = b.
fn forward_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|j| l[i * n + j] * y[j]).sum();
        y[i] = (b[i] - s) / l[i * n + i];
    }
    y
}

/// Backward substitution: solve Lᵀ x = y.
fn backward_sub(l: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = ((i + 1)..n).map(|j| l[j * n + i] * x[j]).sum();
        x[i] = (y[i] - s) / l[i * n + i];
    }
    x
}

/// Solve A x = b given Cholesky factor L of A.
fn chol_solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let y = forward_sub(l, b, n);
    backward_sub(l, &y, n)
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF via Abramowitz & Stegun polynomial approximation.
/// Maximum absolute error ≈ 7.5 × 10⁻⁸.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let p = 1.0 - normal_pdf(x) * poly;
    if x >= 0.0 { p } else { 1.0 - p }
}

/// Expected Improvement: EI(x) = (μ − f* − ξ) Φ(z) + σ φ(z), z = (μ − f* − ξ) / σ.
fn expected_improvement(mu: f64, sigma: f64, f_best: f64, xi: f64) -> f64 {
    if sigma < 1e-10 {
        return 0.0;
    }
    let z = (mu - f_best - xi) / sigma;
    (mu - f_best - xi) * normal_cdf(z) + sigma * normal_pdf(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fitting_core::synthetic_data::Rng;

    fn maximize_space() -> SearchSpace {
        SearchSpace { direction: OptimizeDirection::Maximize }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn make_config(lr: f64, perp: f64) -> TrialConfig {
        TrialConfig {
            learning_rate: lr,
            perplexity: perp,
            momentum_main: 0.8,
            n_iterations: 300,
            early_exaggeration_iterations: 100,
            scaling_loss: 0,
            centering_weight: 0.0,
            global_loss_weight: 0.0,
            norm_loss_weight: 0.0,
        }
    }

    // ─── normal_pdf ───────────────────────────────────────────────────────────

    #[test]
    fn test_normal_pdf_at_zero() {
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!(close(normal_pdf(0.0), expected, 1e-10));
    }

    #[test]
    fn test_normal_pdf_symmetric() {
        assert!(close(normal_pdf(1.5), normal_pdf(-1.5), 1e-15));
        assert!(close(normal_pdf(0.3), normal_pdf(-0.3), 1e-15));
    }

    #[test]
    fn test_normal_pdf_decreases_from_zero() {
        assert!(normal_pdf(0.0) > normal_pdf(1.0));
        assert!(normal_pdf(1.0) > normal_pdf(2.0));
    }

    #[test]
    fn test_normal_pdf_always_positive() {
        for x in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            assert!(normal_pdf(x) > 0.0);
        }
    }

    // ─── normal_cdf ───────────────────────────────────────────────────────────

    #[test]
    fn test_normal_cdf_at_zero() {
        assert!(close(normal_cdf(0.0), 0.5, 1e-6));
    }

    #[test]
    fn test_normal_cdf_known_values() {
        assert!(close(normal_cdf(1.0), 0.841345, 1e-5));
        assert!(close(normal_cdf(-1.0), 0.158655, 1e-5));
        assert!(close(normal_cdf(2.0), 0.977250, 1e-5));
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        for x in [0.5, 1.0, 2.0, 3.0] {
            assert!(close(normal_cdf(x) + normal_cdf(-x), 1.0, 1e-6));
        }
    }

    #[test]
    fn test_normal_cdf_monotone() {
        let mut prev = normal_cdf(-4.0);
        for i in -3..=4 {
            let p = normal_cdf(i as f64);
            assert!(p > prev);
            prev = p;
        }
    }

    #[test]
    fn test_normal_cdf_tails() {
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }

    // ─── expected_improvement ─────────────────────────────────────────────────

    #[test]
    fn test_ei_zero_when_sigma_zero() {
        assert_eq!(expected_improvement(1.0, 0.0, 0.0, 0.0), 0.0);
        assert_eq!(expected_improvement(1.0, 1e-11, 0.0, 0.0), 0.0);
    }

    #[test]
    fn test_ei_nonnegative() {
        for mu in [-2.0, 0.0, 1.0, 2.0] {
            for sigma in [0.1, 0.5, 1.0] {
                assert!(expected_improvement(mu, sigma, 0.0, 0.01) >= 0.0);
            }
        }
    }

    #[test]
    fn test_ei_larger_for_better_mu() {
        let ei_good = expected_improvement(2.0, 0.5, 0.0, 0.01);
        let ei_bad = expected_improvement(-2.0, 0.5, 0.0, 0.01);
        assert!(ei_good > ei_bad);
    }

    #[test]
    fn test_ei_near_zero_far_below_best() {
        let ei = expected_improvement(-10.0, 0.1, 0.0, 0.01);
        assert!(ei < 1e-6);
    }

    #[test]
    fn test_ei_larger_for_higher_uncertainty() {
        // Same mu below f_best — higher sigma should give more EI (exploration bonus).
        let ei_low = expected_improvement(-0.5, 0.1, 0.0, 0.01);
        let ei_high = expected_improvement(-0.5, 1.0, 0.0, 0.01);
        assert!(ei_high > ei_low);
    }

    // ─── rbf ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_rbf_self_is_one() {
        let x = vec![1.0, 2.0, 3.0];
        assert!(close(rbf(&x, &x, 1.0), 1.0, 1e-15));
    }

    #[test]
    fn test_rbf_symmetric() {
        let x = vec![1.0, 0.0];
        let y = vec![0.3, 0.7];
        assert!(close(rbf(&x, &y, 1.0), rbf(&y, &x, 1.0), 1e-15));
    }

    #[test]
    fn test_rbf_decreases_with_distance() {
        let origin = vec![0.0];
        let near = vec![0.5];
        let far = vec![2.0];
        assert!(rbf(&origin, &near, 1.0) > rbf(&origin, &far, 1.0));
    }

    #[test]
    fn test_rbf_larger_length_scale_slower_decay() {
        let x = vec![0.0];
        let y = vec![1.0];
        assert!(rbf(&x, &y, 2.0) > rbf(&x, &y, 0.5));
    }

    #[test]
    fn test_rbf_known_value() {
        // k([0], [1], l=1) = exp(-0.5)
        let val = rbf(&[0.0], &[1.0], 1.0);
        assert!(close(val, (-0.5_f64).exp(), 1e-12));
    }

    // ─── cholesky ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cholesky_recovers_matrix() {
        // A = [[4, 2], [2, 3]], A = L Lᵀ
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2);
        let mut recon = [0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    recon[i * 2 + j] += l[i * 2 + k] * l[j * 2 + k];
                }
            }
        }
        for idx in 0..4 {
            assert!(close(recon[idx], a[idx], 1e-10), "mismatch at index {idx}");
        }
    }

    #[test]
    fn test_cholesky_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let l = cholesky(&a, 2);
        assert!(close(l[0], 1.0, 1e-10));
        assert!(close(l[1], 0.0, 1e-10));
        assert!(close(l[2], 0.0, 1e-10));
        assert!(close(l[3], 1.0, 1e-10));
    }

    #[test]
    fn test_cholesky_lower_triangular() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2);
        // Upper triangle (above diagonal) must be zero.
        assert!(close(l[1], 0.0, 1e-15)); // L[0][1]
    }

    // ─── chol_solve ───────────────────────────────────────────────────────────

    #[test]
    fn test_chol_solve_identity() {
        let l = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 5.0];
        let x = chol_solve(&l, &b, 2);
        assert!(close(x[0], 3.0, 1e-10));
        assert!(close(x[1], 5.0, 1e-10));
    }

    #[test]
    fn test_chol_solve_known_system() {
        // A = [[4,2],[2,3]], b = [2,1]
        // A⁻¹ = (1/8)[[3,-2],[-2,4]], so x = (1/8)[4, 0] = [0.5, 0.0]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2);
        let x = chol_solve(&l, &[2.0, 1.0], 2);
        assert!(close(x[0], 0.5, 1e-8));
        assert!(close(x[1], 0.0, 1e-8));
    }

    #[test]
    fn test_chol_solve_residual_near_zero() {
        // Verify A * x ≈ b for a 3×3 case.
        let a = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
        let b = vec![1.0, 2.0, 3.0];
        let l = cholesky(&a, 3);
        let x = chol_solve(&l, &b, 3);
        for i in 0..3 {
            let ax_i: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            assert!(close(ax_i, b[i], 1e-8), "residual at row {i}: {}", ax_i - b[i]);
        }
    }

    // ─── median_heuristic ─────────────────────────────────────────────────────

    #[test]
    fn test_median_heuristic_single_point() {
        assert_eq!(median_heuristic(&[vec![1.0, 2.0]]), 1.0);
    }

    #[test]
    fn test_median_heuristic_two_points() {
        // sq_dist = (2-0)² = 4, l = sqrt(4/2) = sqrt(2)
        let xs = vec![vec![0.0], vec![2.0]];
        assert!(close(median_heuristic(&xs), 2.0_f64.sqrt(), 1e-10));
    }

    #[test]
    fn test_median_heuristic_identical_points() {
        let xs = vec![vec![1.0], vec![1.0], vec![1.0]];
        assert_eq!(median_heuristic(&xs), 1.0);
    }

    #[test]
    fn test_median_heuristic_positive() {
        let mut rng = Rng::new(0);
        let xs: Vec<Vec<f64>> = (0..10).map(|_| vec![rng.uniform(), rng.uniform()]).collect();
        assert!(median_heuristic(&xs) > 0.0);
    }

    // ─── compute_normalization / standardize ──────────────────────────────────

    #[test]
    fn test_normalization_means_and_stds() {
        let xs = vec![vec![0.0, 0.0], vec![2.0, 4.0]];
        let (means, stds) = compute_normalization(&xs);
        assert!(close(means[0], 1.0, 1e-10));
        assert!(close(means[1], 2.0, 1e-10));
        assert!(close(stds[0], 1.0, 1e-10));
        assert!(close(stds[1], 2.0, 1e-10));
    }

    #[test]
    fn test_normalization_constant_dim_clamped() {
        let xs = vec![vec![5.0], vec![5.0], vec![5.0]];
        let (means, stds) = compute_normalization(&xs);
        assert!(close(means[0], 5.0, 1e-10));
        assert!(stds[0] <= 1e-8 + 1e-12); // clamped to minimum
    }

    #[test]
    fn test_standardize_maps_to_plus_minus_one() {
        let xs = vec![vec![0.0], vec![2.0]];
        let (means, stds) = compute_normalization(&xs);
        assert!(close(standardize(&[0.0], &means, &stds)[0], -1.0, 1e-10));
        assert!(close(standardize(&[2.0], &means, &stds)[0], 1.0, 1e-10));
    }

    #[test]
    fn test_standardize_mean_at_zero() {
        let xs = vec![vec![1.0], vec![3.0], vec![5.0]];
        let (means, stds) = compute_normalization(&xs);
        let z: f64 = xs.iter().map(|x| standardize(x, &means, &stds)[0]).sum();
        assert!(close(z, 0.0, 1e-10));
    }

    // ─── config_to_gp_input ───────────────────────────────────────────────────

    #[test]
    fn test_config_to_gp_input_log_transforms_lr_and_perp() {
        let cfg = make_config(1.0_f64.exp(), 2.0_f64.exp()); // lr = e, perp = e²
        let v = config_to_gp_input(&cfg);
        assert!(close(v[0], 1.0, 1e-10)); // ln(e) = 1
        assert!(close(v[1], 2.0, 1e-10)); // ln(e²) = 2
    }

    #[test]
    fn test_config_to_gp_input_passthrough_fields() {
        let cfg = make_config(1.0, 10.0);
        let v = config_to_gp_input(&cfg);
        assert!(close(v[2], cfg.momentum_main, 1e-15));
        assert!(close(v[3], cfg.n_iterations as f64, 1e-15));
        assert!(close(v[4], cfg.early_exaggeration_iterations as f64, 1e-15));
        assert!(close(v[5], cfg.scaling_loss as f64, 1e-15));
        assert!(close(v[6], cfg.centering_weight, 1e-15));
        assert!(close(v[7], cfg.global_loss_weight, 1e-15));
        assert!(close(v[8], cfg.norm_loss_weight, 1e-15));
    }

    // ─── GpModel ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gp_predict_low_variance_near_observations() {
        // Posterior variance should be small at observed locations.
        let mut rng = Rng::new(42);
        let trials: Vec<Trial> = (0..10)
            .map(|_| Trial { config: TrialConfig::random(&mut rng), metric: rng.uniform() })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize);
        let x_raw = config_to_gp_input(&trials[0].config);
        let x_norm = standardize(&x_raw, &gp.x_means, &gp.x_stds);
        let (_, sigma) = gp.predict(&x_norm);
        assert!(sigma < 0.1, "expected low sigma near observed point, got {sigma}");
    }

    #[test]
    fn test_gp_predict_returns_finite_values() {
        let mut rng = Rng::new(7);
        let trials: Vec<Trial> = (0..8)
            .map(|_| Trial { config: TrialConfig::random(&mut rng), metric: rng.uniform() })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize);
        for _ in 0..20 {
            let x_raw = config_to_gp_input(&TrialConfig::random(&mut rng));
            let x_norm = standardize(&x_raw, &gp.x_means, &gp.x_stds);
            let (mu, sigma) = gp.predict(&x_norm);
            assert!(mu.is_finite(), "mu should be finite");
            assert!(sigma.is_finite() && sigma >= 0.0, "sigma should be finite and non-negative");
        }
    }

    #[test]
    fn test_gp_ei_nonnegative() {
        let mut rng = Rng::new(123);
        let trials: Vec<Trial> = (0..8)
            .map(|_| Trial { config: TrialConfig::random(&mut rng), metric: rng.uniform() })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize);
        for _ in 0..30 {
            let cfg = TrialConfig::random(&mut rng);
            assert!(gp.ei(&cfg) >= 0.0, "EI must be non-negative");
        }
    }

    // ─── GpOptimizer ──────────────────────────────────────────────────────────

    #[test]
    fn test_optimizer_best_trial_maximize() {
        let mut opt = GpOptimizer::new(maximize_space());
        opt.observe(make_config(1.0, 10.0), 0.5);
        opt.observe(make_config(10.0, 20.0), 0.9);
        opt.observe(make_config(5.0, 5.0), 0.3);
        assert!(close(opt.best_trial(), 0.9, 1e-10));
    }

    #[test]
    fn test_optimizer_best_config_is_highest_metric() {
        let mut opt = GpOptimizer::new(maximize_space());
        opt.observe(make_config(1.0, 10.0), 0.5);
        opt.observe(make_config(10.0, 20.0), 0.9);
        opt.observe(make_config(5.0, 5.0), 0.3);
        let best = opt.best_config().unwrap();
        assert!(close(best.learning_rate, 10.0, 1e-10));
    }

    #[test]
    fn test_optimizer_best_trial_empty() {
        let opt = GpOptimizer::new(maximize_space());
        assert_eq!(opt.best_trial(), f64::MIN); // maximize direction default
    }

    #[test]
    fn test_optimizer_suggest_random_phase() {
        // Fewer than N_INIT=5 trials → random search, must not panic.
        let mut opt = GpOptimizer::new(maximize_space());
        let mut rng = Rng::new(42);
        for _ in 0..4 {
            let cfg = opt.suggest(&mut rng);
            opt.observe(cfg, rng.uniform());
        }
        let _ = opt.suggest(&mut rng);
    }

    #[test]
    fn test_optimizer_suggest_gp_phase() {
        // N_INIT+ trials → GP-guided suggest must not panic and return a valid config.
        let mut opt = GpOptimizer::new(maximize_space());
        let mut rng = Rng::new(42);
        for i in 0..10 {
            let cfg = TrialConfig::random(&mut rng);
            opt.observe(cfg, i as f64 * 0.1);
        }
        let cfg = opt.suggest(&mut rng);
        assert!(cfg.learning_rate > 0.0);
        assert!(cfg.perplexity >= 2.0);
        assert!(cfg.n_iterations > 0);
    }

    #[test]
    fn test_optimizer_prefers_high_metric_region() {
        // After observing a clear winner, the GP should suggest configs that
        // resemble it more often than random. We verify EI is highest near the
        // winner, not near the low-metric configs.
        let mut opt = GpOptimizer::new(maximize_space());

        // Seed with diverse trials; one clear winner.
        let configs_and_metrics: &[(f64, f64, f64)] = &[
            (1.0, 5.0, 0.1),
            (200.0, 50.0, 0.15),
            (50.0, 10.0, 0.12),
            (0.5, 80.0, 0.08),
            (10.0, 20.0, 0.9), // clear winner
        ];
        for &(lr, perp, m) in configs_and_metrics {
            opt.observe(make_config(lr, perp), m);
        }

        let trials: Vec<Trial> = configs_and_metrics
            .iter()
            .map(|&(lr, perp, metric)| Trial { config: make_config(lr, perp), metric })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize);

        let ei_winner = gp.ei(&make_config(10.0, 20.0));
        let ei_loser = gp.ei(&make_config(0.5, 80.0));
        assert!(ei_winner >= ei_loser, "EI near winner ({ei_winner:.4}) should be >= loser ({ei_loser:.4})");
    }
}
