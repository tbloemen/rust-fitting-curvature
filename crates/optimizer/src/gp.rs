// gp.rs — Bayesian hyperparameter optimisation with a Gaussian process surrogate.
//
// All notation and algorithm choices follow:
//   Frazier, P.I. (2018). "A Tutorial on Bayesian Optimization."
//
// Key references used below:
//   §3    — GP regression: posterior mean and variance (Eq. 3)
//   §3.1  — Squared exponential (RBF) kernel
//   §3.2  — Maximum likelihood estimation (MLE) of hyperparameters
//   §4.1  — Expected Improvement acquisition function (Eq. 7–8)
//   Alg.1 — Basic BayesOpt loop with initial space-filling design

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

    /// Suggest the next configuration to evaluate.
    ///
    /// Implements Algorithm 1 of Frazier (2018):
    ///   1. Initial space-filling design (random) for the first N_INIT trials.
    ///   2. After N_INIT observations: fit GP surrogate, then return
    ///      argmax_x EI_n(x) solved via random candidates + local refinement (Eq. 9).
    pub fn suggest(&self, rng: &mut Rng) -> TrialConfig {
        const N_INIT: usize = 5;
        if self.trials.len() < N_INIT {
            return TrialConfig::random(rng);
        }

        // Fit GP to all n observations (§3, Eq. 3).
        let gp = GpModel::fit(&self.trials, self.direction);

        // Maximise EI over random candidates (Eq. 9).
        // We mix purely random points with perturbations of existing observations
        // to balance exploration and exploitation in the candidate set.
        let mut best_config = TrialConfig::random(rng);
        let mut best_ei = f64::MIN;

        for _ in 0..self.n_ei_candidates {
            let candidate = if rng.uniform() < 0.3 {
                let idx = (rng.uniform() * self.trials.len() as f64) as usize % self.trials.len();
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

        // Local refinement: greedy hill-climb on EI from the best candidate found.
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
            OptimizeDirection::Maximize => self
                .trials
                .iter()
                .map(|t| t.metric)
                .fold(f64::MIN, f64::max),
            OptimizeDirection::Minimize => self
                .trials
                .iter()
                .map(|t| t.metric)
                .fold(f64::MAX, f64::min),
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

// ─── GP surrogate (Frazier §3) ────────────────────────────────────────────────

/// Gaussian process surrogate fitted to the observed trials.
///
/// **Prior** (Frazier §3.1):
///   - Mean function: µ₀(x) = 0  (constant mean; after output standardisation the
///     MLE estimate of µ is 0, so this is exact).
///   - Kernel: squared exponential  k(x, x') = exp(−‖x − x'‖² / 2l²)
///
/// **Hyperparameter** (Frazier §3.2):
///   - Length-scale l is selected by maximising the log marginal likelihood (MLE)
///     over a log-spaced grid of candidates.
///
/// **Posterior** (Frazier Eq. 3):
///   - µₙ(x) = Σ₀(x, x₁:ₙ) Σ₀(x₁:ₙ, x₁:ₙ)⁻¹ f(x₁:ₙ)
///   - σₙ²(x) = Σ₀(x,x) − Σ₀(x, x₁:ₙ) Σ₀(x₁:ₙ, x₁:ₙ)⁻¹ Σ₀(x₁:ₙ, x)
///
/// Cholesky is used for stability; a small diagonal jitter of JITTER is added
/// as recommended in Frazier §3 (citing Rasmussen & Williams 2006).
struct GpModel {
    xs_norm: Vec<Vec<f64>>,
    x_means: Vec<f64>,
    x_stds: Vec<f64>,
    /// Cholesky factor L such that (K + jitter·I) = L Lᵀ, flat row-major.
    chol: Vec<f64>,
    /// α = K⁻¹ y, precomputed for fast posterior mean evaluation.
    alpha: Vec<f64>,
    /// f*_n: best standardised observed value (current incumbent).
    f_best_norm: f64,
    length_scale: f64,
    n: usize,
}

/// Diagonal noise added to the kernel matrix for numerical stability.
/// Frazier §3 recommends 10⁻⁶; 1e-4 is used here for robustness with small n.
const JITTER: f64 = 1e-4;

impl GpModel {
    fn fit(trials: &[Trial], direction: OptimizeDirection) -> Self {
        // Encode configs as real-valued GP inputs (log-transform log-uniform params).
        let raw_xs: Vec<Vec<f64>> = trials
            .iter()
            .map(|t| config_to_gp_input(&t.config))
            .collect();

        // Internally always maximise; flip sign for minimisation objectives.
        let ys: Vec<f64> = trials
            .iter()
            .map(|t| match direction {
                OptimizeDirection::Maximize => t.metric,
                OptimizeDirection::Minimize => -t.metric,
            })
            .collect();

        // Standardise inputs to zero mean, unit std per dimension.
        let (x_means, x_stds) = compute_normalization(&raw_xs);
        let xs_norm: Vec<Vec<f64>> = raw_xs
            .iter()
            .map(|x| standardize(x, &x_means, &x_stds))
            .collect();

        // Standardise outputs to zero mean, unit variance so that the RBF prior
        // variance k(x,x) = 1 matches the data scale (avoids a separate signal-
        // variance hyperparameter while still fitting the constant mean via MLE).
        let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
        let y_var = ys.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / ys.len() as f64;
        let y_std = y_var.sqrt().max(1e-8);
        let ys_norm: Vec<f64> = ys.iter().map(|&y| (y - y_mean) / y_std).collect();

        // Select length-scale by MLE over a log-spaced grid (Frazier §3.2).
        let length_scale = mle_length_scale(&xs_norm, &ys_norm);

        let n = xs_norm.len();
        let k_mat = build_kernel_matrix(&xs_norm, length_scale, JITTER);
        let chol = cholesky(&k_mat, n);
        let alpha = chol_solve(&chol, &ys_norm, n);
        let f_best_norm = ys_norm.iter().cloned().fold(f64::MIN, f64::max);

        Self {
            xs_norm,
            x_means,
            x_stds,
            chol,
            alpha,
            f_best_norm,
            length_scale,
            n,
        }
    }

    /// GP posterior mean µₙ(x) and standard deviation σₙ(x) at a normalised input x.
    ///
    /// From Frazier Eq. 3 (with µ₀ = 0):
    ///   µₙ(x) = k(x, X) α  where α = K⁻¹ y
    ///   σₙ²(x) = k(x,x) − ‖L⁻¹ k(x, X)ᵀ‖²     (k(x,x) = 1 for RBF)
    fn predict(&self, x_norm: &[f64]) -> (f64, f64) {
        // k* = [k(xᵢ, x*) for each training point xᵢ]
        let k_star: Vec<f64> = self
            .xs_norm
            .iter()
            .map(|xi| rbf(xi, x_norm, self.length_scale))
            .collect();

        let mu: f64 = k_star.iter().zip(&self.alpha).map(|(k, a)| k * a).sum();

        // σₙ²(x) = 1 − ‖v‖²  where v = L⁻¹ k*.
        let v = forward_sub(&self.chol, &k_star, self.n);
        let v_sq: f64 = v.iter().map(|x| x * x).sum();
        let sigma = (1.0 - v_sq).max(0.0).sqrt();

        (mu, sigma)
    }

    /// Expected Improvement for a candidate config (Frazier Eq. 7 / Eq. 8).
    fn ei(&self, config: &TrialConfig) -> f64 {
        let x_raw = config_to_gp_input(config);
        let x_norm = standardize(&x_raw, &self.x_means, &self.x_stds);
        let (mu, sigma) = self.predict(&x_norm);
        expected_improvement(mu, sigma, self.f_best_norm)
    }
}

// ─── MLE for length-scale (Frazier §3.2) ─────────────────────────────────────

/// Select the length-scale that maximises the log marginal likelihood (Frazier §3.2).
///
/// A log-spaced grid of N_GRID candidates in [L_MIN, L_MAX] is evaluated; the
/// highest-LML value is returned.  Grid search is appropriate because we are
/// optimising a single scalar hyperparameter and the LML is typically unimodal
/// in log l.
fn mle_length_scale(xs_norm: &[Vec<f64>], ys_norm: &[f64]) -> f64 {
    const N_GRID: usize = 20;
    const L_MIN: f64 = 0.05;
    const L_MAX: f64 = 10.0;

    (0..N_GRID)
        .map(|i| {
            let t = i as f64 / (N_GRID - 1) as f64;
            let l = (L_MIN.ln() + t * (L_MAX.ln() - L_MIN.ln())).exp();
            let lml = log_marginal_likelihood(xs_norm, ys_norm, l);
            (l, lml)
        })
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(l, _)| l)
        .unwrap_or(1.0)
}

/// Log marginal likelihood of a zero-mean GP with RBF kernel:
///
///   log p(y | X, l) = −½ yᵀ K(l)⁻¹ y  −  ½ log|K(l)|  −  n/2 log(2π)
///
/// Computed from the Cholesky factor L of K(l):
///   yᵀ K⁻¹ y = ‖L⁻¹ y‖²
///   log|K| = 2 Σᵢ log Lᵢᵢ
fn log_marginal_likelihood(xs_norm: &[Vec<f64>], ys_norm: &[f64], length_scale: f64) -> f64 {
    let n = xs_norm.len();
    let k = build_kernel_matrix(xs_norm, length_scale, JITTER);
    let l = cholesky(&k, n);

    // Data-fit term: yᵀ K⁻¹ y = ‖L⁻¹ y‖².
    let v = forward_sub(&l, ys_norm, n);
    let data_fit: f64 = v.iter().map(|a| a * a).sum();

    // Complexity penalty: log|K| = 2 Σᵢ log Lᵢᵢ.
    let log_det: f64 = (0..n).map(|i| l[i * n + i].max(1e-300).ln()).sum::<f64>() * 2.0;

    -0.5 * data_fit - 0.5 * log_det - 0.5 * n as f64 * (2.0 * PI).ln()
}

// ─── Input encoding ───────────────────────────────────────────────────────────

/// Encode a TrialConfig as a real-valued GP input vector.
///
/// Parameters that were sampled log-uniformly (learning_rate, perplexity_ratio)
/// are log-transformed so that the GP sees an approximately uniform distribution
/// over the input space — a standard preprocessing step for log-scale parameters.
fn config_to_gp_input(config: &TrialConfig) -> Vec<f64> {
    vec![
        config.learning_rate.ln(),
        config.perplexity_ratio.ln(),
        config.momentum_main,
        config.centering_weight,
        config.global_loss_weight,
        config.norm_loss_weight,
    ]
}

// ─── Normalisation helpers ────────────────────────────────────────────────────

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
    x.iter()
        .enumerate()
        .map(|(d, &v)| (v - means[d]) / stds[d])
        .collect()
}

// ─── Kernel (Frazier §3.1) ────────────────────────────────────────────────────

/// Squared exponential (Gaussian) kernel from Frazier §3.1:
///
///   Σ₀(x, x') = exp(−‖x − x'‖² / 2l²)
///
/// Uses a single isotropic length-scale l shared across all input dimensions.
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

// ─── Cholesky linear algebra (Frazier §3) ─────────────────────────────────────

/// Cholesky decomposition K = L Lᵀ.  Returns flat row-major lower-triangular L.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
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

/// Solve K x = b given the Cholesky factor L of K.
fn chol_solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let y = forward_sub(l, b, n);
    backward_sub(l, &y, n)
}

// ─── Normal distribution helpers ──────────────────────────────────────────────

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF via Abramowitz & Stegun polynomial approximation.
/// Maximum absolute error ≈ 7.5 × 10⁻⁸.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let p = 1.0 - normal_pdf(x) * poly;
    if x >= 0.0 { p } else { 1.0 - p }
}

// ─── EI acquisition function (Frazier §4.1, Eq. 7–8) ─────────────────────────

/// Expected Improvement at a point with posterior mean µ and std σ,
/// given the current best observed value f*.
///
/// From Frazier Eq. 7:  EI_n(x) = E_n[(f(x) − f*_n)⁺]
///
/// Closed-form evaluation (Jones et al. 1998, referenced in Frazier §4.1):
///   Let Δ = µ − f*,  z = Δ / σ
///   EI = Δ · Φ(z) + σ · φ(z)
///
/// This is non-negative and equals 0 when σ = 0 and µ ≤ f*.
pub fn expected_improvement(mu: f64, sigma: f64, f_best: f64) -> f64 {
    if sigma < 1e-10 {
        return (mu - f_best).max(0.0);
    }
    let delta = mu - f_best;
    let z = delta / sigma;
    delta * normal_cdf(z) + sigma * normal_pdf(z)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fitting_core::synthetic_data::Rng;

    fn maximize_space() -> SearchSpace {
        SearchSpace {
            direction: OptimizeDirection::Maximize,
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn make_config(lr: f64, perp_ratio: f64) -> TrialConfig {
        TrialConfig {
            learning_rate: lr,
            perplexity_ratio: perp_ratio,
            momentum_main: 0.8,
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
    //
    // Testing EI = Δ·Φ(Δ/σ) + σ·φ(Δ/σ)  (Frazier Eq. 7–8, Δ = µ − f*)

    #[test]
    fn test_ei_zero_sigma_below_best() {
        // σ=0, µ < f*: improvement is impossible → EI = 0.
        assert_eq!(expected_improvement(-1.0, 0.0, 0.0), 0.0);
        assert_eq!(expected_improvement(-1.0, 1e-11, 0.0), 0.0);
    }

    #[test]
    fn test_ei_zero_sigma_above_best() {
        // σ=0, µ > f*: improvement is certain → EI = µ − f*.
        assert!(close(expected_improvement(1.5, 0.0, 0.0), 1.5, 1e-10));
        assert!(close(expected_improvement(2.0, 1e-11, 0.0), 2.0, 1e-6));
    }

    #[test]
    fn test_ei_nonnegative() {
        for mu in [-2.0, 0.0, 1.0, 2.0] {
            for sigma in [0.1, 0.5, 1.0] {
                assert!(expected_improvement(mu, sigma, 0.0) >= 0.0);
            }
        }
    }

    #[test]
    fn test_ei_larger_for_better_mu() {
        let ei_good = expected_improvement(2.0, 0.5, 0.0);
        let ei_bad = expected_improvement(-2.0, 0.5, 0.0);
        assert!(ei_good > ei_bad);
    }

    #[test]
    fn test_ei_near_zero_far_below_best() {
        let ei = expected_improvement(-10.0, 0.1, 0.0);
        assert!(ei < 1e-6);
    }

    #[test]
    fn test_ei_larger_for_higher_uncertainty() {
        // Same µ below f* — higher σ should give more EI (exploration bonus).
        let ei_low = expected_improvement(-0.5, 0.1, 0.0);
        let ei_high = expected_improvement(-0.5, 1.0, 0.0);
        assert!(ei_high > ei_low);
    }

    #[test]
    fn test_ei_known_value() {
        // µ = f* = 0, σ = 1: EI = 0·Φ(0) + 1·φ(0) = φ(0) = 1/√(2π)
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!(close(expected_improvement(0.0, 1.0, 0.0), expected, 1e-10));
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
        // k([0], [1], l=1) = exp(−0.5)
        let val = rbf(&[0.0], &[1.0], 1.0);
        assert!(close(val, (-0.5_f64).exp(), 1e-12));
    }

    // ─── cholesky ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cholesky_recovers_matrix() {
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
        assert!(close(l[1], 0.0, 1e-15)); // L[0][1] must be zero
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
        // A = [[4,2],[2,3]], b = [2,1] → x = [0.5, 0.0]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2);
        let x = chol_solve(&l, &[2.0, 1.0], 2);
        assert!(close(x[0], 0.5, 1e-8));
        assert!(close(x[1], 0.0, 1e-8));
    }

    #[test]
    fn test_chol_solve_residual_near_zero() {
        let a = vec![6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
        let b = vec![1.0, 2.0, 3.0];
        let l = cholesky(&a, 3);
        let x = chol_solve(&l, &b, 3);
        for i in 0..3 {
            let ax_i: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            assert!(
                close(ax_i, b[i], 1e-8),
                "residual at row {i}: {}",
                ax_i - b[i]
            );
        }
    }

    // ─── log_marginal_likelihood ──────────────────────────────────────────────

    #[test]
    fn test_lml_is_finite() {
        let xs = vec![vec![0.0], vec![1.0], vec![2.0]];
        let ys = vec![-0.5, 0.3, 1.1];
        let lml = log_marginal_likelihood(&xs, &ys, 1.0);
        assert!(lml.is_finite(), "LML should be finite, got {lml}");
    }

    #[test]
    fn test_lml_prefers_good_length_scale() {
        // Generate data from an RBF with l=1; LML should be higher near l=1 than l=10.
        let xs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.2]).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (-(x[0] - 1.0).powi(2)).exp()).collect();
        let lml_good = log_marginal_likelihood(&xs, &ys, 0.5);
        let lml_bad = log_marginal_likelihood(&xs, &ys, 10.0);
        assert!(
            lml_good > lml_bad,
            "LML at l=0.5 ({lml_good:.3}) should exceed l=10 ({lml_bad:.3}) for smooth data"
        );
    }

    // ─── mle_length_scale ─────────────────────────────────────────────────────

    #[test]
    fn test_mle_length_scale_positive() {
        let mut rng = Rng::new(0);
        let xs: Vec<Vec<f64>> = (0..10)
            .map(|_| vec![rng.uniform(), rng.uniform()])
            .collect();
        let ys: Vec<f64> = (0..10).map(|_| rng.uniform()).collect();
        let l = mle_length_scale(&xs, &ys);
        assert!(l > 0.0, "length scale must be positive, got {l}");
    }

    #[test]
    fn test_mle_length_scale_in_grid_range() {
        let mut rng = Rng::new(7);
        let xs: Vec<Vec<f64>> = (0..8).map(|_| vec![rng.uniform()]).collect();
        let ys: Vec<f64> = (0..8).map(|_| rng.uniform()).collect();
        let l = mle_length_scale(&xs, &ys);
        assert!(
            l >= 0.05 && l <= 10.0,
            "l={l} should be within grid [0.05, 10]"
        );
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
        assert!(stds[0] <= 1e-8 + 1e-12);
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
        assert!(close(v[3], cfg.centering_weight, 1e-15));
        assert!(close(v[4], cfg.global_loss_weight, 1e-15));
        assert!(close(v[5], cfg.norm_loss_weight, 1e-15));
    }

    // ─── GpModel ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gp_predict_low_variance_near_observations() {
        // Posterior variance should be small at observed locations.
        let mut rng = Rng::new(42);
        let trials: Vec<Trial> = (0..10)
            .map(|_| Trial {
                config: TrialConfig::random(&mut rng),
                metric: rng.uniform(),
            })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize);
        let x_raw = config_to_gp_input(&trials[0].config);
        let x_norm = standardize(&x_raw, &gp.x_means, &gp.x_stds);
        let (_, sigma) = gp.predict(&x_norm);
        assert!(
            sigma < 0.15,
            "expected low sigma near observed point, got {sigma}"
        );
    }

    #[test]
    fn test_gp_predict_returns_finite_values() {
        let mut rng = Rng::new(7);
        let trials: Vec<Trial> = (0..8)
            .map(|_| Trial {
                config: TrialConfig::random(&mut rng),
                metric: rng.uniform(),
            })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize);
        for _ in 0..20 {
            let x_raw = config_to_gp_input(&TrialConfig::random(&mut rng));
            let x_norm = standardize(&x_raw, &gp.x_means, &gp.x_stds);
            let (mu, sigma) = gp.predict(&x_norm);
            assert!(mu.is_finite(), "mu should be finite");
            assert!(
                sigma.is_finite() && sigma >= 0.0,
                "sigma should be finite and non-negative"
            );
        }
    }

    #[test]
    fn test_gp_ei_nonnegative() {
        let mut rng = Rng::new(123);
        let trials: Vec<Trial> = (0..8)
            .map(|_| Trial {
                config: TrialConfig::random(&mut rng),
                metric: rng.uniform(),
            })
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
        assert_eq!(opt.best_trial(), f64::MIN);
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
        assert!(cfg.perplexity_ratio >= 0.0004);
        assert!(cfg.momentum_main >= 0.60);
    }

    #[test]
    fn test_optimizer_prefers_high_metric_region() {
        // After observing a clear winner, EI should be highest near that winner.
        let mut opt = GpOptimizer::new(maximize_space());

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
            .map(|&(lr, perp, metric)| Trial {
                config: make_config(lr, perp),
                metric,
            })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize);

        let ei_winner = gp.ei(&make_config(10.0, 20.0));
        let ei_loser = gp.ei(&make_config(0.5, 80.0));
        assert!(
            ei_winner >= ei_loser,
            "EI near winner ({ei_winner:.4}) should be >= loser ({ei_loser:.4})"
        );
    }
}
