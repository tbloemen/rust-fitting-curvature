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

use std::collections::VecDeque;
use std::f64::consts::PI;

use fitting_core::synthetic_data::Rng;
use serde::Serialize;

use crate::metrics::Metric;
use crate::search_space::{HyperParams, OptimizeDirection, ParamSpec, SearchSpace};

// ─── Public API ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Trial {
    pub config: HyperParams,
    pub metric: f64,
}

pub struct GpOptimizer {
    trials: Vec<Trial>,
    space: SearchSpace,
    n_ei_candidates: usize,
}

impl GpOptimizer {
    pub fn new(space: SearchSpace) -> Self {
        Self {
            trials: Vec::new(),
            space,
            n_ei_candidates: 1000,
        }
    }

    fn random_config(&self, rng: &mut Rng) -> HyperParams {
        self.space.sample(rng)
    }

    fn mutate_config(&self, config: &HyperParams, rng: &mut Rng) -> HyperParams {
        self.space.mutate_config(config, rng)
    }

    pub fn observe(&mut self, config: HyperParams, metric: f64) {
        self.trials.push(Trial { config, metric });
    }

    /// Suggest a batch of `n` promising configurations to evaluate in parallel.
    ///
    /// Scores `n_ei_candidates` random/mutated candidates by Expected Improvement,
    /// then returns the top-`n` by EI score, each refined by local hill-climbing.
    /// Evaluating this batch in parallel keeps all workers busy while only requiring
    /// one GP fit and one EI pass per batch — no inter-worker synchronisation needed.
    ///
    /// During the initial random phase (fewer than N_INIT real observations) the
    /// full batch is filled with random configs instead.
    pub fn suggest_batch(&self, n: usize, rng: &mut Rng) -> Vec<HyperParams> {
        const N_INIT: usize = 5;
        if self.trials.len() < N_INIT {
            return (0..n).map(|_| self.random_config(rng)).collect();
        }

        let gp = GpModel::fit(&self.trials, self.space.direction, &self.space.hyper_params);

        // Score all candidates, keeping track of the top-n by EI.
        let mut scored: Vec<(f64, HyperParams)> = (0..self.n_ei_candidates)
            .map(|_| {
                let candidate = if rng.uniform() < 0.3 {
                    let idx =
                        (rng.uniform() * self.trials.len() as f64) as usize % self.trials.len();
                    self.mutate_config(&self.trials[idx].config, rng)
                } else {
                    self.random_config(rng)
                };
                let ei = gp.ei(&candidate);
                (ei, candidate)
            })
            .collect();

        // Partial sort: descending EI, take top-n seeds.
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Refine each seed with local hill-climbing on EI.
        scored
            .into_iter()
            .take(n)
            .map(|(_, seed)| self.local_search(seed, rng, &gp))
            .collect()
    }

    pub fn best_trial(&self) -> f64 {
        if self.trials.is_empty() {
            return match self.space.direction {
                OptimizeDirection::Maximize => f64::MIN,
                OptimizeDirection::Minimize => f64::MAX,
            };
        }
        match self.space.direction {
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

    pub fn best_config(&self) -> Option<&HyperParams> {
        if self.trials.is_empty() {
            return None;
        }
        match self.space.direction {
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

    /// Fit a final GP to all observed trials and return the serialisable state.
    /// Returns `None` if fewer than 2 trials have been observed.
    pub fn export_state(&self) -> Option<GpState> {
        if self.trials.len() < 2 {
            return None;
        }
        let gp = GpModel::fit(&self.trials, self.space.direction, &self.space.hyper_params);
        Some(gp.to_state(&self.trials, self.space.direction))
    }

    fn local_search(&self, initial: HyperParams, rng: &mut Rng, gp: &GpModel) -> HyperParams {
        let mut current = initial;
        let mut current_ei = gp.ei(&current);
        for _ in 0..50 {
            let mutated = self.mutate_config(&current, rng);
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
    /// Raw (log-transformed, pre-standardisation) inputs — kept for export.
    xs_encoded: Vec<Vec<f64>>,
    x_means: Vec<f64>,
    x_stds: Vec<f64>,
    /// Cholesky factor L such that (K + jitter·I) = L Lᵀ, flat row-major.
    chol: Vec<f64>,
    /// α = K⁻¹ y, precomputed for fast posterior mean evaluation.
    alpha: Vec<f64>,
    /// f*_n: best standardised observed value (current incumbent).
    f_best_norm: f64,
    length_scale: f64,
    /// Output standardisation parameters (applied after sign-flip for Minimize).
    y_mean: f64,
    y_std: f64,
    n: usize,
    spec: HyperParams,
}

// ─── GP state export (for Python plotting) ───────────────────────────────────

/// A single observation as stored in the GP state export file.
#[derive(Serialize)]
pub struct GpExportObs {
    /// Original metric value (before any sign-flip).
    pub metric: f64,
    /// Log-transformed but not yet standardised input vector.
    pub x_encoded: Vec<f64>,
    /// Standardised input vector (what the GP kernel actually sees).
    pub x_norm: Vec<f64>,
}

/// Everything Python needs to reproduce GP posterior predictions without
/// re-implementing the fitting (MLE length-scale search, standardisation, etc.).
///
/// `alpha` (K⁻¹y) and the Cholesky factor are NOT stored — they can be
/// recomputed from the observations in O(n³), which is fast for typical n.
/// Python reconstruction:
///   xs_norm  = [obs.x_norm for obs in observations]
///   y_flipped = [-obs.metric if direction=="minimize" else obs.metric]
///   y_norm   = (y_flipped - y_mean) / y_std
///   K        = rbf(xs_norm, xs_norm, length_scale) + 1e-4 * I
///   L        = cholesky(K)
///   alpha    = L.T \ (L \ y_norm)
///   k_star   = rbf(xs_norm, x_test_norm, length_scale)
///   mu_norm  = k_star @ alpha
///   v        = L \ k_star
///   sigma_norm = sqrt(max(0, 1 - v @ v))
#[derive(Serialize)]
pub struct GpState {
    /// Human-readable parameter names in the same order as x_encoded / x_norm.
    pub param_names: Vec<String>,
    /// Which params are log-transformed (Python needs this to encode test points).
    pub log_scale_params: Vec<String>,
    pub direction: String,
    pub length_scale: f64,
    pub x_means: Vec<f64>,
    pub x_stds: Vec<f64>,
    /// Output mean in the (sign-flipped for Minimize) metric space.
    pub y_mean: f64,
    /// Output std in the (sign-flipped for Minimize) metric space.
    pub y_std: f64,
    pub f_best_norm: f64,
    pub n: usize,
    pub observations: Vec<GpExportObs>,
}

/// Diagonal noise added to the kernel matrix for numerical stability.
/// Frazier §3 recommends 10⁻⁶; 1e-4 is used here for robustness with small n.
const JITTER: f64 = 1e-4;

impl GpModel {
    fn fit(trials: &[Trial], direction: OptimizeDirection, spec: &HyperParams) -> Self {
        let raw_xs: Vec<Vec<f64>> = trials
            .iter()
            .map(|t| config_to_gp_input(&t.config, spec))
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
            xs_encoded: raw_xs,
            x_means,
            x_stds,
            chol,
            alpha,
            f_best_norm,
            length_scale,
            y_mean,
            y_std,
            n,
            spec: spec.clone(),
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

    /// Serialise the fitted GP state for external plotting tools.
    fn to_state(&self, trials: &[Trial], direction: OptimizeDirection) -> GpState {
        let observations = trials
            .iter()
            .enumerate()
            .map(|(i, t)| GpExportObs {
                metric: t.metric,
                x_encoded: self.xs_encoded[i].clone(),
                x_norm: self.xs_norm[i].clone(),
            })
            .collect();

        let (param_names, log_scale_params) = gp_param_names(&self.spec);

        GpState {
            param_names,
            log_scale_params,
            direction: direction.to_string(),
            length_scale: self.length_scale,
            x_means: self.x_means.clone(),
            x_stds: self.x_stds.clone(),
            y_mean: self.y_mean,
            y_std: self.y_std,
            f_best_norm: self.f_best_norm,
            n: self.n,
            observations,
        }
    }

    /// Expected Improvement for a candidate config (Frazier Eq. 7 / Eq. 8).
    fn ei(&self, config: &HyperParams) -> f64 {
        let x_raw = config_to_gp_input(config, &self.spec);
        let x_norm = standardize(&x_raw, &self.x_means, &self.x_stds);
        let (mu, sigma) = self.predict(&x_norm);
        expected_improvement(mu, sigma, self.f_best_norm)
    }

    /// Posterior mean at `config` de-standardised back to the raw metric space.
    /// Used by the Kriging Believer hallucination step in qParEGO.
    fn predict_mean_raw(&self, config: &HyperParams) -> f64 {
        let x_raw = config_to_gp_input(config, &self.spec);
        let x_norm = standardize(&x_raw, &self.x_means, &self.x_stds);
        let (mu_norm, _) = self.predict(&x_norm);
        mu_norm * self.y_std + self.y_mean
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

/// Encode a TrialConfig as a real-valued GP input vector containing only the
/// `Optimize` (free) parameters from `hp`, in canonical order.
///
/// Log-uniform parameters (learning_rate, perplexity_ratio) are log-transformed.
/// Fixed parameters are excluded so the GP dimensionality matches the search space.
/// When `optimize_curvature` is true, `curvature_magnitude` is appended (log-transformed).
/// Encode a sampled `HyperParams` as a GP input vector, including only `Optimize` fields.
/// The canonical order matches `gp_param_names` and `HyperParams::specs()`.
fn config_to_gp_input(config: &HyperParams, spec: &HyperParams) -> Vec<f64> {
    let mut v = Vec::new();
    if spec.learning_rate.is_optimized() {
        v.push(config.learning_rate.value().ln());
    }
    if spec.perplexity_ratio.is_optimized() {
        v.push(config.perplexity_ratio.value().ln());
    }
    if spec.momentum_main.is_optimized() {
        v.push(config.momentum_main.value());
    }
    if spec.momentum_early.is_optimized() {
        v.push(config.momentum_early.value());
    }
    if spec.centering_weight.is_optimized() {
        v.push(config.centering_weight.value());
    }
    if spec.global_loss_weight.is_optimized() {
        v.push(config.global_loss_weight.value());
    }
    if spec.norm_loss_weight.is_optimized() {
        v.push(config.norm_loss_weight.value());
    }
    if spec.early_exaggeration_factor.is_optimized() {
        v.push(config.early_exaggeration_factor.value());
    }
    if spec.n_iterations.is_optimized() {
        v.push(config.n_iterations.value());
    }
    if spec.early_exaggeration_iterations.is_optimized() {
        v.push(config.early_exaggeration_iterations.value());
    }
    if spec.curvature_magnitude.is_optimized() {
        v.push(config.curvature_magnitude.value().ln());
    }
    if spec.init_scale.is_optimized() {
        v.push(config.init_scale.value());
    }
    if spec.embed_dim.is_optimized() {
        v.push(config.embed_dim.value());
    }
    v
}

fn gp_param_names(spec: &HyperParams) -> (Vec<String>, Vec<String>) {
    let mut names = Vec::new();
    let mut log_params = Vec::new();
    macro_rules! push {
        ($field:expr, $name:expr, $log:expr) => {
            if $field.is_optimized() {
                names.push($name.to_string());
                if $log {
                    log_params.push($name.to_string());
                }
            }
        };
    }
    push!(spec.learning_rate, "learning_rate", true);
    push!(spec.perplexity_ratio, "perplexity_ratio", true);
    push!(spec.momentum_main, "momentum_main", false);
    push!(spec.momentum_early, "momentum_early", false);
    push!(spec.centering_weight, "centering_weight", false);
    push!(spec.global_loss_weight, "global_loss_weight", false);
    push!(spec.norm_loss_weight, "norm_loss_weight", false);
    push!(
        spec.early_exaggeration_factor,
        "early_exaggeration_factor",
        false
    );
    push!(spec.n_iterations, "n_iterations", false);
    push!(
        spec.early_exaggeration_iterations,
        "early_exaggeration_iterations",
        false
    );
    push!(spec.curvature_magnitude, "curvature_magnitude", true);
    push!(spec.init_scale, "init_scale", false);
    push!(spec.embed_dim, "embed_dim", false);
    (names, log_params)
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

// ─── qParEGO: multi-objective Bayesian optimisation ──────────────────────────
//
// Faithful implementation of Knowles (2006) "ParEGO: A Hybrid Algorithm with
// On-Line Landscape Approximation for Expensive Multiobjective Optimization
// Problems."
//
// Key correspondences to Algorithm 1:
//   LATINHYPERCUBE  → latin_hypercube_sample  (11d−1 stratified initial points)
//   NEWLAMBDA       → sample_discrete_simplex  (λ_j = l/s, uniform over Λ)
//   DACE            → GpModel (RBF-kernel GP with MLE length-scale; functionally
//                     equivalent to kriging; uses subset selection when n ≥ 25)
//   EVOLALG         → ParEgoOptimizer::evolalg ((μ+1)-ES, 10 000 EI evaluations,
//                     SBX crossover, per-gene shift mutation, binary tournament)

/// A single observed trial with all objective values.
#[derive(Debug, Clone)]
pub struct MultiTrial {
    pub config: HyperParams,
    /// Raw metric values in the same order as the `metrics` list.
    pub metrics: Vec<f64>,
}

pub struct ParEgoOptimizer {
    pub trials: Vec<MultiTrial>,
    pub metrics: Vec<Metric>,
    spec: HyperParams,
    /// s parameter (equation 1): weight vectors use λ_j = l/s, l ∈ {0,...,s}.
    s: usize,
    /// LHS configs queued for the initialisation phase (drained before GP phase).
    lhs_queue: VecDeque<HyperParams>,
    lhs_initialized: bool,
}

impl ParEgoOptimizer {
    pub fn new(metrics: Vec<Metric>, spec: HyperParams) -> Self {
        Self {
            trials: Vec::new(),
            metrics,
            spec,
            s: 5,
            lhs_queue: VecDeque::new(),
            lhs_initialized: false,
        }
    }

    pub fn observe(&mut self, config: HyperParams, metrics: Vec<f64>) {
        self.trials.push(MultiTrial { config, metrics });
    }

    /// Indices of Pareto-non-dominated trials (all objectives in maximise space).
    pub fn pareto_front_indices(&self) -> Vec<usize> {
        let points: Vec<Vec<f64>> = self
            .trials
            .iter()
            .map(|t| self.to_max_space(&t.metrics))
            .collect();
        pareto_front_indices(&points)
    }

    pub fn pareto_trials(&self) -> Vec<&MultiTrial> {
        self.pareto_front_indices()
            .into_iter()
            .map(|i| &self.trials[i])
            .collect()
    }

    /// Suggest a batch of `n` configs following qParEGO (Kriging Believer variant):
    ///
    /// 1. **Init phase** (first call): generate `11d−1` LHS points and queue them.
    /// 2. **LHS drain**: return queued points until the queue is empty.
    /// 3. **GP phase** (qParEGO): for each of the `n` members:
    ///    a. Draw λ ~ NEWLAMBDA(k, s).
    ///    b. Fit a GP on real observations + any Kriging Believer hallucinations from earlier members of this batch.
    ///    c. Run EVOLALG to find the config maximising EI under this GP.
    ///    d. Hallucinate: predict the GP posterior mean at the chosen config and add it as a fake observation so the next member avoids the same region.
    ///
    /// With `n = 1` this degenerates to standard (sequential) ParEGO.
    pub fn suggest_batch(&mut self, n: usize, rng: &mut Rng) -> Vec<HyperParams> {
        if !self.lhs_initialized {
            let n_lhs = 11 * self.spec.free_param_count() - 1;
            let lhs = latin_hypercube_sample(n_lhs, &self.spec, rng);
            self.lhs_queue.extend(lhs);
            self.lhs_initialized = true;
        }

        if !self.lhs_queue.is_empty() {
            let mut result = Vec::with_capacity(n);
            for _ in 0..n {
                result.push(
                    self.lhs_queue
                        .pop_front()
                        .unwrap_or_else(|| self.random_config(rng)),
                );
            }
            return result;
        }

        // qParEGO GP phase: Kriging Believer hallucinations accumulate across the batch.
        let mut hallucinated: Vec<Trial> = Vec::with_capacity(n);
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let weights = sample_discrete_simplex(self.metrics.len(), self.s, rng);
            let mut scalar_trials = self.scalarize_trials_subset(&weights, rng);
            scalar_trials.extend_from_slice(&hallucinated);
            let gp = GpModel::fit(&scalar_trials, OptimizeDirection::Maximize, &self.spec);
            let next = self.evolalg(&gp, &weights, rng);
            hallucinated.push(Trial {
                config: next.clone(),
                metric: gp.predict_mean_raw(&next),
            });
            result.push(next);
        }
        result
    }

    /// EVOLALG: (μ+1)-ES with 10 000 EI evaluations (Knowles 2006, procedure at lines 28–35).
    ///
    /// - **Population**: 20 solutions (5 mutants of the top-5 by current scalar + 15 LHS random).
    /// - **Selection**: binary tournament without replacement.
    /// - **Crossover**: SBX (η=2) with probability 0.2; otherwise clone the winner.
    /// - **Mutation**: per gene with prob 1/d, shift by δ*(hi−lo) where δ~U(0.0001,1), random ±.
    /// - **Replacement**: offspring replaces the first tournament parent if offspring EI ≥ parent EI.
    fn evolalg(&self, gp: &GpModel, weights: &[f64], rng: &mut Rng) -> HyperParams {
        const POP_SIZE: usize = 20;
        const N_EVALS: usize = 10_000;
        let dim = self.spec.free_param_count();

        let scalar_vals: Vec<f64> = self
            .trials
            .iter()
            .map(|t| self.scalar_value(t, weights))
            .collect();
        let mut sorted_idx: Vec<usize> = (0..self.trials.len()).collect();
        sorted_idx.sort_unstable_by(|&a, &b| {
            scalar_vals[b]
                .partial_cmp(&scalar_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut population: Vec<(f64, HyperParams)> = Vec::with_capacity(POP_SIZE);
        for i in 0..POP_SIZE {
            let cfg = if i < 5 && i < sorted_idx.len() {
                self.mutate_config(&self.trials[sorted_idx[i]].config, rng)
            } else {
                lhs_random_config(&self.spec, rng)
            };
            let ei = gp.ei(&cfg);
            population.push((ei, cfg));
        }

        let mut evals_used = POP_SIZE;
        while evals_used < N_EVALS {
            let i = (rng.uniform() * POP_SIZE as f64) as usize % POP_SIZE;
            let mut j = (rng.uniform() * (POP_SIZE - 1) as f64) as usize % (POP_SIZE - 1);
            if j >= i {
                j += 1;
            }
            let (winner, loser) = if population[i].0 >= population[j].0 {
                (i, j)
            } else {
                (j, i)
            };

            let offspring = if rng.uniform() < 0.2 {
                let child = sbx_crossover(
                    &population[winner].1,
                    &population[loser].1,
                    2.0,
                    &self.spec,
                    rng,
                );
                evolalg_mutate(&child, dim, &self.spec, rng)
            } else {
                evolalg_mutate(&population[winner].1, dim, &self.spec, rng)
            };

            let offspring_ei = gp.ei(&offspring);
            evals_used += 1;

            // Replace first tournament parent if offspring is at least as good.
            if offspring_ei >= population[winner].0 {
                population[winner] = (offspring_ei, offspring);
            }
        }

        population
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, cfg)| cfg)
            .unwrap_or_else(|| self.random_config(rng))
    }

    /// Scalar fitness of a trial under the current weight vector (higher = closer to ideal).
    fn scalar_value(&self, trial: &MultiTrial, weights: &[f64]) -> f64 {
        let flipped = self.to_max_space(&trial.metrics);
        // Return the raw weighted sum as a proxy for ranking (no normalisation needed here).
        flipped.iter().zip(weights).map(|(f, w)| w * f).sum()
    }

    /// Flip sign on Minimize objectives so all dimensions point upward.
    fn to_max_space(&self, metrics: &[f64]) -> Vec<f64> {
        metrics
            .iter()
            .enumerate()
            .map(|(i, &v)| match self.metrics[i].direction() {
                OptimizeDirection::Maximize => v,
                OptimizeDirection::Minimize => -v,
            })
            .collect()
    }

    /// Build scalar `Trial` list with DACE subset selection (Knowles 2006):
    ///
    /// - n < 25: use all observations.
    /// - n ≥ 25: use n/2 best (by current scalar fitness) + n/2 random without replacement.
    fn scalarize_trials_subset(&self, weights: &[f64], rng: &mut Rng) -> Vec<Trial> {
        let n = self.trials.len();
        let indices: Vec<usize> = if n < 25 {
            (0..n).collect()
        } else {
            let half = n / 2;
            // Sort by scalar fitness (descending) to get the best half.
            let mut scalar_vals: Vec<(usize, f64)> = self
                .trials
                .iter()
                .enumerate()
                .map(|(i, t)| (i, self.scalar_value(t, weights)))
                .collect();
            scalar_vals.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut chosen: Vec<usize> = scalar_vals[..half].iter().map(|&(i, _)| i).collect();
            // Random half without replacement from the remaining indices.
            let mut remaining: Vec<usize> = scalar_vals[half..].iter().map(|&(i, _)| i).collect();
            for i in (1..remaining.len()).rev() {
                let j = (rng.uniform() * (i + 1) as f64) as usize % (i + 1);
                remaining.swap(i, j);
            }
            chosen.extend_from_slice(&remaining[..half.min(remaining.len())]);
            chosen
        };

        self.scalarize_subset(&indices, weights)
    }

    /// Compute normalised Chebyshev scalar for a subset of trial indices.
    fn scalarize_subset(&self, indices: &[usize], weights: &[f64]) -> Vec<Trial> {
        let m = self.metrics.len();
        let flipped: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| self.to_max_space(&self.trials[i].metrics))
            .collect();

        let mut mins = vec![f64::MAX; m];
        let mut maxs = vec![f64::MIN; m];
        for row in &flipped {
            for (d, &v) in row.iter().enumerate() {
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }
        let ranges: Vec<f64> = (0..m).map(|d| (maxs[d] - mins[d]).max(1e-8)).collect();
        let ideal = vec![1.0_f64; m];

        indices
            .iter()
            .zip(&flipped)
            .map(|(&i, row)| {
                let norm: Vec<f64> = (0..m).map(|d| (row[d] - mins[d]) / ranges[d]).collect();
                let scalar = chebyshev_scalarize(&norm, weights, &ideal, 0.05);
                Trial {
                    config: self.trials[i].config.clone(),
                    metric: -scalar,
                }
            })
            .collect()
    }

    fn random_config(&self, rng: &mut Rng) -> HyperParams {
        self.spec.sample(rng)
    }

    fn mutate_config(&self, config: &HyperParams, rng: &mut Rng) -> HyperParams {
        self.spec.mutate(config, rng)
    }
}

// ─── ParEGO helper functions ──────────────────────────────────────────────────

/// Sample a weight vector uniformly from the discrete set Λ (Knowles 2006, eq. 1):
///   Λ = { λ = (λ_1,...,λ_k) | Σ λ_j = 1, λ_j = l/s, l ∈ {0,...,s} }
///
/// Stars-and-bars: pick (dim−1) positions from {0,...,s+dim−2} without
/// replacement via partial Fisher-Yates, sort, compute gaps divided by s.
fn sample_discrete_simplex(dim: usize, s: usize, rng: &mut Rng) -> Vec<f64> {
    if dim == 1 {
        return vec![1.0];
    }
    let total = s + dim - 1;
    let mut pool: Vec<usize> = (0..total).collect();
    for i in 0..(dim - 1) {
        let remaining = total - i;
        let j = i + (rng.uniform() * remaining as f64) as usize % remaining;
        pool.swap(i, j);
    }
    let mut positions = pool[..(dim - 1)].to_vec();
    positions.sort_unstable();

    let mut result = Vec::with_capacity(dim);
    let mut prev = 0usize;
    for &pos in &positions {
        result.push((pos - prev) as f64 / s as f64);
        prev = pos + 1;
    }
    result.push((total - prev) as f64 / s as f64);
    result
}

/// Latin Hypercube Sample of `n` points over the search space defined by `spec`.
///
/// Returns `Vec<HyperParams>` where every field is `Fixed`. Only `Optimize` fields
/// in `spec` consume LHS columns; `Fixed` fields keep their fixed values in every row.
pub fn latin_hypercube_sample(n: usize, spec: &HyperParams, rng: &mut Rng) -> Vec<HyperParams> {
    let dim = spec.free_param_count();

    let cols: Vec<Vec<f64>> = (0..dim)
        .map(|_| {
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (rng.uniform() * (i + 1) as f64) as usize % (i + 1);
                perm.swap(i, j);
            }
            perm.iter()
                .map(|&k| (k as f64 + rng.uniform()) / n as f64)
                .collect()
        })
        .collect();

    let mut col_iter = cols.iter();
    macro_rules! lhs_col {
        ($field:expr) => {
            match &$field {
                ParamSpec::Fixed(v) => vec![*v; n],
                ParamSpec::Optimize {
                    lo,
                    hi,
                    log_scale: true,
                } => col_iter
                    .next()
                    .unwrap()
                    .iter()
                    .map(|&t| lhs_map_log(t, *lo, *hi))
                    .collect(),
                ParamSpec::Optimize {
                    lo,
                    hi,
                    log_scale: false,
                } => col_iter
                    .next()
                    .unwrap()
                    .iter()
                    .map(|&t| lhs_map_linear(t, *lo, *hi))
                    .collect(),
            }
        };
    }

    let lr_col = lhs_col!(spec.learning_rate);
    let perp_col = lhs_col!(spec.perplexity_ratio);
    let mom_col = lhs_col!(spec.momentum_main);
    let mome_col = lhs_col!(spec.momentum_early);
    let cen_col = lhs_col!(spec.centering_weight);
    let glw_col = lhs_col!(spec.global_loss_weight);
    let nlw_col = lhs_col!(spec.norm_loss_weight);
    let eef_col = lhs_col!(spec.early_exaggeration_factor);
    let nit_col = lhs_col!(spec.n_iterations);
    let eei_col = lhs_col!(spec.early_exaggeration_iterations);
    let cur_col = lhs_col!(spec.curvature_magnitude);
    let isc_col = lhs_col!(spec.init_scale);
    let edim_col = lhs_col!(spec.embed_dim);

    (0..n)
        .map(|i| HyperParams {
            learning_rate: ParamSpec::Fixed(lr_col[i]),
            perplexity_ratio: ParamSpec::Fixed(perp_col[i]),
            momentum_main: ParamSpec::Fixed(mom_col[i]),
            momentum_early: ParamSpec::Fixed(mome_col[i]),
            centering_weight: ParamSpec::Fixed(cen_col[i]),
            global_loss_weight: ParamSpec::Fixed(glw_col[i]),
            norm_loss_weight: ParamSpec::Fixed(nlw_col[i]),
            early_exaggeration_factor: ParamSpec::Fixed(eef_col[i]),
            n_iterations: ParamSpec::Fixed(nit_col[i]),
            early_exaggeration_iterations: ParamSpec::Fixed(eei_col[i]),
            curvature_magnitude: ParamSpec::Fixed(cur_col[i]),
            init_scale: ParamSpec::Fixed(isc_col[i]),
            embed_dim: ParamSpec::Fixed(edim_col[i]),
        })
        .collect()
}

fn lhs_random_config(spec: &HyperParams, rng: &mut Rng) -> HyperParams {
    latin_hypercube_sample(1, spec, rng).remove(0)
}

fn lhs_map_log(t: f64, lo: f64, hi: f64) -> f64 {
    (lo.ln() + t * (hi.ln() - lo.ln())).exp().clamp(lo, hi)
}

fn lhs_map_linear(t: f64, lo: f64, hi: f64) -> f64 {
    (lo + t * (hi - lo)).clamp(lo, hi)
}

/// Simulated Binary Crossover (SBX, Deb & Agrawal 1995) between two parents.
///
/// For each parameter, the SBX spread factor β is derived from u~U(0,1):
///   β = (2u)^(1/(η+1))        if u ≤ 0.5
///   β = (1/(2(1-u)))^(1/(η+1)) otherwise
///
/// child = 0.5 * ((1+β)*p1 + (1-β)*p2)
///
/// Log-scale parameters (learning_rate, perplexity_ratio, curvature_magnitude)
/// have SBX applied in log space and are then exponentiated back.
/// SBX crossover between two sampled (all-Fixed) `HyperParams`. Returns a new sampled HP.
pub fn sbx_crossover(
    a: &HyperParams,
    b: &HyperParams,
    eta: f64,
    spec: &HyperParams,
    rng: &mut Rng,
) -> HyperParams {
    macro_rules! cross {
        ($field:expr, $av:expr, $bv:expr) => {
            ParamSpec::Fixed(match &$field {
                ParamSpec::Fixed(v) => *v,
                ParamSpec::Optimize {
                    lo,
                    hi,
                    log_scale: true,
                } => sbx_scalar($av.ln(), $bv.ln(), eta, rng)
                    .exp()
                    .clamp(*lo, *hi),
                ParamSpec::Optimize {
                    lo,
                    hi,
                    log_scale: false,
                } => sbx_scalar($av, $bv, eta, rng).clamp(*lo, *hi),
            })
        };
    }
    HyperParams {
        learning_rate: cross!(
            spec.learning_rate,
            a.learning_rate.value(),
            b.learning_rate.value()
        ),
        perplexity_ratio: cross!(
            spec.perplexity_ratio,
            a.perplexity_ratio.value(),
            b.perplexity_ratio.value()
        ),
        momentum_main: cross!(
            spec.momentum_main,
            a.momentum_main.value(),
            b.momentum_main.value()
        ),
        momentum_early: cross!(
            spec.momentum_early,
            a.momentum_early.value(),
            b.momentum_early.value()
        ),
        centering_weight: cross!(
            spec.centering_weight,
            a.centering_weight.value(),
            b.centering_weight.value()
        ),
        global_loss_weight: cross!(
            spec.global_loss_weight,
            a.global_loss_weight.value(),
            b.global_loss_weight.value()
        ),
        norm_loss_weight: cross!(
            spec.norm_loss_weight,
            a.norm_loss_weight.value(),
            b.norm_loss_weight.value()
        ),
        early_exaggeration_factor: cross!(
            spec.early_exaggeration_factor,
            a.early_exaggeration_factor.value(),
            b.early_exaggeration_factor.value()
        ),
        n_iterations: cross!(
            spec.n_iterations,
            a.n_iterations.value(),
            b.n_iterations.value()
        ),
        early_exaggeration_iterations: cross!(
            spec.early_exaggeration_iterations,
            a.early_exaggeration_iterations.value(),
            b.early_exaggeration_iterations.value()
        ),
        curvature_magnitude: cross!(
            spec.curvature_magnitude,
            a.curvature_magnitude.value(),
            b.curvature_magnitude.value()
        ),
        init_scale: cross!(spec.init_scale, a.init_scale.value(), b.init_scale.value()),
        embed_dim: cross!(spec.embed_dim, a.embed_dim.value(), b.embed_dim.value()),
    }
}

/// Scalar SBX step for a single parameter pair (operates in whatever space the caller uses).
fn sbx_scalar(p1: f64, p2: f64, eta: f64, rng: &mut Rng) -> f64 {
    let u = rng.uniform().max(1e-10);
    let beta = if u <= 0.5 {
        (2.0 * u).powf(1.0 / (eta + 1.0))
    } else {
        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
    };
    0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
}

/// Per-gene shift mutation used inside EVOLALG (Knowles 2006).
///
/// Each gene mutates with probability 1/d.  The shift magnitude is δ*(hi−lo)
/// where δ~U(0.0001, 1), applied in log space for log-scale parameters.
/// Per-gene shift mutation for a sampled (all-Fixed) `HyperParams`. Returns a new sampled HP.
pub fn evolalg_mutate(
    config: &HyperParams,
    dim: usize,
    spec: &HyperParams,
    rng: &mut Rng,
) -> HyperParams {
    let p = 1.0 / dim as f64;

    macro_rules! mutate {
        ($field:expr, $v:expr) => {
            ParamSpec::Fixed(match &$field {
                ParamSpec::Fixed(fv) => *fv,
                ParamSpec::Optimize {
                    lo,
                    hi,
                    log_scale: true,
                } => {
                    if rng.uniform() < p {
                        let d = rng.uniform() * (1.0 - 0.0001) + 0.0001;
                        let s = if rng.uniform() < 0.5 { 1.0_f64 } else { -1.0 };
                        ($v.ln() + s * d * (hi.ln() - lo.ln()))
                            .exp()
                            .clamp(*lo, *hi)
                    } else {
                        $v
                    }
                }
                ParamSpec::Optimize {
                    lo,
                    hi,
                    log_scale: false,
                } => {
                    if rng.uniform() < p {
                        let d = rng.uniform() * (1.0 - 0.0001) + 0.0001;
                        let s = if rng.uniform() < 0.5 { 1.0_f64 } else { -1.0 };
                        ($v + s * d * (hi - lo)).clamp(*lo, *hi)
                    } else {
                        $v
                    }
                }
            })
        };
    }

    HyperParams {
        learning_rate: mutate!(spec.learning_rate, config.learning_rate.value()),
        perplexity_ratio: mutate!(spec.perplexity_ratio, config.perplexity_ratio.value()),
        momentum_main: mutate!(spec.momentum_main, config.momentum_main.value()),
        momentum_early: mutate!(spec.momentum_early, config.momentum_early.value()),
        centering_weight: mutate!(spec.centering_weight, config.centering_weight.value()),
        global_loss_weight: mutate!(spec.global_loss_weight, config.global_loss_weight.value()),
        norm_loss_weight: mutate!(spec.norm_loss_weight, config.norm_loss_weight.value()),
        early_exaggeration_factor: mutate!(
            spec.early_exaggeration_factor,
            config.early_exaggeration_factor.value()
        ),
        n_iterations: mutate!(spec.n_iterations, config.n_iterations.value()),
        early_exaggeration_iterations: mutate!(
            spec.early_exaggeration_iterations,
            config.early_exaggeration_iterations.value()
        ),
        curvature_magnitude: mutate!(spec.curvature_magnitude, config.curvature_magnitude.value()),
        init_scale: mutate!(spec.init_scale, config.init_scale.value()),
        embed_dim: mutate!(spec.embed_dim, config.embed_dim.value()),
    }
}

/// Augmented Chebyshev scalarisation (Knowles 2006, Eq. 1).
///
///   s = max_i(λ_i · (z*_i − f_i))  +  ρ · Σ_i λ_i · (z*_i − f_i)
///
/// Lower s = closer to ideal = better.  ρ=0.05 is the standard value.
fn chebyshev_scalarize(metrics_norm: &[f64], weights: &[f64], ideal: &[f64], rho: f64) -> f64 {
    let diffs: Vec<f64> = metrics_norm
        .iter()
        .zip(ideal)
        .zip(weights)
        .map(|((f, z), w)| w * (z - f))
        .collect();
    let max_term = diffs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_term: f64 = diffs.iter().sum();
    max_term + rho * sum_term
}

/// Return indices of Pareto-non-dominated rows (all dimensions assumed maximise).
fn pareto_front_indices(points: &[Vec<f64>]) -> Vec<usize> {
    let n = points.len();
    let mut non_dominated = Vec::new();
    'outer: for i in 0..n {
        for j in 0..n {
            if i != j && dominates(&points[j], &points[i]) {
                continue 'outer;
            }
        }
        non_dominated.push(i);
    }
    non_dominated
}

/// True if `a` Pareto-dominates `b`: a ≥ b on all objectives, strictly > on at least one.
fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(ai, bi)| ai >= bi) && a.iter().zip(b).any(|(ai, bi)| ai > bi)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fitting_core::synthetic_data::Rng;

    fn maximize_space() -> SearchSpace {
        SearchSpace {
            direction: OptimizeDirection::Maximize,
            hyper_params: HyperParams::all_free(),
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn make_config(lr: f64, perp_ratio: f64) -> HyperParams {
        let mut hp = HyperParams::all_free().sample(&mut fitting_core::synthetic_data::Rng::new(0));
        hp.learning_rate = ParamSpec::Fixed(lr);
        hp.perplexity_ratio = ParamSpec::Fixed(perp_ratio);
        hp.centering_weight = ParamSpec::Fixed(0.0);
        hp.global_loss_weight = ParamSpec::Fixed(0.0);
        hp.norm_loss_weight = ParamSpec::Fixed(0.0);
        hp.early_exaggeration_factor = ParamSpec::Fixed(12.0);
        hp
    }

    fn random_config(rng: &mut Rng) -> HyperParams {
        maximize_space().sample(rng)
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
            (0.05..=10.0).contains(&l),
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
        let hp = HyperParams::all_free();
        let v = config_to_gp_input(&cfg, &hp);
        assert!(close(v[0], 1.0, 1e-10)); // ln(e) = 1
        assert!(close(v[1], 2.0, 1e-10)); // ln(e²) = 2
    }

    #[test]
    fn test_config_to_gp_input_passthrough_fields() {
        let cfg = make_config(1.0, 10.0);
        let hp = HyperParams::all_free();
        let v = config_to_gp_input(&cfg, &hp);
        // all_free: lr(0), perp(1), cen(2), glw(3), nlw(4), eef(5) — momentum is Fixed so skipped
        assert!(close(v[2], cfg.centering_weight.value(), 1e-15));
        assert!(close(v[3], cfg.global_loss_weight.value(), 1e-15));
        assert!(close(v[4], cfg.norm_loss_weight.value(), 1e-15));
    }

    // ─── GpModel ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gp_predict_low_variance_near_observations() {
        let mut rng = Rng::new(42);
        let hp = HyperParams::all_free();
        let trials: Vec<Trial> = (0..10)
            .map(|_| Trial {
                config: random_config(&mut rng),
                metric: rng.uniform(),
            })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize, &hp);
        let x_raw = config_to_gp_input(&trials[0].config, &hp);
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
        let hp = HyperParams::all_free();
        let trials: Vec<Trial> = (0..8)
            .map(|_| Trial {
                config: random_config(&mut rng),
                metric: rng.uniform(),
            })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize, &hp);
        for _ in 0..20 {
            let cfg_tmp = random_config(&mut rng);
            let x_raw = config_to_gp_input(&cfg_tmp, &hp);
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
        let hp = HyperParams::all_free();
        let trials: Vec<Trial> = (0..8)
            .map(|_| Trial {
                config: random_config(&mut rng),
                metric: rng.uniform(),
            })
            .collect();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize, &hp);
        for _ in 0..30 {
            let cfg = random_config(&mut rng);
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
        assert!(close(best.learning_rate.value(), 10.0, 1e-10));
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
            let cfg = opt.suggest_batch(1, &mut rng).remove(0);
            opt.observe(cfg, rng.uniform());
        }
        let _ = opt.suggest_batch(1, &mut rng);
    }

    #[test]
    fn test_optimizer_suggest_gp_phase() {
        // N_INIT+ trials → GP-guided suggest must not panic and return a valid config.
        let mut opt = GpOptimizer::new(maximize_space());
        let mut rng = Rng::new(42);
        for i in 0..10 {
            let cfg = random_config(&mut rng);
            opt.observe(cfg, i as f64 * 0.1);
        }
        let cfg = opt.suggest_batch(1, &mut rng).remove(0);
        assert!(cfg.learning_rate.value() > 0.0);
        assert!(cfg.perplexity_ratio.value() >= 0.0004);
        assert_eq!(cfg.momentum_main.value(), 0.8);
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
        let hp = HyperParams::all_free();
        let gp = GpModel::fit(&trials, OptimizeDirection::Maximize, &hp);

        let ei_winner = gp.ei(&make_config(10.0, 20.0));
        let ei_loser = gp.ei(&make_config(0.5, 80.0));
        assert!(
            ei_winner >= ei_loser,
            "EI near winner ({ei_winner:.4}) should be >= loser ({ei_loser:.4})"
        );
    }

    // ─── latin_hypercube_sample ───────────────────────────────────────────────

    fn assert_config_in_bounds(cfg: &HyperParams, label: &str) {
        use crate::search_space::{
            CEN_MAX, CEN_MIN, GLW_MAX, GLW_MIN, LR_MAX, LR_MIN, NLW_MAX, NLW_MIN, PERP_MAX,
            PERP_MIN,
        };
        let lr = cfg.learning_rate.value();
        let perp = cfg.perplexity_ratio.value();
        let cen = cfg.centering_weight.value();
        let glw = cfg.global_loss_weight.value();
        let nlw = cfg.norm_loss_weight.value();
        assert!((LR_MIN..=LR_MAX).contains(&lr), "{label}: lr={lr}");
        assert!(
            (PERP_MIN..=PERP_MAX).contains(&perp),
            "{label}: perp={perp}"
        );
        assert!((CEN_MIN..=CEN_MAX).contains(&cen), "{label}: cen={cen}");
        assert!((GLW_MIN..=GLW_MAX).contains(&glw), "{label}: glw={glw}");
        assert!((NLW_MIN..=NLW_MAX).contains(&nlw), "{label}: nlw={nlw}");
    }

    #[test]
    fn lhs_correct_count() {
        let mut rng = Rng::new(1);
        let pts = latin_hypercube_sample(65, &HyperParams::all_free(), &mut rng);
        assert_eq!(pts.len(), 65);
    }

    #[test]
    fn lhs_all_params_in_bounds() {
        let mut rng = Rng::new(2);
        let pts = latin_hypercube_sample(65, &HyperParams::all_free(), &mut rng);
        for (i, cfg) in pts.iter().enumerate() {
            assert_config_in_bounds(cfg, &format!("point {i}"));
        }
    }

    #[test]
    fn lhs_lr_log_coverage() {
        use crate::search_space::{LR_MAX, LR_MIN};
        let mut rng = Rng::new(3);
        let pts = latin_hypercube_sample(65, &HyperParams::all_free(), &mut rng);
        let (min_lr, max_lr) = pts.iter().fold((f64::MAX, f64::MIN), |(lo, hi), cfg| {
            (
                lo.min(cfg.learning_rate.value()),
                hi.max(cfg.learning_rate.value()),
            )
        });
        let log_range = LR_MAX.ln() - LR_MIN.ln();
        let log_covered = max_lr.ln() - min_lr.ln();
        assert!(
            log_covered > 0.5 * log_range,
            "lr log coverage {log_covered:.3} < half of {log_range:.3}"
        );
    }

    #[test]
    fn lhs_stratified_per_dim() {
        // Each dimension must have exactly one point per bin (floor(v * n) is a permutation of 0..n).
        use crate::search_space::{
            CEN_MAX, CEN_MIN, GLW_MAX, GLW_MIN, LR_MAX, LR_MIN, NLW_MAX, NLW_MIN, PERP_MAX,
            PERP_MIN,
        };
        let n = 30usize;
        let mut rng = Rng::new(4);
        let pts = latin_hypercube_sample(n, &HyperParams::all_free(), &mut rng);

        let check_bins = |values: Vec<f64>, lo: f64, hi: f64, name: &str| {
            let mut bins = vec![false; n];
            for v in values {
                let t = (v - lo) / (hi - lo);
                let b = ((t * n as f64) as usize).min(n - 1);
                assert!(!bins[b], "{name}: bin {b} occupied twice");
                bins[b] = true;
            }
        };
        check_bins(
            pts.iter().map(|c| c.learning_rate.value().ln()).collect(),
            LR_MIN.ln(),
            LR_MAX.ln(),
            "lr",
        );
        check_bins(
            pts.iter()
                .map(|c| c.perplexity_ratio.value().ln())
                .collect(),
            PERP_MIN.ln(),
            PERP_MAX.ln(),
            "perp",
        );
        // momentum_main is Fixed(0.8) in all_free — not a stratified LHS dimension
        check_bins(
            pts.iter().map(|c| c.centering_weight.value()).collect(),
            CEN_MIN,
            CEN_MAX,
            "cen",
        );
        check_bins(
            pts.iter().map(|c| c.global_loss_weight.value()).collect(),
            GLW_MIN,
            GLW_MAX,
            "glw",
        );
        check_bins(
            pts.iter().map(|c| c.norm_loss_weight.value()).collect(),
            NLW_MIN,
            NLW_MAX,
            "nlw",
        );
    }

    #[test]
    fn lhs_curvature_disabled() {
        // When curvature_magnitude is Fixed(0.0), LHS points should all have magnitude=0.
        let mut rng = Rng::new(5);
        let pts = latin_hypercube_sample(20, &HyperParams::all_free(), &mut rng);
        for cfg in &pts {
            assert_eq!(
                cfg.curvature_magnitude.value(),
                0.0,
                "curvature should be 0 when Fixed(0)"
            );
        }
    }

    #[test]
    fn lhs_curvature_in_bounds() {
        let k_min = 0.1;
        let k_max = 3.0;
        let mut rng = Rng::new(6);
        let mut spec = HyperParams::all_free();
        spec.curvature_magnitude = ParamSpec::Optimize {
            lo: k_min,
            hi: k_max,
            log_scale: true,
        };
        let pts = latin_hypercube_sample(30, &spec, &mut rng);
        for cfg in &pts {
            let v = cfg.curvature_magnitude.value();
            assert!(
                v >= k_min && v <= k_max,
                "curvature {v} out of [{k_min},{k_max}]"
            );
        }
    }

    #[test]
    fn lhs_11d_minus_1_count() {
        let mut rng = Rng::new(7);
        // all_free without curvature: lr, perp, cen, glw, nlw, eef = 6 free params
        let spec6 = HyperParams::all_free();
        let dim6 = spec6.free_param_count();
        let n6 = 11 * dim6 - 1;
        let pts = latin_hypercube_sample(n6, &spec6, &mut rng);
        assert_eq!(pts.len(), n6, "expected {n6} LHS points");

        // with curvature: 7 free params
        let mut spec7 = HyperParams::all_free();
        spec7.curvature_magnitude = ParamSpec::Optimize {
            lo: 0.001,
            hi: 5.0,
            log_scale: true,
        };
        let dim7 = spec7.free_param_count();
        let n7 = 11 * dim7 - 1;
        let pts7 = latin_hypercube_sample(n7, &spec7, &mut rng);
        assert_eq!(pts7.len(), n7, "expected {n7} LHS points with curvature");
    }

    // ─── sample_discrete_simplex ──────────────────────────────────────────────

    #[test]
    fn discrete_simplex_sums_to_one() {
        let mut rng = Rng::new(10);
        for &(dim, s) in &[(2, 5), (3, 5), (5, 4), (10, 5), (10, 3)] {
            for _ in 0..50 {
                let w = sample_discrete_simplex(dim, s, &mut rng);
                let sum: f64 = w.iter().sum();
                assert!((sum - 1.0).abs() < 1e-12, "dim={dim} s={s}: sum={sum}");
            }
        }
    }

    #[test]
    fn discrete_simplex_all_multiples_of_1_over_s() {
        let mut rng = Rng::new(11);
        let s = 5usize;
        for _ in 0..100 {
            let w = sample_discrete_simplex(10, s, &mut rng);
            for &wi in &w {
                let scaled = wi * s as f64;
                assert!(
                    (scaled - scaled.round()).abs() < 1e-10,
                    "w*s={scaled} not integer"
                );
            }
        }
    }

    #[test]
    fn discrete_simplex_non_negative() {
        let mut rng = Rng::new(12);
        for _ in 0..200 {
            let w = sample_discrete_simplex(10, 5, &mut rng);
            for &wi in &w {
                assert!(wi >= 0.0, "negative weight: {wi}");
            }
        }
    }

    #[test]
    fn discrete_simplex_dim1_returns_one() {
        let mut rng = Rng::new(13);
        let w = sample_discrete_simplex(1, 5, &mut rng);
        assert_eq!(w, vec![1.0]);
    }

    #[test]
    fn discrete_simplex_covers_extremes() {
        // Over many samples, at least one weight == 1.0 and one == 0.0 should appear.
        let mut rng = Rng::new(14);
        let mut saw_one = false;
        let mut saw_zero = false;
        for _ in 0..2000 {
            let w = sample_discrete_simplex(5, 5, &mut rng);
            for &wi in &w {
                if wi == 1.0 {
                    saw_one = true;
                }
                if wi == 0.0 {
                    saw_zero = true;
                }
            }
        }
        assert!(saw_one, "never sampled a weight of 1.0");
        assert!(saw_zero, "never sampled a weight of 0.0");
    }

    // ─── sbx_crossover ────────────────────────────────────────────────────────

    /// A concrete (all-Fixed) HyperParams for use as a parent/input in SBX/evolalg tests.
    fn default_config() -> HyperParams {
        HyperParams {
            learning_rate: ParamSpec::Fixed(5.0),
            perplexity_ratio: ParamSpec::Fixed(0.01),
            momentum_main: ParamSpec::Fixed(0.8),
            momentum_early: ParamSpec::Fixed(0.5),
            centering_weight: ParamSpec::Fixed(1.0),
            global_loss_weight: ParamSpec::Fixed(0.5),
            norm_loss_weight: ParamSpec::Fixed(0.01),
            early_exaggeration_factor: ParamSpec::Fixed(12.0),
            n_iterations: ParamSpec::Fixed(800.0),
            early_exaggeration_iterations: ParamSpec::Fixed(250.0),
            curvature_magnitude: ParamSpec::Fixed(0.0),
            init_scale: ParamSpec::Fixed(1.0),
            embed_dim: ParamSpec::Fixed(2.0),
        }
    }

    #[test]
    fn sbx_offspring_in_bounds() {
        let mut rng = Rng::new(20);
        let a = default_config();
        let b = HyperParams {
            learning_rate: ParamSpec::Fixed(10.0),
            perplexity_ratio: ParamSpec::Fixed(0.005),
            momentum_main: ParamSpec::Fixed(0.9),
            centering_weight: ParamSpec::Fixed(0.5),
            global_loss_weight: ParamSpec::Fixed(0.3),
            norm_loss_weight: ParamSpec::Fixed(0.005),
            early_exaggeration_factor: ParamSpec::Fixed(12.0),
            curvature_magnitude: ParamSpec::Fixed(0.0),
            ..default_config()
        };
        let hp = HyperParams::all_free();
        for i in 0..200 {
            let child = sbx_crossover(&a, &b, 2.0, &hp, &mut rng);
            assert_config_in_bounds(&child, &format!("sbx child {i}"));
        }
    }

    #[test]
    fn sbx_identical_parents_returns_parent() {
        let mut rng = Rng::new(21);
        let a = default_config();
        // When both parents are identical, SBX with β computed from u=0.5 gives β=1 → child = parent.
        // This holds in expectation but depends on the random draw; test with many trials.
        let mut all_same = true;
        for _ in 0..50 {
            let child = sbx_crossover(&a, &a, 2.0, &HyperParams::all_free(), &mut rng);
            if (child.learning_rate.value() - a.learning_rate.value()).abs() > 1e-6 {
                all_same = false;
                break;
            }
        }
        // Identical parents should always produce the parent (SBX: child = 0.5*(1+β)*p + 0.5*(1-β)*p = p).
        assert!(
            all_same,
            "SBX with identical parents should always return the parent"
        );
    }

    #[test]
    fn sbx_not_always_parent_a() {
        let mut rng = Rng::new(22);
        let a = default_config();
        let b = HyperParams {
            learning_rate: ParamSpec::Fixed(1.0),
            ..default_config()
        };
        let differs: bool = (0..50).any(|_| {
            let child = sbx_crossover(&a, &b, 2.0, &HyperParams::all_free(), &mut rng);
            (child.learning_rate.value() - a.learning_rate.value()).abs() > 1e-9
        });
        assert!(
            differs,
            "SBX should sometimes produce a child different from parent a"
        );
    }

    #[test]
    fn sbx_log_params_stay_positive() {
        let mut rng = Rng::new(23);
        let a = default_config();
        let b = HyperParams {
            learning_rate: ParamSpec::Fixed(0.6),
            perplexity_ratio: ParamSpec::Fixed(0.0005),
            ..default_config()
        };
        for _ in 0..200 {
            let child = sbx_crossover(&a, &b, 2.0, &HyperParams::all_free(), &mut rng);
            assert!(child.learning_rate.value() > 0.0, "lr must be positive");
            assert!(
                child.perplexity_ratio.value() > 0.0,
                "perplexity_ratio must be positive"
            );
        }
    }

    // ─── evolalg_mutate ───────────────────────────────────────────────────────

    #[test]
    fn evolalg_mutate_in_bounds() {
        let mut rng = Rng::new(30);
        let cfg = default_config();
        for i in 0..500 {
            let m = evolalg_mutate(&cfg, 6, &HyperParams::all_free(), &mut rng);
            assert_config_in_bounds(&m, &format!("mutant {i}"));
        }
    }

    #[test]
    fn evolalg_mutate_changes_something() {
        let mut rng = Rng::new(31);
        let cfg = default_config();
        let changed = (0..100).any(|_| {
            let m = evolalg_mutate(&cfg, 6, &HyperParams::all_free(), &mut rng);
            (m.learning_rate.value() - cfg.learning_rate.value()).abs() > 1e-9
                || (m.centering_weight.value() - cfg.centering_weight.value()).abs() > 1e-9
                || (m.global_loss_weight.value() - cfg.global_loss_weight.value()).abs() > 1e-9
        });
        assert!(
            changed,
            "mutation should change at least one parameter over 100 calls"
        );
    }

    #[test]
    fn evolalg_mutate_log_params_positive() {
        let mut rng = Rng::new(32);
        let cfg = default_config();
        for _ in 0..200 {
            let m = evolalg_mutate(&cfg, 6, &HyperParams::all_free(), &mut rng);
            assert!(m.learning_rate.value() > 0.0, "lr must be positive");
            assert!(
                m.perplexity_ratio.value() > 0.0,
                "perplexity_ratio must be positive"
            );
        }
    }

    // ─── evolalg ─────────────────────────────────────────────────────────────

    fn make_pareto_optimizer_with_trials(n: usize) -> (ParEgoOptimizer, Vec<Metric>) {
        use crate::metrics::Metric;
        let metrics = vec![Metric::Trustworthiness, Metric::Continuity];
        let mut opt = ParEgoOptimizer::new(metrics.clone(), HyperParams::all_free());
        let mut rng = Rng::new(99);
        let space = maximize_space();
        for _ in 0..n {
            let cfg = space.sample(&mut rng);
            opt.observe(cfg, vec![rng.uniform(), rng.uniform()]);
        }
        (opt, metrics)
    }

    #[test]
    fn evolalg_returns_valid_config() {
        let (opt, _) = make_pareto_optimizer_with_trials(10);
        let mut rng = Rng::new(40);
        let weights = sample_discrete_simplex(2, 5, &mut rng);
        let scalar_trials =
            opt.scalarize_subset(&(0..opt.trials.len()).collect::<Vec<_>>(), &weights);
        let hp = HyperParams::all_free();
        let gp = GpModel::fit(&scalar_trials, OptimizeDirection::Maximize, &hp);
        let result = opt.evolalg(&gp, &weights, &mut rng);
        assert_config_in_bounds(&result, "evolalg result");
    }

    #[test]
    fn evolalg_improves_over_random() {
        // The config returned by EVOLALG should have higher EI than the median of 100 random configs.
        let (opt, _) = make_pareto_optimizer_with_trials(15);
        let mut rng = Rng::new(41);
        let weights = sample_discrete_simplex(2, 5, &mut rng);
        let scalar_trials =
            opt.scalarize_subset(&(0..opt.trials.len()).collect::<Vec<_>>(), &weights);
        let hp = HyperParams::all_free();
        let gp = GpModel::fit(&scalar_trials, OptimizeDirection::Maximize, &hp);

        let best = opt.evolalg(&gp, &weights, &mut rng);
        let best_ei = gp.ei(&best);

        let space = maximize_space();
        let mut random_eis: Vec<f64> = (0..100).map(|_| gp.ei(&space.sample(&mut rng))).collect();
        random_eis.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ei = random_eis[50];

        assert!(
            best_ei >= median_ei,
            "EVOLALG EI {best_ei:.6} should be >= median random EI {median_ei:.6}"
        );
    }

    // ─── ParEgoOptimizer::suggest_batch (integration) ─────────────────────────

    #[test]
    fn pareto_lhs_exactly_11d_minus_1() {
        use crate::metrics::Metric;
        let metrics = vec![Metric::Trustworthiness, Metric::Continuity];
        let mut opt = ParEgoOptimizer::new(metrics, HyperParams::all_free());
        let mut rng = Rng::new(50);
        // First suggest_batch populates the queue with 11*6-1=65 points.
        let batch = opt.suggest_batch(200, &mut rng);
        // We asked for 200 but only 65 LHS exist; the rest would come from GP phase.
        // However, since we have 0 trials, the GP phase would fail — so the queue
        // just drains. Verify by checking the LHS queue was exactly 65 and all came out.
        assert_eq!(
            opt.lhs_queue.len(),
            0,
            "LHS queue should be empty after draining"
        );
        assert_eq!(batch.len(), 200);
        // First 65 must be LHS (all in bounds).
        for (i, cfg) in batch[..65].iter().enumerate() {
            assert_config_in_bounds(cfg, &format!("LHS point {i}"));
        }
    }

    #[test]
    fn pareto_suggest_returns_lhs_first() {
        use crate::metrics::Metric;
        let metrics = vec![Metric::Trustworthiness, Metric::Continuity];
        let mut opt = ParEgoOptimizer::new(metrics, HyperParams::all_free());
        let mut rng = Rng::new(51);
        // Drain the LHS in small batches.
        let mut all_lhs = Vec::new();
        while !opt.lhs_queue.is_empty() || !opt.lhs_initialized {
            let batch = opt.suggest_batch(5, &mut rng);
            all_lhs.extend(batch);
            if opt.lhs_queue.is_empty() {
                break;
            }
        }
        assert_eq!(
            all_lhs.len(),
            65,
            "should have drained exactly 65 LHS points"
        );
        for (i, cfg) in all_lhs.iter().enumerate() {
            assert_config_in_bounds(cfg, &format!("LHS {i}"));
        }
    }

    #[test]
    fn pareto_suggest_gp_phase_after_lhs() {
        use crate::metrics::Metric;
        let metrics = vec![Metric::Trustworthiness, Metric::Continuity];
        let mut opt = ParEgoOptimizer::new(metrics, HyperParams::all_free());
        let mut rng = Rng::new(52);

        // Drain the LHS.
        opt.suggest_batch(65, &mut rng);

        // Provide enough observations for the GP.
        for _ in 0..30 {
            let cfg = random_config(&mut rng);
            opt.observe(cfg, vec![rng.uniform(), rng.uniform()]);
        }

        // GP phase: should produce valid configs.
        let batch = opt.suggest_batch(3, &mut rng);
        assert_eq!(batch.len(), 3);
        for (i, cfg) in batch.iter().enumerate() {
            assert_config_in_bounds(cfg, &format!("GP phase config {i}"));
        }
    }

    // ─── scalarize_trials_subset ──────────────────────────────────────────────

    #[test]
    fn scalarize_subset_uses_all_when_few() {
        use crate::metrics::Metric;
        let metrics = vec![Metric::Trustworthiness, Metric::Continuity];
        let mut opt = ParEgoOptimizer::new(metrics, HyperParams::all_free());
        let mut rng = Rng::new(60);
        for _ in 0..20 {
            opt.observe(random_config(&mut rng), vec![rng.uniform(), rng.uniform()]);
        }
        let weights = vec![0.5, 0.5];
        let scalar_trials = opt.scalarize_trials_subset(&weights, &mut rng);
        assert_eq!(
            scalar_trials.len(),
            20,
            "all 20 trials should be used when n < 25"
        );
    }

    #[test]
    fn scalarize_subset_caps_at_n_when_many() {
        use crate::metrics::Metric;
        let metrics = vec![Metric::Trustworthiness, Metric::Continuity];
        let mut opt = ParEgoOptimizer::new(metrics, HyperParams::all_free());
        let mut rng = Rng::new(61);
        let n = 40usize;
        for _ in 0..n {
            opt.observe(random_config(&mut rng), vec![rng.uniform(), rng.uniform()]);
        }
        let weights = vec![0.5, 0.5];
        let scalar_trials = opt.scalarize_trials_subset(&weights, &mut rng);
        assert_eq!(
            scalar_trials.len(),
            n,
            "subset should equal n trials when n >= 25"
        );
    }
}
