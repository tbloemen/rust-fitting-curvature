use fitting_core::config::{InitMethod, ScalingLossType, TrainingConfig};
use fitting_core::matrices::get_default_init_scale;
use fitting_core::synthetic_data::Rng;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizeDirection {
    Maximize,
    Minimize,
}

impl fmt::Display for OptimizeDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizeDirection::Maximize => write!(f, "maximize"),
            OptimizeDirection::Minimize => write!(f, "minimize"),
        }
    }
}

/// Minimum curvature magnitude — used as a floor when deriving GP bounds from CLI args.
pub const DEFAULT_CURVATURE_MAG_MIN: f64 = 0.001;

#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub direction: OptimizeDirection,
    /// Whether to treat curvature magnitude as a 7th BO hyperparameter.
    /// When true, `TrialConfig::curvature_magnitude` is sampled/mutated.
    pub optimize_curvature: bool,
    /// Inclusive lower bound for curvature magnitude sampling/mutation.
    pub curvature_mag_min: f64,
    /// Inclusive upper bound for curvature magnitude sampling/mutation.
    pub curvature_mag_max: f64,
}

impl SearchSpace {
    /// Sample curvature magnitude log-uniformly from `[curvature_mag_min, curvature_mag_max]`.
    pub fn sample_curvature_magnitude(&self, rng: &mut Rng) -> f64 {
        let lo = self.curvature_mag_min;
        let hi = self.curvature_mag_max;
        (rng.uniform() * (hi.ln() - lo.ln()) + lo.ln())
            .exp()
            .clamp(lo, hi)
    }

    /// Perturb a curvature magnitude by a multiplicative log-scale step.
    pub fn mutate_curvature_magnitude(&self, current: f64, rng: &mut Rng) -> f64 {
        let lo = self.curvature_mag_min;
        let hi = self.curvature_mag_max;
        (current * 2.0_f64.powf((rng.uniform() - 0.5) * 1.0)).clamp(lo, hi)
    }
}

/// Fixed iteration counts — not tuned, set to high-quality defaults.
pub const FIXED_N_ITERATIONS: usize = 800;
pub const FIXED_EARLY_EXAG_ITERATIONS: usize = 250;

#[derive(Debug, Clone)]
pub struct TrialConfig {
    pub learning_rate: f64,
    /// Perplexity expressed as a fraction of n_points.
    /// Actual perplexity = max(2.0, perplexity_ratio * n_points).
    pub perplexity_ratio: f64,
    pub momentum_main: f64,
    pub centering_weight: f64,
    pub global_loss_weight: f64,
    pub norm_loss_weight: f64,
    /// Curvature magnitude (> 0). Only used when SearchSpace::optimize_curvature = true;
    /// set to 0.0 in fixed-curvature mode (the sign-assigned curvature is passed separately).
    pub curvature_magnitude: f64,
}

impl TrialConfig {
    pub fn to_training_config(&self, n_points: usize, curvature: f64, seed: u64) -> TrainingConfig {
        let perplexity = (self.perplexity_ratio * n_points as f64).max(2.0);
        TrainingConfig {
            n_points,
            embed_dim: 2,
            curvature,
            perplexity,
            n_iterations: FIXED_N_ITERATIONS,
            early_exaggeration_iterations: FIXED_EARLY_EXAG_ITERATIONS,
            early_exaggeration_factor: 12.0,
            learning_rate: self.learning_rate,
            momentum_early: 0.5,
            momentum_main: self.momentum_main,
            init_method: InitMethod::Pca,
            init_scale: get_default_init_scale(2),
            centering_weight: self.centering_weight,
            scaling_loss_type: ScalingLossType::MeanDistance,
            global_loss_weight: self.global_loss_weight,
            norm_loss_weight: self.norm_loss_weight,
            seed,
        }
    }

    pub fn random(rng: &mut fitting_core::synthetic_data::Rng) -> Self {
        // Log-uniform on [a, b]: exp(uniform() * (ln(b) - ln(a)) + ln(a))
        let lr = (rng.uniform() * (50.0_f64.ln() - 0.5_f64.ln()) + 0.5_f64.ln())
            .exp()
            .clamp(0.5, 50.0);

        const PERP_RATIO_MIN: f64 = 0.0004;
        const PERP_RATIO_MAX: f64 = 0.03;
        let perp_ratio = (rng.uniform() * (PERP_RATIO_MAX.ln() - PERP_RATIO_MIN.ln())
            + PERP_RATIO_MIN.ln())
        .exp()
        .clamp(PERP_RATIO_MIN, PERP_RATIO_MAX);

        let momentum = rng.uniform() * 0.4 + 0.6; // [0.60, 1.00]
        let centering_weight = rng.uniform() * 2.0;
        let global_loss_weight = rng.uniform() * 2.0;
        let norm_loss_weight = rng.uniform() * 0.02;

        Self {
            learning_rate: lr,
            perplexity_ratio: perp_ratio,
            momentum_main: momentum,
            centering_weight,
            global_loss_weight,
            norm_loss_weight,
            curvature_magnitude: 0.0,
        }
    }

    /// Perturb this config slightly, keeping all values within their valid ranges.
    /// Used for local search when maximising the EI acquisition function.
    pub fn mutate(&self, rng: &mut fitting_core::synthetic_data::Rng) -> Self {
        let mut cfg = self.clone();
        if rng.uniform() < 0.3 {
            cfg.learning_rate =
                (cfg.learning_rate * 2.0_f64.powf((rng.uniform() - 0.5) * 1.0)).clamp(0.5, 50.0);
        }
        if rng.uniform() < 0.3 {
            cfg.perplexity_ratio = (cfg.perplexity_ratio
                * 2.0_f64.powf((rng.uniform() - 0.5) * 0.8))
            .clamp(0.0004, 0.03);
        }
        if rng.uniform() < 0.3 {
            cfg.momentum_main = (cfg.momentum_main + (rng.uniform() - 0.5) * 0.2).clamp(0.60, 1.0);
        }
        if rng.uniform() < 0.3 {
            cfg.centering_weight =
                (cfg.centering_weight + (rng.uniform() - 0.5) * 0.5).clamp(0.0, 2.0);
        }
        if rng.uniform() < 0.3 {
            cfg.global_loss_weight =
                (cfg.global_loss_weight + (rng.uniform() - 0.5) * 0.8).clamp(0.0, 2.0);
        }
        if rng.uniform() < 0.3 {
            cfg.norm_loss_weight =
                (cfg.norm_loss_weight + (rng.uniform() - 0.5) * 0.008).clamp(0.0, 0.02);
        }
        cfg
    }
}
