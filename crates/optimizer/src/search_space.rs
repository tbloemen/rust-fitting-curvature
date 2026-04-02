use fitting_core::config::{InitMethod, ScalingLossType, TrainingConfig};
use fitting_core::matrices::get_default_init_scale;

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
        }
    }
}
