use fitting_core::config::{InitMethod, ScalingLossType, TrainingConfig};
use fitting_core::matrices::get_default_init_scale;
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

#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub direction: OptimizeDirection,
}

/// Encode a u8 scaling-loss index as a `ScalingLossType`.
/// 0=None, 1=HardBarrier, 2=SoftplusBarrier, 3=Rms, 4=MeanDistance.
pub fn scaling_loss_from_u8(v: u8) -> ScalingLossType {
    match v {
        1 => ScalingLossType::HardBarrier,
        2 => ScalingLossType::SoftplusBarrier,
        3 => ScalingLossType::Rms,
        4 => ScalingLossType::MeanDistance,
        _ => ScalingLossType::None,
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
    /// 0=None, 1=HardBarrier, 2=SoftplusBarrier, 3=Rms, 4=MeanDistance
    pub scaling_loss: u8,
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
            scaling_loss_type: scaling_loss_from_u8(self.scaling_loss),
            global_loss_weight: self.global_loss_weight,
            norm_loss_weight: self.norm_loss_weight,
            seed,
        }
    }

    pub fn random(rng: &mut fitting_core::synthetic_data::Rng) -> Self {
        // Log-uniform on [a, b]: exp(uniform() * (ln(b) - ln(a)) + ln(a))
        let lr = (rng.uniform() * (300.0_f64.ln() - 0.5_f64.ln()) + 0.5_f64.ln())
            .exp()
            .clamp(0.5, 300.0);

        // perplexity_ratio in [0.0004, 0.01], log-uniform.
        // For n=5000 this yields perplexity in [2, 50]; scales automatically with dataset size.
        const PERP_RATIO_MIN: f64 = 0.0004;
        const PERP_RATIO_MAX: f64 = 0.01;
        let perp_ratio = (rng.uniform() * (PERP_RATIO_MAX.ln() - PERP_RATIO_MIN.ln())
            + PERP_RATIO_MIN.ln())
        .exp()
        .clamp(PERP_RATIO_MIN, PERP_RATIO_MAX);

        let momentum = rng.uniform() * 0.25 + 0.7; // [0.70, 0.95]

        let scaling_loss = (rng.uniform() * 5.0) as u8; // 0..=4
        let centering_weight = rng.uniform() * 2.0;
        let global_loss_weight = rng.uniform() * 2.0; // narrowed from [0, 50]
        let norm_loss_weight = rng.uniform() * 0.02; // narrowed from [0, 0.05]

        Self {
            learning_rate: lr,
            perplexity_ratio: perp_ratio,
            momentum_main: momentum,
            scaling_loss,
            centering_weight,
            global_loss_weight,
            norm_loss_weight,
        }
    }

    pub fn mutate(&self, rng: &mut fitting_core::synthetic_data::Rng) -> Self {
        let mut cfg = self.clone();
        if rng.uniform() < 0.3 {
            cfg.learning_rate =
                (cfg.learning_rate * 2.0_f64.powf((rng.uniform() - 0.5) * 1.0)).clamp(0.5, 300.0);
        }
        if rng.uniform() < 0.3 {
            cfg.perplexity_ratio = (cfg.perplexity_ratio
                * 2.0_f64.powf((rng.uniform() - 0.5) * 0.8))
            .clamp(0.0004, 0.01);
        }
        if rng.uniform() < 0.3 {
            cfg.momentum_main = (cfg.momentum_main + (rng.uniform() - 0.5) * 0.2).clamp(0.70, 0.95);
        }
        if rng.uniform() < 0.2 {
            cfg.scaling_loss = (rng.uniform() * 5.0) as u8;
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
