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

impl SearchSpace {
    pub fn default_tsne() -> Self {
        Self {
            direction: OptimizeDirection::Maximize,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrialConfig {
    pub learning_rate: f64,
    pub perplexity: f64,
    pub momentum_main: f64,
    pub n_iterations: i64,
    pub early_exaggeration_iterations: i64,
    pub curvature: f64,
}

impl TrialConfig {
    pub fn to_training_config(&self, n_points: usize, seed: u64) -> TrainingConfig {
        TrainingConfig {
            n_points,
            embed_dim: 2,
            curvature: self.curvature,
            perplexity: self.perplexity,
            n_iterations: self.n_iterations as usize,
            early_exaggeration_iterations: self.early_exaggeration_iterations as usize,
            early_exaggeration_factor: 12.0,
            learning_rate: self.learning_rate,
            momentum_early: 0.5,
            momentum_main: self.momentum_main,
            init_method: InitMethod::Pca,
            init_scale: get_default_init_scale(2),
            centering_weight: 0.0,
            scaling_loss_type: if self.curvature < 0.0 {
                ScalingLossType::HardBarrier
            } else {
                ScalingLossType::None
            },
            global_loss_weight: 0.0,
            norm_loss_weight: 0.0,
            seed,
        }
    }

    pub fn random(rng: &mut fitting_core::synthetic_data::Rng) -> Self {
        // Log-uniform on [a, b]: exp(uniform() * (ln(b) - ln(a)) + ln(a))
        let lr = (rng.uniform() * (500.0_f64.ln() - 0.1_f64.ln()) + 0.1_f64.ln())
            .exp()
            .clamp(0.1, 500.0);

        let perp = (rng.uniform() * (100.0_f64.ln() - 2.0_f64.ln()) + 2.0_f64.ln())
            .exp()
            .clamp(2.0, 100.0);

        let momentum = rng.uniform() * 0.45 + 0.5;

        let n_iterations = (rng.uniform() * 600.0 + 200.0) as i64;
        let early_exag = (rng.uniform() * 350.0 + 50.0) as i64;

        let curvature = rng.uniform() * 2.0 - 1.0;

        Self {
            learning_rate: lr,
            perplexity: perp,
            momentum_main: momentum,
            n_iterations,
            early_exaggeration_iterations: early_exag,
            curvature,
        }
    }

    pub fn mutate(&self, rng: &mut fitting_core::synthetic_data::Rng) -> Self {
        let mut cfg = self.clone();
        if rng.uniform() < 0.3 {
            cfg.learning_rate =
                (cfg.learning_rate * 2.0_f64.powf((rng.uniform() - 0.5) * 1.0)).clamp(0.1, 500.0);
        }
        if rng.uniform() < 0.3 {
            cfg.perplexity =
                (cfg.perplexity * 2.0_f64.powf((rng.uniform() - 0.5) * 0.8)).clamp(2.0, 100.0);
        }
        if rng.uniform() < 0.3 {
            cfg.momentum_main = (cfg.momentum_main + (rng.uniform() - 0.5) * 0.2).clamp(0.5, 0.95);
        }
        if rng.uniform() < 0.3 {
            cfg.n_iterations =
                ((cfg.n_iterations as f64 + (rng.uniform() - 0.5) * 300.0) as i64).clamp(200, 800);
        }
        if rng.uniform() < 0.3 {
            cfg.early_exaggeration_iterations =
                ((cfg.early_exaggeration_iterations as f64 + (rng.uniform() - 0.5) * 100.0) as i64)
                    .clamp(50, 400);
        }
        if rng.uniform() < 0.3 {
            cfg.curvature = (cfg.curvature + (rng.uniform() - 0.5) * 0.5).clamp(-1.0, 1.0);
        }
        cfg
    }

    pub fn as_f64_array(&self) -> Vec<f64> {
        vec![
            self.learning_rate,
            self.perplexity,
            self.momentum_main,
            self.n_iterations as f64,
            self.early_exaggeration_iterations as f64,
            self.curvature,
        ]
    }
}
