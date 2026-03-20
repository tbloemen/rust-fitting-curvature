use fitting_core::config::{InitMethod, ScalingLossType, TrainingConfig};
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
pub enum ParamKind {
    Continuous(f64, f64),
    LogUniform(f64, f64),
    Int(i64, i64),
    Categorical(&'static [f64]),
    CategoricalInit(&'static [InitMethod]),
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: &'static str,
    pub kind: ParamKind,
}

#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub params: Vec<Parameter>,
    pub direction: OptimizeDirection,
}

impl SearchSpace {
    pub fn default_tsne() -> Self {
        Self {
            params: vec![
                Parameter {
                    name: "learning_rate",
                    kind: ParamKind::LogUniform(0.1, 500.0),
                },
                Parameter {
                    name: "perplexity",
                    kind: ParamKind::LogUniform(2.0, 100.0),
                },
                Parameter {
                    name: "momentum_main",
                    kind: ParamKind::Continuous(0.5, 0.95),
                },
                Parameter {
                    name: "init_scale",
                    kind: ParamKind::LogUniform(1e-5, 1.0),
                },
                Parameter {
                    name: "n_iterations",
                    kind: ParamKind::Int(200, 800),
                },
                Parameter {
                    name: "early_exaggeration_iterations",
                    kind: ParamKind::Int(50, 400),
                },
                Parameter {
                    name: "curvature",
                    kind: ParamKind::Categorical(&[-1.0, -0.5, 0.0, 0.5, 1.0]),
                },
                Parameter {
                    name: "init_method",
                    kind: ParamKind::CategoricalInit(&[InitMethod::Random, InitMethod::Pca]),
                },
            ],
            direction: OptimizeDirection::Maximize,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrialConfig {
    pub learning_rate: f64,
    pub perplexity: f64,
    pub momentum_main: f64,
    pub init_scale: f64,
    pub n_iterations: i64,
    pub early_exaggeration_iterations: i64,
    pub curvature: f64,
    pub init_method: InitMethod,
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
            init_method: self.init_method,
            init_scale: self.init_scale,
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
        let lr_base = (500.0_f64 / 0.1_f64).ln();
        let lr = (rng.uniform().exp2() * lr_base + (0.1_f64).ln())
            .exp()
            .clamp(0.1, 500.0);

        let perp_base = (100.0_f64 / 2.0_f64).ln();
        let perp = (rng.uniform().exp2() * perp_base + (2.0_f64).ln())
            .exp()
            .clamp(2.0, 100.0);

        let momentum = rng.uniform() * 0.45 + 0.5;

        let init_base = (1e-5_f64 / 1.0_f64).ln();
        let init_scale = (rng.uniform().exp2() * init_base + (1e-5_f64).ln())
            .exp()
            .clamp(1e-5, 1.0);

        let n_iterations = (rng.uniform() * 1300.0 + 200.0) as i64;
        let early_exag = (rng.uniform() * 350.0 + 50.0) as i64;

        let curvatures = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let curvature = curvatures[(rng.uniform() * 5.0) as usize % 5];

        let init_method = if rng.uniform() < 0.5 {
            InitMethod::Random
        } else {
            InitMethod::Pca
        };

        Self {
            learning_rate: lr,
            perplexity: perp,
            momentum_main: momentum,
            init_scale,
            n_iterations,
            early_exaggeration_iterations: early_exag,
            curvature,
            init_method,
        }
    }

    pub fn mutate(
        &self,
        rng: &mut fitting_core::synthetic_data::Rng,
        _space: &SearchSpace,
    ) -> Self {
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
            cfg.init_scale =
                (cfg.init_scale * 2.0_f64.powf((rng.uniform() - 0.5) * 1.5)).clamp(1e-5, 1.0);
        }
        if rng.uniform() < 0.3 {
            cfg.n_iterations =
                ((cfg.n_iterations as f64 + (rng.uniform() - 0.5) * 300.0) as i64).clamp(200, 1500);
        }
        if rng.uniform() < 0.3 {
            cfg.early_exaggeration_iterations =
                ((cfg.early_exaggeration_iterations as f64 + (rng.uniform() - 0.5) * 100.0) as i64)
                    .clamp(50, 400);
        }
        if rng.uniform() < 0.3 {
            let curvatures = [-1.0, -0.5, 0.0, 0.5, 1.0];
            cfg.curvature = curvatures[(rng.uniform() * 5.0) as usize % 5];
        }
        if rng.uniform() < 0.3 {
            cfg.init_method = match cfg.init_method {
                InitMethod::Random => InitMethod::Pca,
                InitMethod::Pca => InitMethod::Random,
            };
        }
        cfg
    }

    pub fn as_f64_array(&self) -> Vec<f64> {
        vec![
            self.learning_rate,
            self.perplexity,
            self.momentum_main,
            self.init_scale,
            self.n_iterations as f64,
            self.early_exaggeration_iterations as f64,
            self.curvature,
        ]
    }

    pub fn dim(&self) -> usize {
        8
    }
}
