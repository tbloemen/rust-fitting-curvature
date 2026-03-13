/// Initialization method for embedding points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitMethod {
    Random,
    Pca,
}

/// Scaling loss strategy for hyperbolic embeddings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalingLossType {
    Rms,
    HardBarrier,
    SoftplusBarrier,
    MeanDistance,
    None,
}

/// Full training configuration, mirrors config.toml.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub n_points: usize,
    pub embed_dim: usize,
    pub curvature: f64,
    pub perplexity: f64,
    pub n_iterations: usize,
    pub early_exaggeration_iterations: usize,
    pub early_exaggeration_factor: f64,
    pub learning_rate: f64,
    pub momentum_early: f64,
    pub momentum_main: f64,
    pub init_method: InitMethod,
    pub init_scale: f64,
    pub centering_weight: f64,
    pub scaling_loss_type: ScalingLossType,
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            n_points: 1000,
            embed_dim: 2,
            curvature: -1.0,
            perplexity: 30.0,
            n_iterations: 1000,
            early_exaggeration_iterations: 250,
            early_exaggeration_factor: 4.0,
            learning_rate: 20.0,
            momentum_early: 0.5,
            momentum_main: 0.8,
            init_method: InitMethod::Pca,
            init_scale: 1.0,
            centering_weight: 0.5,
            scaling_loss_type: ScalingLossType::MeanDistance,
            seed: 42,
        }
    }
}
