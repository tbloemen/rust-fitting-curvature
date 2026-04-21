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

/// Floor for curvature magnitude when building an Optimize spec from CLI args.
pub const DEFAULT_CURVATURE_MAG_MIN: f64 = 0.001;

pub const LR_MIN: f64 = 0.5;
pub const LR_MAX: f64 = 30.0;
pub const PERP_MIN: f64 = 0.0004;
pub const PERP_MAX: f64 = 0.03;
pub const CEN_MIN: f64 = 0.0;
pub const CEN_MAX: f64 = 2.0;
pub const GLW_MIN: f64 = 0.0;
pub const GLW_MAX: f64 = 1.0;
pub const NLW_MIN: f64 = 0.0;
pub const NLW_MAX: f64 = 0.02;
pub const EEF_MIN: f64 = 4.0;
pub const EEF_MAX: f64 = 24.0;

// ─── ParamSpec ────────────────────────────────────────────────────────────────

/// Whether a hyperparameter is held fixed or included in the BO search space.
#[derive(Debug, Clone)]
pub enum ParamSpec {
    Fixed(f64),
    Optimize { lo: f64, hi: f64, log_scale: bool },
}

impl ParamSpec {
    /// Sample a concrete value. Returns the fixed value for `Fixed`; draws
    /// uniformly (log or linear) for `Optimize`.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            ParamSpec::Fixed(v) => *v,
            ParamSpec::Optimize {
                lo,
                hi,
                log_scale: true,
            } => (rng.uniform() * (hi.ln() - lo.ln()) + lo.ln())
                .exp()
                .clamp(*lo, *hi),
            ParamSpec::Optimize {
                lo,
                hi,
                log_scale: false,
            } => (rng.uniform() * (hi - lo) + lo).clamp(*lo, *hi),
        }
    }

    /// Like `sample`, but wraps the result in `Fixed`. Used by `HyperParams::sample`.
    pub fn sample_fixed(&self, rng: &mut Rng) -> Self {
        ParamSpec::Fixed(self.sample(rng))
    }

    /// Perturb a value. No-op (returns fixed value) for `Fixed`.
    pub fn mutate(&self, current: f64, rng: &mut Rng) -> f64 {
        match self {
            ParamSpec::Fixed(v) => *v,
            ParamSpec::Optimize {
                lo,
                hi,
                log_scale: true,
            } => (current * 2.0_f64.powf((rng.uniform() - 0.5) * 1.0)).clamp(*lo, *hi),
            ParamSpec::Optimize {
                lo,
                hi,
                log_scale: false,
            } => (current + (rng.uniform() - 0.5) * (hi - lo) * 0.25).clamp(*lo, *hi),
        }
    }

    pub fn is_optimized(&self) -> bool {
        matches!(self, ParamSpec::Optimize { .. })
    }

    /// Extract the concrete value. Panics on `Optimize` — call `sample` first.
    pub fn value(&self) -> f64 {
        match self {
            ParamSpec::Fixed(v) => *v,
            ParamSpec::Optimize { .. } => {
                panic!("ParamSpec::value() called on Optimize — sample the HyperParams first")
            }
        }
    }
}

// ─── TrialConfig ──────────────────────────────────────────────────────────────

/// Full hyperparameter specification for one experiment.
///
/// Each numeric field is either `Fixed(v)` (held constant) or `Optimize{lo,hi}` (in the
/// BO search space).  A fully-sampled `HyperParams` (all fields `Fixed`) serves directly
/// as the trial config.
///
/// `scaling_loss_type` is always `MeanDistance` and `init_method` is always `Pca`
/// — both are hardcoded in `to_training_config`.
///
/// Canonical field order (also the GP input vector order):
///   learning_rate, perplexity_ratio, momentum_main, momentum_early,
///   centering_weight, global_loss_weight, norm_loss_weight,
///   early_exaggeration_factor, n_iterations, early_exaggeration_iterations,
///   curvature_magnitude, init_scale, embed_dim
#[derive(Debug, Clone)]
pub struct TrialConfig {
    pub learning_rate: ParamSpec,
    /// Fraction of n_points; converted to absolute perplexity in `to_training_config`.
    pub perplexity_ratio: ParamSpec,
    pub momentum_main: ParamSpec,
    pub momentum_early: ParamSpec,
    pub centering_weight: ParamSpec,
    pub global_loss_weight: ParamSpec,
    pub norm_loss_weight: ParamSpec,
    pub early_exaggeration_factor: ParamSpec,
    pub n_iterations: ParamSpec,
    pub early_exaggeration_iterations: ParamSpec,
    /// Unsigned magnitude; sign is supplied by the caller of `to_training_config`.
    pub curvature_magnitude: ParamSpec,
    pub init_scale: ParamSpec,
    pub embed_dim: ParamSpec,
}

impl TrialConfig {
    /// Number of `Optimize` (free) parameters — the dimensionality seen by the GP.
    pub fn free_param_count(&self) -> usize {
        self.specs().iter().filter(|s| s.is_optimized()).count()
    }

    /// All specs in canonical order (mirrors `to_gp_input` / `gp_param_names`).
    fn specs(&self) -> [&ParamSpec; 13] {
        [
            &self.learning_rate,
            &self.perplexity_ratio,
            &self.momentum_main,
            &self.momentum_early,
            &self.centering_weight,
            &self.global_loss_weight,
            &self.norm_loss_weight,
            &self.early_exaggeration_factor,
            &self.n_iterations,
            &self.early_exaggeration_iterations,
            &self.curvature_magnitude,
            &self.init_scale,
            &self.embed_dim,
        ]
    }

    /// Sample a new `HyperParams` where every `Optimize` field becomes `Fixed(sampled)`.
    /// This is the "trial config" — pass the result to `to_training_config`.
    pub fn sample(&self, rng: &mut Rng) -> Self {
        Self {
            learning_rate: self.learning_rate.sample_fixed(rng),
            perplexity_ratio: self.perplexity_ratio.sample_fixed(rng),
            momentum_main: self.momentum_main.sample_fixed(rng),
            momentum_early: self.momentum_early.sample_fixed(rng),
            centering_weight: self.centering_weight.sample_fixed(rng),
            global_loss_weight: self.global_loss_weight.sample_fixed(rng),
            norm_loss_weight: self.norm_loss_weight.sample_fixed(rng),
            early_exaggeration_factor: self.early_exaggeration_factor.sample_fixed(rng),
            n_iterations: self.n_iterations.sample_fixed(rng),
            early_exaggeration_iterations: self.early_exaggeration_iterations.sample_fixed(rng),
            curvature_magnitude: self.curvature_magnitude.sample_fixed(rng),
            init_scale: self.init_scale.sample_fixed(rng),
            embed_dim: self.embed_dim.sample_fixed(rng),
        }
    }

    /// Perturb a sampled `HyperParams` using this spec's bounds.
    /// Fixed fields in the spec are left unchanged; Optimize fields are nudged.
    pub fn mutate(&self, current: &Self, rng: &mut Rng) -> Self {
        macro_rules! maybe_mutate {
            ($field:ident) => {
                ParamSpec::Fixed(if rng.uniform() < 0.3 {
                    self.$field.mutate(current.$field.value(), rng)
                } else {
                    current.$field.value()
                })
            };
        }
        Self {
            learning_rate: maybe_mutate!(learning_rate),
            perplexity_ratio: maybe_mutate!(perplexity_ratio),
            momentum_main: maybe_mutate!(momentum_main),
            momentum_early: maybe_mutate!(momentum_early),
            centering_weight: maybe_mutate!(centering_weight),
            global_loss_weight: maybe_mutate!(global_loss_weight),
            norm_loss_weight: maybe_mutate!(norm_loss_weight),
            early_exaggeration_factor: maybe_mutate!(early_exaggeration_factor),
            n_iterations: maybe_mutate!(n_iterations),
            early_exaggeration_iterations: maybe_mutate!(early_exaggeration_iterations),
            curvature_magnitude: maybe_mutate!(curvature_magnitude),
            init_scale: maybe_mutate!(init_scale),
            embed_dim: maybe_mutate!(embed_dim),
        }
    }

    /// Build a `TrainingConfig` from a fully-sampled `HyperParams` (all fields `Fixed`).
    ///
    /// `curvature_sign`: -1.0 for hyperbolic, +1.0 for spherical, 0.0 for Euclidean.
    pub fn to_training_config(
        &self,
        n_points: usize,
        curvature_sign: f64,
        seed: u64,
    ) -> TrainingConfig {
        let perplexity = (self.perplexity_ratio.value() * n_points as f64).max(2.0);
        TrainingConfig {
            n_points,
            embed_dim: self.embed_dim.value() as usize,
            curvature: curvature_sign * self.curvature_magnitude.value(),
            perplexity,
            n_iterations: self.n_iterations.value() as usize,
            early_exaggeration_iterations: self.early_exaggeration_iterations.value() as usize,
            early_exaggeration_factor: self.early_exaggeration_factor.value(),
            learning_rate: self.learning_rate.value(),
            momentum_early: self.momentum_early.value(),
            momentum_main: self.momentum_main.value(),
            init_method: InitMethod::Pca,
            init_scale: self.init_scale.value(),
            centering_weight: self.centering_weight.value(),
            scaling_loss_type: ScalingLossType::MeanDistance,
            global_loss_weight: self.global_loss_weight.value(),
            norm_loss_weight: self.norm_loss_weight.value(),
            seed,
        }
    }

    // ─── Named constructors (experiment variants) ────────────────────────────

    fn base() -> Self {
        Self {
            learning_rate: ParamSpec::Optimize {
                lo: LR_MIN,
                hi: LR_MAX,
                log_scale: true,
            },
            perplexity_ratio: ParamSpec::Optimize {
                lo: PERP_MIN,
                hi: PERP_MAX,
                log_scale: true,
            },
            momentum_main: ParamSpec::Fixed(0.8),
            momentum_early: ParamSpec::Fixed(0.5),
            centering_weight: ParamSpec::Fixed(0.0),
            global_loss_weight: ParamSpec::Fixed(0.0),
            norm_loss_weight: ParamSpec::Fixed(0.0),
            early_exaggeration_factor: ParamSpec::Optimize {
                lo: EEF_MIN,
                hi: EEF_MAX,
                log_scale: false,
            },
            n_iterations: ParamSpec::Fixed(800.0),
            early_exaggeration_iterations: ParamSpec::Fixed(250.0),
            curvature_magnitude: ParamSpec::Fixed(0.0),
            init_scale: ParamSpec::Fixed(get_default_init_scale(2)),
            embed_dim: ParamSpec::Fixed(2.0),
        }
    }

    /// All three loss weights fixed to 0. Only lr, perp, eef are optimized.
    pub fn all_off() -> Self {
        Self::base()
    }

    /// Only centering_weight (MeanDistance scaling loss weight) is optimized.
    pub fn centering_only() -> Self {
        Self {
            centering_weight: ParamSpec::Optimize {
                lo: CEN_MIN,
                hi: CEN_MAX,
                log_scale: false,
            },
            ..Self::base()
        }
    }

    /// Only global_loss_weight is optimized.
    pub fn global_only() -> Self {
        Self {
            global_loss_weight: ParamSpec::Optimize {
                lo: GLW_MIN,
                hi: GLW_MAX,
                log_scale: false,
            },
            ..Self::base()
        }
    }

    /// Only norm_loss_weight is optimized.
    pub fn norm_only() -> Self {
        Self {
            norm_loss_weight: ParamSpec::Optimize {
                lo: NLW_MIN,
                hi: NLW_MAX,
                log_scale: false,
            },
            ..Self::base()
        }
    }

    /// All three loss weights are optimized alongside lr, perp, eef.
    pub fn all_free() -> Self {
        Self {
            centering_weight: ParamSpec::Optimize {
                lo: CEN_MIN,
                hi: CEN_MAX,
                log_scale: false,
            },
            global_loss_weight: ParamSpec::Optimize {
                lo: GLW_MIN,
                hi: GLW_MAX,
                log_scale: false,
            },
            norm_loss_weight: ParamSpec::Optimize {
                lo: NLW_MIN,
                hi: NLW_MAX,
                log_scale: false,
            },
            ..Self::base()
        }
    }
}

// ─── SearchSpace ──────────────────────────────────────────────────────────────

/// Wraps a `HyperParams` spec with an optimization direction for the GP.
///
/// Curvature optimization is encoded in `hyper_params.curvature_magnitude` —
/// if it is `Optimize`, the magnitude is included in the GP search space.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub direction: OptimizeDirection,
    pub hyper_params: TrialConfig,
}

impl SearchSpace {
    /// Sample a fresh `HyperParams` (all fields `Fixed`) according to the spec.
    pub fn sample(&self, rng: &mut Rng) -> TrialConfig {
        self.hyper_params.sample(rng)
    }

    /// Perturb a sampled `HyperParams` according to the spec.
    pub fn mutate_config(&self, config: &TrialConfig, rng: &mut Rng) -> TrialConfig {
        self.hyper_params.mutate(config, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fitting_core::synthetic_data::Rng;

    fn test_space() -> SearchSpace {
        let mut hp = TrialConfig::all_free();
        hp.curvature_magnitude = ParamSpec::Optimize {
            lo: 0.001,
            hi: 25.0,
            log_scale: true,
        };
        SearchSpace {
            direction: OptimizeDirection::Maximize,
            hyper_params: hp,
        }
    }

    fn curvature_spec() -> ParamSpec {
        ParamSpec::Optimize {
            lo: 0.001,
            hi: 25.0,
            log_scale: true,
        }
    }

    #[test]
    fn sample_curvature_magnitude_is_within_bounds() {
        let spec = curvature_spec();
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let v = spec.sample(&mut rng);
            assert!(
                (0.001..=25.0).contains(&v),
                "sample {v} out of [0.001, 25.0]"
            );
        }
    }

    #[test]
    fn sample_curvature_magnitude_is_varied() {
        let spec = curvature_spec();
        let mut rng = Rng::new(42);
        let samples: Vec<f64> = (0..200).map(|_| spec.sample(&mut rng)).collect();
        let min = samples.iter().cloned().fold(f64::MAX, f64::min);
        let max = samples.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            max / min > 100.0,
            "samples not varied enough: min={min:.4}, max={max:.4}, ratio={:.1}",
            max / min
        );
        let at_max = samples.iter().filter(|&&v| v >= 24.9).count();
        assert!(at_max < 20, "{at_max}/200 samples were at the upper bound");
    }

    #[test]
    fn mutate_curvature_magnitude_is_within_bounds() {
        let spec = curvature_spec();
        let mut rng = Rng::new(99);
        for start in [0.001, 0.1, 1.0, 10.0, 25.0] {
            for _ in 0..200 {
                let v = spec.mutate(start, &mut rng);
                assert!(
                    (0.001..=25.0).contains(&v),
                    "mutate({start}) → {v} out of bounds"
                );
            }
        }
    }

    #[test]
    fn mutate_curvature_magnitude_single_step_range() {
        let spec = curvature_spec();
        let mut rng = Rng::new(7);
        for &start in &[0.01f64, 0.1, 1.0, 5.0, 10.0] {
            for _ in 0..500 {
                let v = spec.mutate(start, &mut rng);
                let ratio = if v > start { v / start } else { start / v };
                assert!(
                    ratio <= 2.0_f64.sqrt() + 1e-9,
                    "single step too large from {start}: {v}, ratio={ratio:.3}"
                );
            }
        }
    }

    #[test]
    fn mutate_curvature_magnitude_random_walk_explores_range() {
        let spec = curvature_spec();
        let mut rng = Rng::new(7);
        let mut current = (0.001_f64 * 25.0_f64).sqrt();
        let mut min_seen = current;
        let mut max_seen = current;
        for _ in 0..200 {
            current = spec.mutate(current, &mut rng);
            if current < min_seen {
                min_seen = current;
            }
            if current > max_seen {
                max_seen = current;
            }
        }
        assert!(
            max_seen / min_seen > 10.0,
            "random walk not varied enough: min={min_seen:.4}, max={max_seen:.4}"
        );
    }

    #[test]
    fn mutate_curvature_magnitude_can_decrease_from_upper_bound() {
        let spec = curvature_spec();
        let mut rng = Rng::new(13);
        let mut current = 25.0_f64;
        for _ in 0..50 {
            current = spec.mutate(current, &mut rng);
        }
        assert!(
            current < 25.0 * 0.9,
            "walk stuck at upper bound after 50 steps: {current:.3}"
        );
    }

    #[test]
    fn mutate_from_zero_clamps_to_lo() {
        // Multiplicative mutation of 0.0 would stay at 0; clamp brings it to lo.
        let spec = curvature_spec();
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let v = spec.mutate(0.0, &mut rng);
            assert_eq!(v, 0.001, "mutate(0.0) should clamp to lo=0.001");
        }
    }

    #[test]
    fn hyperparams_sample_produces_all_fixed() {
        let spec = TrialConfig::all_free();
        let mut rng = Rng::new(42);
        let sampled = spec.sample(&mut rng);
        // Every field in a sampled HP must be Fixed.
        assert!(!sampled.learning_rate.is_optimized());
        assert!(!sampled.perplexity_ratio.is_optimized());
        assert!(!sampled.centering_weight.is_optimized());
        assert!(!sampled.curvature_magnitude.is_optimized());
    }

    #[test]
    fn hyperparams_to_training_config_basic() {
        let mut hp = TrialConfig::all_free().sample(&mut Rng::new(7));
        hp.curvature_magnitude = ParamSpec::Fixed(2.5);
        let tc = hp.to_training_config(500, -1.0, 99);
        assert_eq!(tc.n_points, 500);
        assert!(
            tc.curvature < 0.0,
            "hyperbolic curvature should be negative"
        );
        assert!((tc.curvature / -2.5 - 1.0).abs() < 1e-12);
        assert_eq!(tc.seed, 99);
    }

    #[test]
    fn test_space_curvature_is_optimized() {
        let space = test_space();
        assert!(space.hyper_params.curvature_magnitude.is_optimized());
    }
}
