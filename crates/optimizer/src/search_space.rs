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

pub const LR_MIN: f64 = 0.5;
pub const LR_MAX: f64 = 20.0;
pub const PERP_MIN: f64 = 0.0004;
pub const PERP_MAX: f64 = 0.03;
pub const MOM_MIN: f64 = 0.60;
pub const MOM_MAX: f64 = 1.0;
pub const CEN_MIN: f64 = 0.0;
pub const CEN_MAX: f64 = 2.0;
pub const GLW_MIN: f64 = 0.0;
pub const GLW_MAX: f64 = 1.0;
pub const NLW_MIN: f64 = 0.0;
pub const NLW_MAX: f64 = 0.02;

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

#[cfg(test)]
mod tests {
    use super::*;
    use fitting_core::synthetic_data::Rng;

    fn test_space() -> SearchSpace {
        SearchSpace {
            direction: OptimizeDirection::Maximize,
            optimize_curvature: true,
            curvature_mag_min: 0.001,
            curvature_mag_max: 25.0,
        }
    }

    #[test]
    fn sample_curvature_magnitude_is_within_bounds() {
        let space = test_space();
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let v = space.sample_curvature_magnitude(&mut rng);
            assert!(
                v >= space.curvature_mag_min && v <= space.curvature_mag_max,
                "sample {v} out of [{}, {}]",
                space.curvature_mag_min,
                space.curvature_mag_max
            );
        }
    }

    #[test]
    fn sample_curvature_magnitude_is_varied() {
        let space = test_space();
        let mut rng = Rng::new(42);
        let samples: Vec<f64> = (0..200)
            .map(|_| space.sample_curvature_magnitude(&mut rng))
            .collect();
        let min = samples.iter().cloned().fold(f64::MAX, f64::min);
        let max = samples.iter().cloned().fold(f64::MIN, f64::max);
        // With 200 log-uniform samples over [0.001, 25] we should see at least a 100x spread.
        assert!(
            max / min > 100.0,
            "samples not varied enough: min={min:.4}, max={max:.4}, ratio={:.1}",
            max / min
        );
        // Should not be stuck at the upper bound.
        let at_max = samples.iter().filter(|&&v| v >= 24.9).count();
        assert!(
            at_max < 20,
            "{at_max}/200 samples were at the upper bound (25.0) — sampling appears stuck"
        );
    }

    #[test]
    fn mutate_curvature_magnitude_is_within_bounds() {
        let space = test_space();
        let mut rng = Rng::new(99);
        for start in [0.001, 0.1, 1.0, 10.0, 25.0] {
            for _ in 0..200 {
                let v = space.mutate_curvature_magnitude(start, &mut rng);
                assert!(
                    v >= space.curvature_mag_min && v <= space.curvature_mag_max,
                    "mutate({start}) → {v} out of [{}, {}]",
                    space.curvature_mag_min,
                    space.curvature_mag_max
                );
            }
        }
    }

    #[test]
    fn mutate_curvature_magnitude_single_step_range() {
        // A single mutation multiplies/divides by at most 2^0.5 ≈ 1.41.
        // Verify this holds for a variety of starting values.
        let space = test_space();
        let mut rng = Rng::new(7);
        for &start in &[0.01f64, 0.1, 1.0, 5.0, 10.0] {
            for _ in 0..500 {
                let v = space.mutate_curvature_magnitude(start, &mut rng);
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
        // Chain mutations like local_search does.  A 200-step multiplicative random
        // walk (±0.5 bits per step) from the log-midpoint should span ≥ 10× range.
        let space = test_space();
        let mut rng = Rng::new(7);
        let mut current = (space.curvature_mag_min * space.curvature_mag_max).sqrt();
        let mut min_seen = current;
        let mut max_seen = current;
        for _ in 0..200 {
            current = space.mutate_curvature_magnitude(current, &mut rng);
            if current < min_seen {
                min_seen = current;
            }
            if current > max_seen {
                max_seen = current;
            }
        }
        assert!(
            max_seen / min_seen > 10.0,
            "random walk not varied enough: min={min_seen:.4}, max={max_seen:.4}, ratio={:.1}",
            max_seen / min_seen
        );
    }

    #[test]
    fn mutate_curvature_magnitude_can_decrease_from_upper_bound() {
        // If local search starts at the upper bound (e.g. warm-start data all at 25.0),
        // the walk must be able to move downward.
        let space = test_space();
        let mut rng = Rng::new(13);
        let mut current = space.curvature_mag_max;
        // After 50 steps, the walk should have gone below 20.
        for _ in 0..50 {
            current = space.mutate_curvature_magnitude(current, &mut rng);
        }
        assert!(
            current < space.curvature_mag_max * 0.9,
            "walk stuck at upper bound after 50 steps: {current:.3}"
        );
    }

    #[test]
    fn mutate_from_zero_does_not_panic_and_clamps_to_lo() {
        // If a warm-started trial had curvature_magnitude=0.0 (from random-mode results),
        // multiplicative mutation would yield 0*anything=0, which should clamp to lo.
        let space = test_space();
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let v = space.mutate_curvature_magnitude(0.0, &mut rng);
            assert_eq!(v, space.curvature_mag_min, "mutate(0.0) should clamp to lo");
        }
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
        let lr = (rng.uniform() * (LR_MAX.ln() - LR_MIN.ln()) + LR_MIN.ln())
            .exp()
            .clamp(LR_MIN, LR_MAX);
        let perp_ratio = (rng.uniform() * (PERP_MAX.ln() - PERP_MIN.ln()) + PERP_MIN.ln())
            .exp()
            .clamp(PERP_MIN, PERP_MAX);
        let momentum = rng.uniform() * (MOM_MAX - MOM_MIN) + MOM_MIN;
        let centering_weight = rng.uniform() * CEN_MAX;
        let global_loss_weight = rng.uniform() * GLW_MAX;
        let norm_loss_weight = rng.uniform() * NLW_MAX;

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
            cfg.learning_rate = (cfg.learning_rate * 2.0_f64.powf((rng.uniform() - 0.5) * 1.0))
                .clamp(LR_MIN, LR_MAX);
        }
        if rng.uniform() < 0.3 {
            cfg.perplexity_ratio =
                (cfg.perplexity_ratio * 2.0_f64.powf((rng.uniform() - 0.5) * 0.8))
                    .clamp(PERP_MIN, PERP_MAX);
        }
        if rng.uniform() < 0.3 {
            cfg.momentum_main =
                (cfg.momentum_main + (rng.uniform() - 0.5) * 0.2).clamp(MOM_MIN, MOM_MAX);
        }
        if rng.uniform() < 0.3 {
            cfg.centering_weight =
                (cfg.centering_weight + (rng.uniform() - 0.5) * 0.5).clamp(CEN_MIN, CEN_MAX);
        }
        if rng.uniform() < 0.3 {
            cfg.global_loss_weight =
                (cfg.global_loss_weight + (rng.uniform() - 0.5) * 0.4).clamp(GLW_MIN, GLW_MAX);
        }
        if rng.uniform() < 0.3 {
            cfg.norm_loss_weight =
                (cfg.norm_loss_weight + (rng.uniform() - 0.5) * 0.008).clamp(NLW_MIN, NLW_MAX);
        }
        cfg
    }
}
