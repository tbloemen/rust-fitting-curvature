use fitting_core::synthetic_data::Rng;

use crate::search_space::{OptimizeDirection, SearchSpace, TrialConfig};

#[derive(Debug, Clone)]
pub struct Trial {
    pub config: TrialConfig,
    pub metric: f64,
}

pub struct TpeOptimizer {
    trials: Vec<Trial>,
    space: SearchSpace,
    direction: OptimizeDirection,
    n_ei_candidates: usize,
}

impl TpeOptimizer {
    pub fn new(space: SearchSpace) -> Self {
        let direction = space.direction;
        Self {
            trials: Vec::new(),
            space,
            direction,
            n_ei_candidates: 1000,
        }
    }

    pub fn observe(&mut self, config: TrialConfig, metric: f64) {
        self.trials.push(Trial { config, metric });
    }

    pub fn suggest(&self, rng: &mut Rng) -> TrialConfig {
        if self.trials.is_empty() {
            return TrialConfig::random(rng);
        }

        let best = self.best_trial();
        let threshold = self.compute_threshold();

        let mut best_config = self.trials[0].config.clone();
        let mut best_score = f64::MIN;

        for _ in 0..self.n_ei_candidates {
            let candidate = if rng.uniform() < 0.3 && !self.trials.is_empty() {
                let idx = (rng.uniform() * self.trials.len() as f64) as usize % self.trials.len();
                self.trials[idx].config.mutate(rng, &self.space)
            } else {
                TrialConfig::random(rng)
            };

            let score = self.expected_improvement(&candidate, best, threshold);
            if score > best_score {
                best_score = score;
                best_config = candidate;
            }
        }

        self.local_search(&best_config, rng, best, threshold)
    }

    fn local_search(
        &self,
        initial: &TrialConfig,
        rng: &mut Rng,
        best: f64,
        threshold: f64,
    ) -> TrialConfig {
        let mut current = initial.clone();
        let mut current_ei = self.expected_improvement(&current, best, threshold);

        for _ in 0..50 {
            let mutated = current.mutate(rng, &self.space);
            let ei = self.expected_improvement(&mutated, best, threshold);

            if ei > current_ei {
                current = mutated;
                current_ei = ei;
            }
        }

        current
    }

    fn expected_improvement(&self, config: &TrialConfig, best: f64, threshold: f64) -> f64 {
        let (l, g) = self.compute_l_g(config, threshold);

        if l < 1e-12 || g < 1e-12 {
            return 0.0;
        }

        let ratio = g / (l + g);

        match self.direction {
            OptimizeDirection::Maximize => {
                let z = (best - threshold).exp() * ratio;
                z * (1.0 + (z / (2.0 * ratio)).ln().exp() * (1.0 - ratio))
            }
            OptimizeDirection::Minimize => {
                let z = (threshold - best).exp() * ratio;
                z * (1.0 + (z / (2.0 * ratio)).ln().exp() * (1.0 - ratio))
            }
        }
    }

    fn compute_l_g(&self, config: &TrialConfig, threshold: f64) -> (f64, f64) {
        let better: Vec<Vec<f64>> = self
            .trials
            .iter()
            .filter(|t| match self.direction {
                OptimizeDirection::Maximize => t.metric > threshold,
                OptimizeDirection::Minimize => t.metric < threshold,
            })
            .map(|t| self.config_to_vec(&t.config))
            .collect();

        let worse: Vec<Vec<f64>> = self
            .trials
            .iter()
            .filter(|t| match self.direction {
                OptimizeDirection::Maximize => t.metric <= threshold,
                OptimizeDirection::Minimize => t.metric >= threshold,
            })
            .map(|t| self.config_to_vec(&t.config))
            .collect();

        let point = self.config_to_vec(config);
        let l = self.gaussian_kde(&better, &point);
        let g = self.gaussian_kde(&worse, &point);

        (l, g)
    }

    fn gaussian_kde(&self, samples: &[Vec<f64>], point: &[f64]) -> f64 {
        if samples.is_empty() {
            return 1.0;
        }

        let dim = point.len();
        let n = samples.len() as f64;

        let mut variances = vec![0.0; dim];
        for s in samples {
            for d in 0..dim {
                let diff = point[d] - s[d];
                variances[d] += diff * diff;
            }
        }
        for v in variances.iter_mut() {
            *v = (*v / n).max(1e-6);
        }

        let bandwidths: Vec<f64> = variances
            .iter()
            .map(|&v| (v * 0.5_f64.sqrt()).max(1e-6))
            .collect();

        let mut total = 0.0;
        for s in samples {
            let mut diff_sq = 0.0;
            for d in 0..dim {
                let diff = point[d] - s[d];
                diff_sq += diff * diff / (2.0 * bandwidths[d] * bandwidths[d]);
            }
            total += (-diff_sq).exp();
        }

        total / (n * self.geometric_harmonic_mean(&bandwidths, dim))
    }

    fn geometric_harmonic_mean(&self, vals: &[f64], dim: usize) -> f64 {
        let n = dim;
        let log_sum: f64 = vals.iter().map(|v| v.ln()).sum();
        let mean_log = log_sum / n as f64;

        let inv_sum: f64 = vals.iter().map(|v| 1.0 / v).sum();
        let harm_mean = n as f64 / inv_sum;

        (mean_log.exp() + harm_mean) / 2.0
    }

    fn config_to_vec(&self, config: &TrialConfig) -> Vec<f64> {
        config.as_f64_array()
    }

    fn compute_threshold(&self) -> f64 {
        let n = self.trials.len();
        let gamma = 0.25_f64;

        let mut metrics: Vec<f64> = self.trials.iter().map(|t| t.metric).collect();

        match self.direction {
            OptimizeDirection::Maximize => {
                metrics.sort_by(|a, b| b.partial_cmp(a).unwrap());
            }
            OptimizeDirection::Minimize => {
                metrics.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
        }

        let idx = ((n as f64) * gamma) as usize;
        metrics[idx.min(n - 1)]
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

    pub fn n_trials(&self) -> usize {
        self.trials.len()
    }
}
