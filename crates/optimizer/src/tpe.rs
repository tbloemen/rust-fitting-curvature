use fitting_core::synthetic_data::Rng;

use crate::search_space::{OptimizeDirection, SearchSpace, TrialConfig};

#[derive(Debug, Clone)]
pub struct Trial {
    pub config: TrialConfig,
    pub metric: f64,
}

pub struct TpeOptimizer {
    trials: Vec<Trial>,
    direction: OptimizeDirection,
    n_ei_candidates: usize,
}

impl TpeOptimizer {
    pub fn new(space: SearchSpace) -> Self {
        Self {
            trials: Vec::new(),
            direction: space.direction,
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

        let threshold = self.compute_threshold();

        let mut best_config = self.trials[0].config.clone();
        let mut best_score = f64::MIN;

        for _ in 0..self.n_ei_candidates {
            let candidate = if rng.uniform() < 0.3 {
                let idx = (rng.uniform() * self.trials.len() as f64) as usize % self.trials.len();
                self.trials[idx].config.mutate(rng)
            } else {
                TrialConfig::random(rng)
            };

            let score = self.acquisition(&candidate, threshold);
            if score > best_score {
                best_score = score;
                best_config = candidate;
            }
        }

        self.local_search(&best_config, rng, threshold)
    }

    fn local_search(&self, initial: &TrialConfig, rng: &mut Rng, threshold: f64) -> TrialConfig {
        let mut current = initial.clone();
        let mut current_score = self.acquisition(&current, threshold);

        for _ in 0..50 {
            let mutated = current.mutate(rng);
            let score = self.acquisition(&mutated, threshold);

            if score > current_score {
                current = mutated;
                current_score = score;
            }
        }

        current
    }

    /// TPE acquisition: l(x) / g(x), where l models good trials and g models bad trials.
    /// We maximize this ratio to find candidates likely in the good region.
    fn acquisition(&self, config: &TrialConfig, threshold: f64) -> f64 {
        let (l, g) = self.compute_l_g(config, threshold);

        if g < 1e-12 {
            return 0.0;
        }

        l / g
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

        // Compute per-dimension mean of the sample distribution
        let mut means = vec![0.0; dim];
        for s in samples {
            for d in 0..dim {
                means[d] += s[d];
            }
        }
        for m in means.iter_mut() {
            *m /= n;
        }

        // Silverman's rule: h = sigma * (4 / (3n))^(1/5)
        let silverman = (4.0 / (3.0 * n)).powf(0.2);
        let bandwidths: Vec<f64> = (0..dim)
            .map(|d| {
                let variance = samples
                    .iter()
                    .map(|s| (s[d] - means[d]).powi(2))
                    .sum::<f64>()
                    / n;
                (variance.sqrt() * silverman).max(1e-6)
            })
            .collect();

        let norm = bandwidths.iter().product::<f64>();

        let mut total = 0.0;
        for s in samples {
            let exponent = (0..dim)
                .map(|d| ((point[d] - s[d]) / bandwidths[d]).powi(2))
                .sum::<f64>();
            total += (-0.5 * exponent).exp();
        }

        total / (n * norm)
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
}
