use crate::affinities::compute_perplexity_affinities;
use crate::config::TrainingConfig;
use crate::kernels::compute_q_matrix_with_distances;
use crate::kl_divergence::kl_gradient;
use crate::kl_divergence::kl_loss;
use crate::manifolds;
use crate::manifolds::Manifold;
use crate::optimizer::RiemannianSGDMomentum;

/// Embedding state for step-by-step iteration.
///
/// Allows running one iteration at a time, suitable for animated rendering
/// where the caller needs to yield control between steps.
pub struct EmbeddingState {
    pub points: Vec<f64>,
    pub n_points: usize,
    pub ambient_dim: usize,
    pub iteration: usize,
    pub loss: f64,
    config: TrainingConfig,
    manifold: Box<dyn Manifold>,
    optimizer: RiemannianSGDMomentum,
    p_base: Vec<f64>,
    /// Original input data, kept for metric computation.
    input_data: Vec<f64>,
    n_features: usize,
}

impl EmbeddingState {
    /// Initialize embedding state from input data and config.
    pub fn new(data: &[f64], n_features: usize, config: &TrainingConfig) -> Self {
        let n_points = config.n_points;
        let manifold = manifolds::create_manifold(config.curvature, config.scaling_loss_type);
        let ambient_dim = manifold.ambient_dim(config.embed_dim);

        let p_base = compute_perplexity_affinities(data, n_points, n_features, config.perplexity);
        let points =
            manifold.init_points(n_points, config.embed_dim, config.init_scale, config.seed);
        let optimizer = RiemannianSGDMomentum::new(
            config.learning_rate,
            config.momentum_early,
            n_points,
            ambient_dim,
        );

        Self {
            points,
            n_points,
            ambient_dim,
            iteration: 0,
            loss: 0.0,
            config: config.clone(),
            manifold,
            optimizer,
            p_base,
            input_data: data.to_vec(),
            n_features,
        }
    }

    /// Run one training iteration. Returns the current phase name.
    pub fn step(&mut self) -> &str {
        let n_points = self.n_points;
        let ambient_dim = self.ambient_dim;

        // Phase transition
        if self.iteration == self.config.early_exaggeration_iterations {
            self.optimizer.set_momentum(self.config.momentum_main);
        }

        // Current P (with or without exaggeration)
        let p_current: Vec<f64> = if self.iteration < self.config.early_exaggeration_iterations {
            self.p_base
                .iter()
                .map(|&x| x * self.config.early_exaggeration_factor)
                .collect()
        } else {
            self.p_base.clone()
        };

        // Compute Q and distances
        let (q, distances) = compute_q_matrix_with_distances(
            self.manifold.as_ref(),
            &self.points,
            n_points,
            ambient_dim,
            1.0,
        );

        // Compute loss
        self.loss = kl_loss(&q, &p_current, n_points);

        // Compute Riemannian gradient (already a tangent vector)
        let mut grad = kl_gradient(
            self.manifold.as_ref(),
            &self.points,
            &q,
            &p_current,
            &distances,
            n_points,
            ambient_dim,
        );

        // Add scaling loss gradient (Euclidean, needs projection to tangent space)
        if self.config.centering_weight > 0.0 {
            let (_, mut scale_grad) =
                self.manifold
                    .scaling_loss(&self.points, n_points, ambient_dim);
            self.manifold
                .project_to_tangent(&self.points, &mut scale_grad, n_points, ambient_dim);
            for k in 0..grad.len() {
                grad[k] += self.config.centering_weight * scale_grad[k];
            }
        }

        // Optimizer step
        self.optimizer.step(
            self.manifold.as_ref(),
            &mut self.points,
            &grad,
            n_points,
            ambient_dim,
        );

        // Hard center
        self.manifold
            .center(&mut self.points, n_points, ambient_dim);

        let phase = if self.iteration < self.config.early_exaggeration_iterations {
            "early"
        } else {
            "main"
        };

        self.iteration += 1;
        phase
    }

    /// Whether all iterations have been completed.
    pub fn is_done(&self) -> bool {
        self.iteration >= self.config.n_iterations
    }

    /// Current training phase: "early" during exaggeration, "main" after.
    pub fn phase(&self) -> &str {
        if self.iteration < self.config.early_exaggeration_iterations {
            "early"
        } else {
            "main"
        }
    }

    /// Run all remaining iterations, calling `on_step` after each.
    /// Return `false` from the callback to stop early.
    pub fn run(&mut self, mut on_step: impl FnMut(&Self) -> bool) {
        while !self.is_done() {
            self.step();
            if !on_step(self) {
                break;
            }
        }
    }

    /// Access the training config.
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Compute the high-dimensional Euclidean distance matrix from stored input data.
    pub fn high_dim_distances(&self) -> Vec<f64> {
        crate::matrices::compute_euclidean_distance_matrix(
            &self.input_data,
            self.n_points,
            self.n_features,
        )
    }

    /// Compute the embedded pairwise distance matrix using the manifold metric.
    pub fn embedded_distances(&self) -> Vec<f64> {
        self.manifold
            .pairwise_distances(&self.points, self.n_points, self.ambient_dim)
    }
}
