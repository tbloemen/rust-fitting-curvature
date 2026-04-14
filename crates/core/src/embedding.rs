use crate::affinities::{compute_perplexity_affinities, compute_perplexity_affinities_from_distances};
use crate::config::{InitMethod, TrainingConfig};
use crate::kernels::compute_q_matrix_with_distances;
use crate::kl_divergence::{compute_global_similarities, kl_gradient, kl_loss, norm_loss_gradient};
use crate::manifolds;
use crate::manifolds::Manifold;
use crate::matrices::{compute_euclidean_distance_matrix, pca, pca_from_distances};
use crate::optimizer::RiemannianSGDMomentum;
use crate::scaling_loss;

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
    /// Globally-normalized input similarities p̂_ij for the Zhou & Sharpee global loss.
    /// Only populated when `config.global_loss_weight > 0`.
    p_hat: Vec<f64>,
    /// Original input data, kept for metric computation.
    input_data: Vec<f64>,
    n_features: usize,
}

impl EmbeddingState {
    /// Initialize embedding state from input data and config.
    pub fn new(data: &[f64], n_features: usize, config: &TrainingConfig) -> Self {
        let n_points = config.n_points;
        let manifold = manifolds::create_manifold(config.curvature);
        let ambient_dim = manifold.ambient_dim(config.embed_dim);

        let p_base = compute_perplexity_affinities(data, n_points, n_features, config.perplexity);
        let points = match config.init_method {
            InitMethod::Pca => {
                let coords = pca(data, n_points, n_features, config.embed_dim, config.seed);
                lift_pca_to_manifold(
                    &coords,
                    n_points,
                    config.embed_dim,
                    ambient_dim,
                    config.init_scale,
                    config.curvature,
                    manifold.radius(),
                )
            }
            InitMethod::Random => {
                manifold.init_points(n_points, config.embed_dim, config.init_scale, config.seed)
            }
        };
        let optimizer = RiemannianSGDMomentum::new(
            config.learning_rate,
            config.momentum_early,
            n_points,
            ambient_dim,
        );

        // Precompute p̂ from input Euclidean distances (fixed throughout training).
        let p_hat = if config.global_loss_weight > 0.0 {
            let input_dists = compute_euclidean_distance_matrix(data, n_points, n_features);
            compute_global_similarities(&input_dists, n_points)
        } else {
            Vec::new()
        };

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
            p_hat,
            input_data: data.to_vec(),
            n_features,
        }
    }

    /// Initialize embedding state from a pre-computed pairwise distance matrix.
    ///
    /// Uses `compute_perplexity_affinities_from_distances` so the t-SNE affinities
    /// are driven by the provided distances (e.g. tree distances for WordNet) rather
    /// than Euclidean distances in some feature space.
    ///
    /// `InitMethod::Pca` is handled via classical MDS (PCoA): the distance matrix is
    /// double-centered to form a Gram matrix whose top eigenvectors give coordinates.
    /// `norm_loss_weight` is ignored in this path (no input feature vectors exist).
    pub fn from_distances(distances: &[f64], n_points: usize, config: &TrainingConfig) -> Self {
        let manifold = manifolds::create_manifold(config.curvature);
        let ambient_dim = manifold.ambient_dim(config.embed_dim);

        let p_base =
            compute_perplexity_affinities_from_distances(distances, n_points, config.perplexity);

        let points = match config.init_method {
            InitMethod::Pca => {
                let coords = pca_from_distances(distances, n_points, config.embed_dim, config.seed);
                lift_pca_to_manifold(
                    &coords,
                    n_points,
                    config.embed_dim,
                    ambient_dim,
                    config.init_scale,
                    config.curvature,
                    manifold.radius(),
                )
            }
            InitMethod::Random => {
                manifold.init_points(n_points, config.embed_dim, config.init_scale, config.seed)
            }
        };

        let optimizer = RiemannianSGDMomentum::new(
            config.learning_rate,
            config.momentum_early,
            n_points,
            ambient_dim,
        );

        // Global loss: use input distances directly for p̂.
        let p_hat = if config.global_loss_weight > 0.0 {
            compute_global_similarities(distances, n_points)
        } else {
            Vec::new()
        };

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
            p_hat,
            // No input feature vectors in the distance-based path.
            input_data: Vec::new(),
            n_features: 0,
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

        // Radial scaling loss (hyperbolic only): penalizes spread of points from origin.
        // Gradient is in ambient coordinates; project to tangent space before adding.
        if self.config.centering_weight > 0.0 {
            let (scaling_loss, mut scale_grad) = scaling_loss::compute(
                self.config.scaling_loss_type,
                &self.points,
                n_points,
                ambient_dim,
                self.manifold.radius(),
                self.config.curvature,
            );
            self.manifold
                .project_to_tangent(&self.points, &mut scale_grad, n_points, ambient_dim);
            for k in 0..grad.len() {
                grad[k] += self.config.centering_weight * scale_grad[k];
            }
            self.loss += self.config.centering_weight * scaling_loss;
        }

        // Global t-SNE loss (Zhou & Sharpee): adds a KL term with globally-normalized
        // similarities, improving preservation of large-scale structure. Applies to all
        // geometries. The gradient has the same Riemannian structure as the KL gradient.
        if self.config.global_loss_weight > 0.0 {
            let q_hat = compute_global_similarities(&distances, n_points);
            let global_grad = kl_gradient(
                self.manifold.as_ref(),
                &self.points,
                &q_hat,
                &self.p_hat,
                &distances,
                n_points,
                ambient_dim,
            );
            for k in 0..grad.len() {
                grad[k] += self.config.global_loss_weight * global_grad[k];
            }
            self.loss += self.config.global_loss_weight * kl_loss(&q_hat, &self.p_hat, n_points);
        }

        // Norm loss: penalizes mismatch between ||x_i||² and ||y_i||².
        // Gradient is in ambient coordinates; project to tangent space first.
        // Skipped when initialized from distances (no input feature vectors).
        if self.config.norm_loss_weight > 0.0 && self.n_features > 0 {
            let (norm_loss, mut norm_grad) = norm_loss_gradient(
                &self.input_data,
                &self.points,
                n_points,
                self.n_features,
                ambient_dim,
            );
            self.manifold
                .project_to_tangent(&self.points, &mut norm_grad, n_points, ambient_dim);
            for k in 0..grad.len() {
                grad[k] += self.config.norm_loss_weight * norm_grad[k];
            }
            self.loss += self.config.norm_loss_weight * norm_loss;
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

/// Lift `embed_dim`-dimensional PCA coordinates onto the target manifold.
///
/// Scales coords so the max Euclidean norm equals `init_scale`, then:
/// - **Euclidean** (k=0): use as-is.
/// - **Hyperboloid** (k<0): spatial components = scaled PCA coords (indices 1..);
///   time component (index 0) = sqrt(r² + ||spatial||²).
/// - **Sphere** (k>0): spatial components = scaled PCA coords (indices 0..embed_dim);
///   last component (index embed_dim) = sqrt(r² - ||spatial||²).
fn lift_pca_to_manifold(
    coords: &[f64],
    n_points: usize,
    embed_dim: usize,
    ambient_dim: usize,
    init_scale: f64,
    curvature: f64,
    radius: f64,
) -> Vec<f64> {
    // Scale so the maximum per-point Euclidean norm equals init_scale.
    let max_norm = coords
        .chunks(embed_dim)
        .map(|p| p.iter().map(|x| x * x).sum::<f64>().sqrt())
        .fold(0.0f64, f64::max);
    let scale = if max_norm > 1e-12 {
        init_scale / max_norm
    } else {
        init_scale
    };

    let mut pts = vec![0.0f64; n_points * ambient_dim];
    for i in 0..n_points {
        let src = &coords[i * embed_dim..(i + 1) * embed_dim];
        let dst = &mut pts[i * ambient_dim..(i + 1) * ambient_dim];

        if curvature < 0.0 {
            // Hyperboloid layout: [time, x1, ..., x_d]
            let spatial_norm_sq: f64 = src.iter().map(|&x| (x * scale).powi(2)).sum();
            dst[0] = (radius * radius + spatial_norm_sq).sqrt();
            for d in 0..embed_dim {
                dst[1 + d] = src[d] * scale;
            }
        } else if curvature > 0.0 {
            // Sphere layout: [x1, ..., x_d, x_{d+1}]
            // Analogous to hyperboloid: place PCA coords in the first embed_dim slots,
            // compute the last coord from the sphere constraint ||x||^2 = r^2.
            // Requires ||src*scale|| <= r, guaranteed when init_scale <= radius.
            let spatial_norm_sq: f64 = src.iter().map(|&x| (x * scale).powi(2)).sum();
            let last_sq = radius * radius - spatial_norm_sq;
            for d in 0..embed_dim {
                dst[d] = src[d] * scale;
            }
            dst[embed_dim] = last_sq.max(0.0).sqrt();
        } else {
            // Euclidean: straight copy.
            for d in 0..embed_dim {
                dst[d] = src[d] * scale;
            }
        }
    }
    pts
}
