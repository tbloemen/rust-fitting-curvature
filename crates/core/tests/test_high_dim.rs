//! Tests for high-dimensional input data (e.g., MNIST-like scenarios).
//! Verifies the full pipeline works with many features and labeled data.

use fitting_core::config::{InitMethod, ScalingLossType, TrainingConfig};
use fitting_core::embedding::EmbeddingState;
use fitting_core::synthetic_data::Rng;

/// Generate MNIST-like data: n_points of dim features, pixel-like values in [0, 1].
fn create_mnist_like_data(n_points: usize, n_features: usize, seed: u64) -> (Vec<f64>, Vec<u32>) {
    let mut rng = Rng::new(seed);

    // Simulate pixel data: values in [0, 1]
    let data: Vec<f64> = (0..n_points * n_features).map(|_| rng.uniform()).collect();

    // Labels 0-9
    let labels: Vec<u32> = (0..n_points).map(|i| (i % 10) as u32).collect();

    (data, labels)
}

/// Generate clustered MNIST-like data where each label has a distinct mean.
fn create_clustered_data(
    n_points: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Vec<f64>, Vec<u32>) {
    let mut rng = Rng::new(seed);

    let mut data = vec![0.0; n_points * n_features];
    let mut labels = vec![0u32; n_points];

    for i in 0..n_points {
        let class = i % n_classes;
        labels[i] = class as u32;
        for f in 0..n_features {
            // Each class has a different mean offset in each feature
            let class_mean = if f % n_classes == class { 0.8 } else { 0.2 };
            data[i * n_features + f] = (class_mean + 0.1 * rng.normal()).clamp(0.0, 1.0);
        }
    }

    (data, labels)
}

#[test]
fn test_high_dim_euclidean_no_nan() {
    let (data, _labels) = create_mnist_like_data(100, 784, 42);

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: 0.0,
        perplexity: 15.0,
        n_iterations: 30,
        early_exaggeration_iterations: 10,
        learning_rate: 100.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    };

    let mut state = EmbeddingState::new(&data, 784, &config);
    state.run(|_| true);

    assert_eq!(state.n_points, 100);
    assert_eq!(state.ambient_dim, 2);
    assert!(
        !state.points.iter().any(|v| v.is_nan()),
        "NaN in high-dim Euclidean embeddings"
    );
    assert!(
        !state.points.iter().any(|v| v.is_infinite()),
        "Inf in high-dim Euclidean embeddings"
    );
}

#[test]
fn test_high_dim_hyperbolic_no_nan() {
    let (data, _labels) = create_mnist_like_data(100, 784, 42);

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 15.0,
        n_iterations: 30,
        early_exaggeration_iterations: 10,
        learning_rate: 50.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    };

    let mut state = EmbeddingState::new(&data, 784, &config);
    state.run(|_| true);

    assert_eq!(state.n_points, 100);
    assert_eq!(state.ambient_dim, 3);
    assert!(
        !state.points.iter().any(|v| v.is_nan()),
        "NaN in high-dim hyperbolic embeddings"
    );

    // Verify hyperboloid constraint: -x0^2 + x1^2 + x2^2 = -1
    for i in 0..100 {
        let x0 = state.points[i * 3];
        let x1 = state.points[i * 3 + 1];
        let x2 = state.points[i * 3 + 2];
        let constraint = -x0 * x0 + x1 * x1 + x2 * x2;
        assert!(
            (constraint + 1.0).abs() < 5e-3,
            "Hyperboloid constraint violated at point {i}: {constraint}"
        );
    }
}

#[test]
fn test_high_dim_spherical_no_nan() {
    let (data, _labels) = create_mnist_like_data(100, 784, 42);

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: 1.0,
        perplexity: 15.0,
        n_iterations: 30,
        early_exaggeration_iterations: 10,
        learning_rate: 50.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        scaling_loss_type: ScalingLossType::None,
        ..Default::default()
    };

    let mut state = EmbeddingState::new(&data, 784, &config);
    state.run(|_| true);

    assert_eq!(state.n_points, 100);
    assert_eq!(state.ambient_dim, 3);
    assert!(
        !state.points.iter().any(|v| v.is_nan()),
        "NaN in high-dim spherical embeddings"
    );

    // Verify unit sphere constraint
    for i in 0..100 {
        let mut norm_sq = 0.0;
        for d in 0..3 {
            norm_sq += state.points[i * 3 + d].powi(2);
        }
        assert!(
            (norm_sq - 1.0).abs() < 1e-4,
            "Spherical constraint violated at point {i}: norm_sq={norm_sq}"
        );
    }
}

#[test]
fn test_high_dim_loss_decreases() {
    let (data, _labels) = create_mnist_like_data(100, 784, 42);

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: 0.0,
        perplexity: 15.0,
        n_iterations: 100,
        early_exaggeration_iterations: 25,
        learning_rate: 100.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    };

    let mut losses = Vec::new();
    let mut state = EmbeddingState::new(&data, 784, &config);
    state.run(|s| {
        losses.push(s.loss);
        true
    });

    assert_eq!(losses.len(), 100);

    // Loss should decrease from early to late phase
    let early_avg: f64 = losses[..10].iter().sum::<f64>() / 10.0;
    let late_avg: f64 = losses[losses.len() - 10..].iter().sum::<f64>() / 10.0;
    assert!(
        late_avg < early_avg,
        "Loss did not decrease: early={early_avg:.4}, late={late_avg:.4}"
    );
}

#[test]
fn test_high_dim_clustered_data() {
    // Verify the pipeline handles data with clear cluster structure
    let (data, labels) = create_clustered_data(100, 200, 5, 42);

    assert_eq!(data.len(), 100 * 200);
    assert_eq!(labels.len(), 100);
    assert!(labels.iter().all(|&l| l < 5));

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: 0.0,
        perplexity: 15.0,
        n_iterations: 50,
        early_exaggeration_iterations: 20,
        learning_rate: 100.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    };

    let mut state = EmbeddingState::new(&data, 200, &config);
    state.run(|_| true);

    assert!(
        !state.points.iter().any(|v| v.is_nan()),
        "NaN in clustered high-dim embeddings"
    );
}

#[test]
fn test_high_dim_step_by_step() {
    // Verify step-by-step execution (matching how WASM runner calls it)
    let (data, _labels) = create_mnist_like_data(50, 784, 42);

    let config = TrainingConfig {
        n_points: 50,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 10.0,
        n_iterations: 20,
        early_exaggeration_iterations: 10,
        learning_rate: 50.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    };

    let mut state = EmbeddingState::new(&data, 784, &config);

    // Step one at a time like the WASM runner does
    for i in 0..20 {
        assert!(!state.is_done());
        assert_eq!(state.iteration, i);
        state.step();
        assert!(
            !state.points.iter().any(|v| v.is_nan()),
            "NaN at iteration {i}"
        );
    }

    assert!(state.is_done());
    assert_eq!(state.iteration, 20);
}

#[test]
fn test_high_dim_different_feature_counts() {
    // Test with various feature dimensions to ensure nothing breaks
    for &n_features in &[50, 100, 200, 500, 784] {
        let (data, _labels) = create_mnist_like_data(50, n_features, 42);

        let config = TrainingConfig {
            n_points: 50,
            embed_dim: 2,
            curvature: 0.0,
            perplexity: 10.0,
            n_iterations: 10,
            early_exaggeration_iterations: 5,
            learning_rate: 100.0,
            init_method: InitMethod::Random,
            init_scale: 0.001,
            ..Default::default()
        };

        let mut state = EmbeddingState::new(&data, n_features, &config);
        state.run(|_| true);

        assert!(
            !state.points.iter().any(|v| v.is_nan()),
            "NaN with n_features={n_features}"
        );
    }
}
