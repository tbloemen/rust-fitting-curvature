//! Integration tests for t-SNE embedding workflow.
//! Ported from Python test/test_integration.py

use fitting_core::config::{InitMethod, ScalingLossType, TrainingConfig};
use fitting_core::embedding::EmbeddingState;
use fitting_core::matrices::get_default_init_scale;
use fitting_core::synthetic_data::Rng;

fn create_test_data(n_samples: usize, n_features: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n_samples * n_features).map(|_| rng.normal()).collect()
}

fn run_embedding(data: &[f64], n_features: usize, config: &TrainingConfig) -> EmbeddingState {
    let mut state = EmbeddingState::new(data, n_features, config);
    state.run(|_| true);
    state
}

#[test]
fn test_integration_hyperbolic() {
    let data = create_test_data(100, 10, 42);
    let init_scale = get_default_init_scale(2);

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 15.0,
        n_iterations: 50,
        early_exaggeration_iterations: 20,
        learning_rate: 50.0,
        init_method: InitMethod::Random,
        init_scale,
        ..Default::default()
    };

    let state = run_embedding(&data, 10, &config);

    assert_eq!(state.n_points, 100);
    assert_eq!(state.ambient_dim, 3); // embed_dim + 1 for hyperboloid
    assert!(
        !state.points.iter().any(|v| v.is_nan()),
        "NaN in hyperbolic embeddings"
    );
    assert!(
        !state.points.iter().any(|v| v.is_infinite()),
        "Inf in hyperbolic embeddings"
    );
}

#[test]
fn test_integration_euclidean() {
    let data = create_test_data(100, 10, 42);
    let init_scale = get_default_init_scale(2);

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: 0.0,
        perplexity: 15.0,
        n_iterations: 50,
        early_exaggeration_iterations: 20,
        learning_rate: 100.0,
        init_method: InitMethod::Random,
        init_scale,
        ..Default::default()
    };

    let state = run_embedding(&data, 10, &config);

    assert_eq!(state.n_points, 100);
    assert_eq!(state.ambient_dim, 2);
    assert!(
        !state.points.iter().any(|v| v.is_nan()),
        "NaN in Euclidean embeddings"
    );
}

#[test]
fn test_integration_spherical() {
    let data = create_test_data(100, 10, 42);
    let init_scale = get_default_init_scale(2);

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: 1.0,
        perplexity: 15.0,
        n_iterations: 50,
        early_exaggeration_iterations: 20,
        learning_rate: 50.0,
        init_method: InitMethod::Random,
        init_scale,
        scaling_loss_type: ScalingLossType::None,
        ..Default::default()
    };

    let state = run_embedding(&data, 10, &config);

    assert_eq!(state.n_points, 100);
    assert_eq!(state.ambient_dim, 3);
    assert!(
        !state.points.iter().any(|v| v.is_nan()),
        "NaN in spherical embeddings"
    );
}

#[test]
fn test_integration_multiple_curvatures() {
    let data = create_test_data(50, 10, 42);

    for &k in &[-1.0, 0.0, 1.0] {
        let config = TrainingConfig {
            n_points: 50,
            embed_dim: 2,
            curvature: k,
            perplexity: 10.0,
            n_iterations: 30,
            early_exaggeration_iterations: 10,
            learning_rate: 50.0,
            init_method: InitMethod::Random,
            init_scale: 0.001,
            scaling_loss_type: if k < 0.0 {
                ScalingLossType::HardBarrier
            } else {
                ScalingLossType::None
            },
            ..Default::default()
        };

        let state = run_embedding(&data, 10, &config);
        assert!(
            !state.points.iter().any(|v| v.is_nan()),
            "NaN in embeddings for k={k}"
        );
    }
}

#[test]
fn test_integration_manifold_constraints_preserved() {
    let data = create_test_data(80, 10, 42);

    // Test hyperbolic constraint: -x0^2 + ||x_spatial||^2 = -1
    let config_hyp = TrainingConfig {
        n_points: 80,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 15.0,
        n_iterations: 50,
        early_exaggeration_iterations: 20,
        learning_rate: 50.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    };

    let state_hyp = run_embedding(&data, 10, &config_hyp);
    for i in 0..80 {
        let x0 = state_hyp.points[i * 3];
        let x1 = state_hyp.points[i * 3 + 1];
        let x2 = state_hyp.points[i * 3 + 2];
        let constraint = -x0 * x0 + x1 * x1 + x2 * x2;
        assert!(
            (constraint + 1.0).abs() < 5e-3,
            "Hyperboloid constraint violated at point {i}: {constraint}"
        );
    }

    // Test spherical constraint: ||x||^2 = 1
    let config_sph = TrainingConfig {
        n_points: 80,
        embed_dim: 2,
        curvature: 1.0,
        perplexity: 15.0,
        n_iterations: 50,
        early_exaggeration_iterations: 20,
        learning_rate: 50.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        scaling_loss_type: ScalingLossType::None,
        ..Default::default()
    };

    let state_sph = run_embedding(&data, 10, &config_sph);
    for i in 0..80 {
        let mut norm_sq = 0.0;
        for d in 0..3 {
            norm_sq += state_sph.points[i * 3 + d].powi(2);
        }
        assert!(
            (norm_sq - 1.0).abs() < 1e-4,
            "Spherical constraint violated at point {i}: norm_sq={norm_sq}"
        );
    }
}

#[test]
fn test_integration_different_perplexities() {
    let data = create_test_data(100, 10, 42);

    for &perp in &[5.0, 15.0, 30.0] {
        let config = TrainingConfig {
            n_points: 100,
            embed_dim: 2,
            curvature: 0.0,
            perplexity: perp,
            n_iterations: 30,
            early_exaggeration_iterations: 10,
            learning_rate: 100.0,
            init_method: InitMethod::Random,
            init_scale: 0.001,
            ..Default::default()
        };

        let state = run_embedding(&data, 10, &config);
        assert!(
            !state.points.iter().any(|v| v.is_nan()),
            "NaN for perplexity={perp}"
        );
    }
}

#[test]
fn test_integration_loss_callback() {
    let data = create_test_data(60, 5, 42);

    let config = TrainingConfig {
        n_points: 60,
        embed_dim: 2,
        curvature: 0.0,
        perplexity: 10.0,
        n_iterations: 100,
        early_exaggeration_iterations: 25,
        learning_rate: 100.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    };

    let mut losses = Vec::new();
    let mut state = EmbeddingState::new(&data, 5, &config);
    state.run(|s| {
        losses.push(s.loss);
        true
    });

    assert_eq!(losses.len(), 100);
    // Loss should generally decrease: compare first 10% vs last 10%
    let early_avg: f64 = losses[..10].iter().sum::<f64>() / 10.0;
    let late_avg: f64 = losses[losses.len() - 10..].iter().sum::<f64>() / 10.0;
    assert!(
        late_avg < early_avg,
        "Loss did not decrease: early={early_avg:.4}, late={late_avg:.4}"
    );
}
