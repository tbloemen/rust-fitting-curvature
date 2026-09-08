//! Integration tests for t-SNE embedding workflow.
//! Ported from Python test/test_integration.py

use fitting_core::config::{InitMethod, ScalingLossType, TrainingConfig};
use fitting_core::embedding::EmbeddingState;
use fitting_core::metrics::{
    CLUSTER_DENSITY_MEASURE, CONTINUITY, CONTINUITY_MANIFOLD, DAVIES_BOULDIN_RATIO,
    NEIGHBORHOOD_HIT, NEIGHBORHOOD_HIT_MANIFOLD, NORMALIZED_STRESS, NORMALIZED_STRESS_MANIFOLD,
    SHEPARD_GOODNESS, SHEPARD_GOODNESS_MANIFOLD, TRUSTWORTHINESS, TRUSTWORTHINESS_MANIFOLD,
};
use fitting_core::matrices::get_default_init_scale;
use fitting_core::synthetic_data::{load_synthetic, Rng};
use fitting_core::visualisation::SphericalProjection;

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

#[test]
fn test_global_loss_no_nan() {
    // Verify global t-SNE loss (Zhou & Sharpee) runs without NaN/Inf for all geometries.
    let data = create_test_data(60, 5, 42);

    for &curvature in &[-1.0, 0.0, 1.0] {
        let config = TrainingConfig {
            n_points: 60,
            embed_dim: 2,
            curvature,
            perplexity: 10.0,
            n_iterations: 30,
            early_exaggeration_iterations: 10,
            learning_rate: 20.0,
            init_method: InitMethod::Random,
            init_scale: 0.001,
            global_loss_weight: 5.0,
            ..Default::default()
        };

        let state = run_embedding(&data, 5, &config);
        assert!(
            !state.points.iter().any(|v| v.is_nan()),
            "NaN with global loss at curvature={curvature}"
        );
        assert!(
            !state.points.iter().any(|v| v.is_infinite()),
            "Inf with global loss at curvature={curvature}"
        );
    }
}

#[test]
fn test_pca_init_no_nan() {
    let data = create_test_data(100, 10, 42);

    for &curvature in &[-1.0, 0.0, 1.0] {
        let config = TrainingConfig {
            n_points: 100,
            embed_dim: 2,
            curvature,
            perplexity: 15.0,
            n_iterations: 10,
            early_exaggeration_iterations: 5,
            learning_rate: 50.0,
            init_method: InitMethod::Pca,
            init_scale: 1e-4,
            ..Default::default()
        };

        let state = run_embedding(&data, 10, &config);
        assert!(
            !state.points.iter().any(|v| v.is_nan()),
            "NaN in PCA init for curvature={curvature}"
        );
        assert!(
            !state.points.iter().any(|v| v.is_infinite()),
            "Inf in PCA init for curvature={curvature}"
        );
    }
}

#[test]
fn test_pca_init_hyperboloid_constraint() {
    let data = create_test_data(80, 10, 42);

    let config = TrainingConfig {
        n_points: 80,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 15.0,
        n_iterations: 0,
        init_method: InitMethod::Pca,
        init_scale: 1e-4,
        ..Default::default()
    };

    let state = EmbeddingState::new(&data, 10, &config);

    for i in 0..80 {
        let x0 = state.points[i * 3];
        let x1 = state.points[i * 3 + 1];
        let x2 = state.points[i * 3 + 2];
        let constraint = -x0 * x0 + x1 * x1 + x2 * x2;
        assert!(
            (constraint + 1.0).abs() < 1e-6,
            "Hyperboloid constraint violated at point {i}: {constraint}"
        );
    }
}

#[test]
fn test_pca_init_sphere_constraint() {
    let data = create_test_data(80, 10, 42);

    let config = TrainingConfig {
        n_points: 80,
        embed_dim: 2,
        curvature: 1.0,
        perplexity: 15.0,
        n_iterations: 0,
        init_method: InitMethod::Pca,
        init_scale: 1e-4,
        ..Default::default()
    };

    let state = EmbeddingState::new(&data, 10, &config);

    for i in 0..80 {
        let mut norm_sq = 0.0;
        for d in 0..3 {
            norm_sq += state.points[i * 3 + d].powi(2);
        }
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "Sphere constraint violated at point {i}: norm_sq={norm_sq}"
        );
    }
}

#[test]
fn test_pca_init_euclidean_preserves_variance() {
    let data = create_test_data(100, 5, 42);

    let config = TrainingConfig {
        n_points: 100,
        embed_dim: 2,
        curvature: 0.0,
        perplexity: 15.0,
        n_iterations: 0,
        init_method: InitMethod::Pca,
        init_scale: 1.0,
        ..Default::default()
    };

    let state = EmbeddingState::new(&data, 5, &config);

    let mut var = [0.0, 0.0];
    for i in 0..100 {
        for (d, var_item) in var.iter_mut().enumerate() {
            *var_item += state.points[i * 2 + d].powi(2);
        }
    }

    let total_var: f64 = var.iter().sum();
    assert!(
        total_var > 0.0,
        "PCA init should preserve some variance in Euclidean space"
    );
}

#[test]
fn test_pca_init_deterministic() {
    let data = create_test_data(50, 8, 123);

    let config = TrainingConfig {
        n_points: 50,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 10.0,
        n_iterations: 0,
        init_method: InitMethod::Pca,
        init_scale: 1e-4,
        seed: 42,
        ..Default::default()
    };

    let state1 = EmbeddingState::new(&data, 8, &config);
    let state2 = EmbeddingState::new(&data, 8, &config);

    for (a, b) in state1.points.iter().zip(state2.points.iter()) {
        assert!((a - b).abs() < 1e-10, "PCA init should be deterministic");
    }
}

// ---------------------------------------------------------------------------
// compute_metrics: on-demand end-of-training metrics
// ---------------------------------------------------------------------------

fn small_config(n: usize, n_iterations: usize) -> TrainingConfig {
    TrainingConfig {
        n_points: n,
        embed_dim: 2,
        curvature: 0.0,
        perplexity: 10.0,
        n_iterations,
        early_exaggeration_iterations: n_iterations / 4,
        learning_rate: 50.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    }
}

#[test]
fn test_compute_metrics_manual_call() {
    let data = create_test_data(50, 5, 42);
    let mut state = EmbeddingState::new(&data, 5, &small_config(50, 30));
    state.run(|_| true);
    let (m, _spread) = state.compute_metrics();
    assert!((0.0..=1.0).contains(&m[TRUSTWORTHINESS_MANIFOLD]));
    assert!((0.0..=1.0).contains(&m[TRUSTWORTHINESS]));
    assert!(m.get(NEIGHBORHOOD_HIT_MANIFOLD).is_none());
}

#[test]
fn test_compute_metrics_with_labels_gives_some() {
    let data = create_test_data(60, 5, 42);
    let labels: Vec<u32> = (0..60u32).map(|i| i / 20).collect();
    let mut state = EmbeddingState::new(&data, 5, &small_config(60, 30)).with_labels(labels);
    state.run(|_| true);
    let (m, _spread) = state.compute_metrics();
    assert!(m.get(NEIGHBORHOOD_HIT_MANIFOLD).is_some());
    assert!(m.get(NEIGHBORHOOD_HIT).is_some());
    assert!(m.get(CLUSTER_DENSITY_MEASURE).is_some());
    assert!(m.get(DAVIES_BOULDIN_RATIO).is_some());
}

#[test]
fn test_compute_metrics_values_in_range() {
    let data = create_test_data(50, 5, 42);
    let labels: Vec<u32> = (0..50u32).map(|i| i / 25).collect();
    let mut state = EmbeddingState::new(&data, 5, &small_config(50, 20)).with_labels(labels);
    state.run(|_| true);
    let (m, _spread) = state.compute_metrics();
    assert!((0.0..=1.0).contains(&m[TRUSTWORTHINESS_MANIFOLD]));
    assert!((0.0..=1.0).contains(&m[TRUSTWORTHINESS]));
    assert!((0.0..=1.0).contains(&m[CONTINUITY_MANIFOLD]));
    assert!((0.0..=1.0).contains(&m[CONTINUITY]));
    assert!(m[NORMALIZED_STRESS_MANIFOLD] >= 0.0);
    assert!(m[NORMALIZED_STRESS] >= 0.0);
    assert!((0.0..=1.0).contains(&m[SHEPARD_GOODNESS_MANIFOLD]));
    assert!((0.0..=1.0).contains(&m[SHEPARD_GOODNESS]));
    assert!(!m[TRUSTWORTHINESS_MANIFOLD].is_nan());
    assert!(!m[TRUSTWORTHINESS].is_nan());
}

#[test]
fn test_with_projection_spherical_no_nan() {
    let data = create_test_data(50, 5, 42);
    let mut cfg = small_config(50, 20);
    cfg.curvature = 1.0;
    let mut state =
        EmbeddingState::new(&data, 5, &cfg).with_projection(SphericalProjection::Stereographic);
    state.run(|_| true);
    let (m, _spread) = state.compute_metrics();
    assert!(
        !m[TRUSTWORTHINESS].is_nan(),
        "NaN in trustworthiness_2d"
    );
    assert!(
        !m[NORMALIZED_STRESS].is_nan(),
        "NaN in normalized_stress_2d"
    );
    assert!(
        !m[SHEPARD_GOODNESS].is_nan(),
        "NaN in shepard_goodness_2d"
    );
}

#[test]
fn test_with_projection_hyperbolic_no_nan() {
    let data = create_test_data(50, 5, 42);
    let mut cfg = small_config(50, 20);
    cfg.curvature = -1.0;
    let mut state = EmbeddingState::new(&data, 5, &cfg)
        .with_projection(SphericalProjection::AzimuthalEquidistant);
    state.run(|_| true);
    let (m, _spread) = state.compute_metrics();
    assert!(
        !m[TRUSTWORTHINESS].is_nan(),
        "NaN in trustworthiness_2d"
    );
    assert!(
        !m[NORMALIZED_STRESS].is_nan(),
        "NaN in normalized_stress_2d"
    );
}

/// The manifold and projected readings must actually be different numbers
/// under curvature, or reporting both says nothing — and `figures/exp4.rs`,
/// which is the evidence for having dropped the manifold objectives, would be
/// comparing a column against itself.
///
/// Replaces a `compute_snapshot` test that manufactured the difference by
/// passing coordinates unrelated to the distance matrix. `MetricContext`
/// derives both readings from one set of points, so the difference now has to
/// come from where it really comes from: the projection discarding curvature.
#[test]
fn test_manifold_and_projected_readings_differ_under_curvature() {
    let data = create_test_data(60, 5, 42);
    let mut cfg = small_config(60, 40);
    cfg.curvature = -1.0;
    let mut state = EmbeddingState::new(&data, 5, &cfg);
    state.run(|_| true);
    let (m, _spread) = state.compute_metrics();

    assert!(
        (m[NORMALIZED_STRESS] - m[NORMALIZED_STRESS_MANIFOLD]).abs() > 1e-9,
        "projected stress {} and manifold stress {} are indistinguishable on a \
         curved manifold",
        m[NORMALIZED_STRESS],
        m[NORMALIZED_STRESS_MANIFOLD]
    );
}

/// The converse, and the reason `normalized_stress`'s doc says the two
/// variants coincide for Euclidean embeddings: the optimal scale α is divided
/// out, and `project_to_2d` only rescales flat coordinates for display. If this
/// ever fails, the projection has started doing something to Euclidean output.
#[test]
fn test_the_two_readings_coincide_in_flat_space() {
    let data = create_test_data(50, 5, 42);
    let mut state = EmbeddingState::new(&data, 5, &small_config(50, 30));
    state.run(|_| true);
    let (m, _spread) = state.compute_metrics();

    assert!(
        (m[NORMALIZED_STRESS] - m[NORMALIZED_STRESS_MANIFOLD]).abs() < 1e-9,
        "flat stress readings diverged: {} vs {}",
        m[NORMALIZED_STRESS],
        m[NORMALIZED_STRESS_MANIFOLD]
    );
}

#[test]
fn test_compute_metrics_from_distances() {
    let mut rng = Rng::new(42);
    let n = 40;
    let mut dist = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = rng.uniform() * 5.0 + 0.1;
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    let cfg = TrainingConfig {
        n_points: n,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 8.0,
        n_iterations: 20,
        early_exaggeration_iterations: 5,
        learning_rate: 20.0,
        init_method: InitMethod::Random,
        init_scale: 0.001,
        ..Default::default()
    };
    let mut state = EmbeddingState::from_distances(&dist, n, &cfg);
    state.run(|_| true);
    let (m, _spread) = state.compute_metrics();
    assert!(!m[TRUSTWORTHINESS_MANIFOLD].is_nan());
    assert!(!m[TRUSTWORTHINESS].is_nan());
}

/// Regression: the hyperbolic feature norm loss must target the *bounded* input
/// Poincaré radius, not the raw ambient ‖x‖². Tree data places nodes near the
/// Poincaré boundary, where ‖x‖² diverges (~1e6), which previously produced a
/// loss of ~1e9 that dwarfed the KL gradient. With the bounded target the norm
/// loss is O(1) and stays comparable to the no-norm-loss run.
#[test]
fn test_tree_norm_loss_bounded() {
    let synth = load_synthetic("tree_structured", 300, 42).unwrap();

    let base = TrainingConfig {
        n_points: synth.n_points,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 30.0,
        n_iterations: 100,
        learning_rate: 20.0,
        init_method: InitMethod::Pca,
        ..Default::default()
    };

    let mut without = EmbeddingState::new(&synth.x, synth.ambient_dim, &base);
    without.run(|_| true);

    let with_cfg = TrainingConfig {
        norm_loss_weight: 10.0,
        ..base
    };
    let mut with = EmbeddingState::new(&synth.x, synth.ambient_dim, &with_cfg);
    with.run(|_| true);

    assert!(with.loss.is_finite(), "loss must be finite");
    // Bounded target ⇒ the norm-loss term is ≤ weight·1 (here ≤ 10): the total
    // stays in the KL regime, nowhere near the ~1e9 the raw-‖x‖² formulation
    // produced for this dataset (which would also trip this bound by ~7 orders).
    assert!(
        with.loss < 1000.0,
        "tree norm-loss should be bounded, got {}",
        with.loss
    );
    // The added term cannot exceed weight·max((r_y - r_x)²) = 10·1 = 10, so the
    // total must stay close to the no-norm-loss run (with generous slack).
    assert!(
        with.loss < without.loss + 15.0,
        "norm loss should not blow up the total: with={} without={}",
        with.loss,
        without.loss
    );
}
