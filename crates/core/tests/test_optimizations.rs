//! Verifies the 5000-sample performance work:
//!   1. correctness — the optimized kernel/gradient produce the same numbers as
//!      a naive reference, and disabling loss tracking does not change the
//!      embedding trajectory;
//!   2. speedup — the optimized kernel and gradient are measurably faster than
//!      the naive reference doing the same computation.
//!
//! The timing checks compare naive vs optimized *within the same build* (same
//! opt level), take the minimum over several repetitions to suppress scheduler
//! noise, and only assert the directional result (optimized < naive). The real
//! margin is large (~2x on the transcendental-heavy paths), so this is not
//! flaky even in an unoptimized `cargo test` build.

use fitting_core::config::{InitMethod, TrainingConfig};
use fitting_core::embedding::EmbeddingState;
use fitting_core::kernels::compute_q_matrix_with_distances;
use fitting_core::kl_divergence::{kl_gradient, kl_loss};
use fitting_core::manifolds::{self, Manifold};
use fitting_core::synthetic_data::Rng;
use std::time::{Duration, Instant};

// ───────────────────────── naive reference implementations ─────────────────
// These mirror the original (pre-optimization) code: a `powf` kernel and the
// full i≠j double loop that recomputed each pair's geometry twice.

fn naive_kernel(distances: &[f64], dof: f64) -> Vec<f64> {
    let exponent = -(dof + 1.0) / 2.0;
    distances
        .iter()
        .map(|&d| (1.0 + d * d / dof).powf(exponent))
        .collect()
}

fn naive_kl_gradient(
    manifold: &dyn Manifold,
    points: &[f64],
    q: &[f64],
    p: &[f64],
    distances: &[f64],
    n: usize,
    ad: usize,
) -> Vec<f64> {
    let k = manifold.curvature();
    let r_sq = manifold.radius() * manifold.radius();
    let mut grad = vec![0.0; n * ad];

    for i in 0..n {
        let oi = i * ad;
        for j in 0..n {
            if i == j {
                continue;
            }
            let idx = i * n + j;
            let d = distances[idx];
            let oj = j * ad;

            if k < 0.0 {
                let w = 1.0 / (1.0 + d * d);
                let coeff = 4.0 * (p[idx] - q[idx]) * w;
                let mut lorentz = -points[oi] * points[oj];
                for dim in 1..ad {
                    lorentz += points[oi + dim] * points[oj + dim];
                }
                let alpha = (-lorentz / r_sq).max(1.0);
                let scale = if alpha < 1.0 + 1e-10 {
                    1.0
                } else {
                    alpha.acosh() / (alpha * alpha - 1.0).sqrt()
                };
                for dim in 0..ad {
                    let u = points[oj + dim] - alpha * points[oi + dim];
                    grad[oi + dim] += coeff * (-scale * u);
                }
            } else if k > 0.0 {
                let w = 1.0 / (1.0 + d * d);
                let coeff = 4.0 * (p[idx] - q[idx]) * w;
                let mut inner = 0.0;
                for dim in 0..ad {
                    inner += points[oi + dim] * points[oj + dim];
                }
                let cos_theta = (inner / r_sq).clamp(-1.0 + 1e-7, 1.0 - 1e-7);
                let theta = cos_theta.acos();
                let sin_theta = theta.sin();
                let scale = if sin_theta < 1e-10 {
                    1.0
                } else {
                    theta / sin_theta
                };
                for dim in 0..ad {
                    let u = points[oj + dim] - cos_theta * points[oi + dim];
                    grad[oi + dim] += coeff * (-scale * u);
                }
            } else {
                let factor = 4.0 * (p[idx] - q[idx]) / (1.0 + d * d);
                for dim in 0..ad {
                    let diff = points[oi + dim] - points[oj + dim];
                    grad[oi + dim] += factor * diff;
                }
            }
        }
    }
    grad
}

// ───────────────────────────────── helpers ─────────────────────────────────

/// Build points on the manifold plus a symmetric, normalized P matrix and the
/// matching Q/distances — everything `kl_gradient` needs.
struct Setup {
    manifold: Box<dyn Manifold>,
    points: Vec<f64>,
    p: Vec<f64>,
    q: Vec<f64>,
    distances: Vec<f64>,
    n: usize,
    ad: usize,
}

impl Setup {
    fn new(curvature: f64, n: usize) -> Self {
        let embed_dim = 2;
        let manifold = manifolds::create_manifold(curvature);
        let ad = manifold.ambient_dim(embed_dim);
        let points = manifold.init_points(n, embed_dim, 0.3, 7);

        // Symmetric, normalized P (the actual t-SNE invariant the gradient relies on).
        let mut p = vec![0.0; n * n];
        let mut rng = Rng::new(123);
        for i in 0..n {
            for j in (i + 1)..n {
                let v = rng.uniform() + 0.1;
                p[i * n + j] = v;
                p[j * n + i] = v;
            }
        }
        let sum: f64 = p.iter().sum();
        for v in &mut p {
            *v /= sum;
        }

        let (q, distances) =
            compute_q_matrix_with_distances(manifold.as_ref(), &points, n, ad, 1.0);

        Self {
            manifold,
            points,
            p,
            q,
            distances,
            n,
            ad,
        }
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Minimum wall-clock over `reps` runs of `f` (robust to scheduler noise).
fn min_time(reps: usize, mut f: impl FnMut()) -> Duration {
    // Warm up once (caches, branch predictors) before measuring.
    f();
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
    }
    best
}

// ──────────────────────────── correctness checks ───────────────────────────

#[test]
fn optimized_gradient_matches_naive_all_geometries() {
    let n = 60;
    for &k in &[0.0, -1.0, -0.4, 1.0, 0.7] {
        let s = Setup::new(k, n);
        let optimized = kl_gradient(
            s.manifold.as_ref(),
            &s.points,
            &s.q,
            &s.p,
            &s.distances,
            s.n,
            s.ad,
        );
        let naive = naive_kl_gradient(
            s.manifold.as_ref(),
            &s.points,
            &s.q,
            &s.p,
            &s.distances,
            s.n,
            s.ad,
        );
        let diff = max_abs_diff(&optimized, &naive);
        // The symmetric rewrite reorders nothing that the original didn't also
        // compute, so the result is bit-for-bit identical.
        assert_eq!(
            diff, 0.0,
            "k={k}: optimized gradient differs from naive by {diff:e}"
        );
    }
}

#[test]
fn loss_tracking_toggle_does_not_change_trajectory() {
    // Disabling loss tracking must only skip a diagnostic; the points produced
    // each step must be identical with it on or off.
    let n = 80;
    let n_features = 5;
    let mut rng = Rng::new(2024);
    let data: Vec<f64> = (0..n * n_features).map(|_| rng.normal()).collect();

    let config = TrainingConfig {
        n_points: n,
        embed_dim: 2,
        curvature: -1.0,
        perplexity: 15.0,
        n_iterations: 60,
        early_exaggeration_iterations: 20,
        early_exaggeration_factor: 12.0,
        init_method: InitMethod::Random,
        ..TrainingConfig::default()
    };

    let mut with_loss = EmbeddingState::new(&data, n_features, &config).with_loss_tracking(true);
    let mut no_loss = EmbeddingState::new(&data, n_features, &config).with_loss_tracking(false);

    for _ in 0..config.n_iterations {
        with_loss.step();
        no_loss.step();
        assert_eq!(
            max_abs_diff(&with_loss.points, &no_loss.points),
            0.0,
            "trajectory diverged at iteration {} after toggling loss tracking",
            with_loss.iteration
        );
    }
    // The tracked run actually populated a finite loss; the other left it at 0.
    assert!(with_loss.loss.is_finite() && with_loss.loss != 0.0);
    assert_eq!(no_loss.loss, 0.0);
}

// ───────────────────────────── speedup checks ──────────────────────────────

/// Naive Q matrix: `powf` kernel + normalization, mirroring the original code.
fn naive_compute_q(
    manifold: &dyn Manifold,
    points: &[f64],
    n: usize,
    ad: usize,
) -> (Vec<f64>, Vec<f64>) {
    let dist = manifold.pairwise_distances(points, n, ad);
    let mut kernel = naive_kernel(&dist, 1.0);
    for i in 0..n {
        kernel[i * n + i] = 0.0;
    }
    let total: f64 = kernel.iter().sum();
    let total = if total == 0.0 { 1e-10 } else { total };
    for v in &mut kernel {
        *v /= total;
    }
    (kernel, dist)
}

/// Aggregate per-iteration work: this is the headline number for the sweeps.
///
/// NAIVE mirrors the original `step`: re-scale (clone) P every iteration, build
/// Q, compute the KL loss, then the double-loop gradient. OPTIMIZED is the
/// current path: reuse a pre-scaled P (no per-step clone), skip the (unused)
/// loss, symmetric gradient. The Q/kernel computation is identical on both
/// sides, so the win comes purely from those three changes.
#[test]
fn full_step_work_is_faster_than_naive() {
    let reps = 10;
    let factor = 12.0; // early-exaggeration factor
    for &k in &[-1.0, 1.0, 0.0] {
        let s = Setup::new(k, 600);
        let p_early: Vec<f64> = s.p.iter().map(|&x| x * factor).collect();

        let naive = min_time(reps, || {
            let p_current: Vec<f64> = s.p.iter().map(|&x| x * factor).collect();
            let (q, dist) = naive_compute_q(s.manifold.as_ref(), &s.points, s.n, s.ad);
            let loss = kl_loss(&q, &p_current, s.n);
            let g = naive_kl_gradient(
                s.manifold.as_ref(),
                &s.points,
                &q,
                &p_current,
                &dist,
                s.n,
                s.ad,
            );
            std::hint::black_box((loss, g));
        });

        let opt = min_time(reps, || {
            let (q, dist) =
                compute_q_matrix_with_distances(s.manifold.as_ref(), &s.points, s.n, s.ad, 1.0);
            let g = kl_gradient(
                s.manifold.as_ref(),
                &s.points,
                &q,
                std::hint::black_box(&p_early),
                &dist,
                s.n,
                s.ad,
            );
            std::hint::black_box(g);
        });

        println!(
            "step work k={k}: optimized {opt:?} vs naive {naive:?}  ({:.2}x)",
            naive.as_secs_f64() / opt.as_secs_f64()
        );
        assert!(
            opt < naive,
            "k={k}: optimized per-step work ({opt:?}) should beat naive ({naive:?})"
        );
    }
}

#[test]
fn optimized_gradient_is_faster_than_naive() {
    let reps = 12;
    for &k in &[-1.0, 1.0, 0.0] {
        let s = Setup::new(k, 600);
        let opt = min_time(reps, || {
            std::hint::black_box(kl_gradient(
                s.manifold.as_ref(),
                &s.points,
                &s.q,
                &s.p,
                &s.distances,
                s.n,
                s.ad,
            ));
        });
        let naive = min_time(reps, || {
            std::hint::black_box(naive_kl_gradient(
                s.manifold.as_ref(),
                &s.points,
                &s.q,
                &s.p,
                &s.distances,
                s.n,
                s.ad,
            ));
        });
        println!(
            "gradient k={k}: optimized {opt:?} vs naive {naive:?}  ({:.2}x)",
            naive.as_secs_f64() / opt.as_secs_f64()
        );
        assert!(
            opt < naive,
            "k={k}: symmetric gradient ({opt:?}) should beat naive double loop ({naive:?})"
        );
    }
}
