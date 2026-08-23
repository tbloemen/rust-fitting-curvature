//! Why the non-divergent eigenvalues of `Z(r)` converge to the *compression*
//! `S = (−½A)|_{1⊥}` and not to `−½A` itself.
//!
//! Isolates the linear algebra from the kernel Taylor expansion by scanning the
//! exact model matrix `M(t) = tJ + B`, `B = −½A`, as `t → ∞`.  Checks three
//! claims:
//!
//! 1. `spec(M(t))` splits into one divergent eigenvalue `≈ tn + uᵀBu` and `n−1`
//!    that converge to `spec(S)`, `u = 1/√n`.
//! 2. The convergence is `O(1/μ)`, `μ = tn` — so `O(1/r²)` for `t = r²`.
//! 3. The total shift of the small block is *exactly* `−‖b‖²/μ` at leading
//!    order, where `b = VᵀBu` is the coupling between `span(1)` and `1⊥`.
//!    Equivalently `‖b‖² = ‖P·A1‖²/(4n)`.
//!
//! And it shows `spec(−½A)` is nowhere near `spec(S)` — they differ by an
//! extensive `O(n·d_max²)` outlier, so the projection is not a technicality.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_rank1_splitting`

use fitting_core::curvature_detection::eigenvalues_symmetric;
use fitting_core::synthetic_data::{
    generate_uniform_ball_2d, generate_uniform_hyperbolic, generate_uniform_sphere,
};

const N: usize = 200;
const SEED: u64 = 42;
const E_RADIUS: f64 = 5.0;
const H_MAX_RHO: f64 = 5.0;

/// `G = −½ P A P` by double centring, `A = D∘D`.
fn mds_gram(a: &[f64], n: usize) -> Vec<f64> {
    let row_mean: Vec<f64> = (0..n)
        .map(|i| a[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64)
        .collect();
    let grand = row_mean.iter().sum::<f64>() / n as f64;
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = -0.5 * (a[i * n + j] - row_mean[i] - row_mean[j] + grand);
        }
    }
    g
}

/// `spec(G) = {0} ∪ spec(S)` because `G1 = 0`.  Drop the eigenvalue closest to
/// zero to recover `spec(S)` (`n−1` values, ascending).
fn drop_one_zero(spec: &[f64]) -> Vec<f64> {
    let k = (0..spec.len())
        .min_by(|&i, &j| spec[i].abs().partial_cmp(&spec[j].abs()).unwrap())
        .unwrap();
    let mut out: Vec<f64> = spec.to_vec();
    out.remove(k);
    out
}

fn main() {
    let cases: [(&str, Vec<f64>); 3] = [
        ("E²", generate_uniform_ball_2d(N, SEED, E_RADIUS).distances),
        ("S²", generate_uniform_sphere(N, SEED).distances),
        (
            "H²",
            generate_uniform_hyperbolic(N, SEED, H_MAX_RHO).distances,
        ),
    ];

    for (name, d) in &cases {
        let n = N;
        let d_max = d.iter().cloned().fold(0.0_f64, f64::max);
        let a: Vec<f64> = d.iter().map(|x| x * x).collect();
        let b: Vec<f64> = a.iter().map(|x| -0.5 * x).collect(); // B = −½A

        let spec_b = eigenvalues_symmetric(&b, n);
        let spec_g = eigenvalues_symmetric(&mds_gram(&a, n), n);
        let spec_s = drop_one_zero(&spec_g);
        let tr_s: f64 = spec_s.iter().sum();

        // uᵀBu with u = 1/√n, and the coupling ‖b‖² = ‖P·A1‖² / (4n).
        let row_sums: Vec<f64> = (0..n).map(|i| a[i * n..(i + 1) * n].iter().sum()).collect();
        let mean_rs = row_sums.iter().sum::<f64>() / n as f64;
        let coupling2: f64 = row_sums.iter().map(|x| (x - mean_rs).powi(2)).sum::<f64>()
            / (4.0 * n as f64);
        let ubu = -0.5 * row_sums.iter().sum::<f64>() / n as f64;

        println!("\n══ {name}  (n={n}, d_max={d_max:.4}) ══");
        println!(
            "  spec(−½A):  min {:>12.4e}   max {:>12.4e}",
            spec_b[0],
            spec_b[n - 1]
        );
        println!(
            "  spec(S)  :  min {:>12.4e}   max {:>12.4e}     ← the actual limit",
            spec_s[0],
            spec_s[n - 2]
        );
        println!(
            "  gap between them: |λ_min(−½A) − λ_min(S)| = {:.4e}   (n·d_max² = {:.4e})",
            (spec_b[0] - spec_s[0]).abs(),
            n as f64 * d_max * d_max
        );
        println!("  uᵀBu = {ubu:.6e}   ‖b‖² = {coupling2:.6e}");
        println!();
        println!(
            "  {:>9}  {:>12}  {:>12}  {:>13}  {:>12}",
            "t/d_max²", "max|Δλ|", "max|Δλ|·μ", "(ΣΔλ)·μ", "−‖b‖²"
        );

        for k in 0..6 {
            let t = d_max * d_max * 10.0_f64.powi(k);
            let mu = t * n as f64;
            let mut m = b.clone();
            for x in m.iter_mut() {
                *x += t; // M = tJ + B
            }
            let spec_m = eigenvalues_symmetric(&m, n);
            // Ascending, so the divergent (+μ) eigenvalue is the last one.
            let small = &spec_m[..n - 1];
            let err = small
                .iter()
                .zip(&spec_s)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max);
            let shift = small.iter().sum::<f64>() - tr_s;
            println!(
                "  {:>9.0e}  {:>12.4e}  {:>12.4e}  {:>13.6e}  {:>12.6e}",
                10.0_f64.powi(k),
                err,
                err * mu,
                shift * mu,
                -coupling2
            );
        }

        // The divergent eigenvalue itself: λ_big ≈ μ + uᵀBu.
        let t = d_max * d_max * 1e4;
        let mu = t * n as f64;
        let mut m = b.clone();
        for x in m.iter_mut() {
            *x += t;
        }
        let spec_m = eigenvalues_symmetric(&m, n);
        println!(
            "  divergent λ at t/d_max²=1e4: {:.8e}   μ + uᵀBu = {:.8e}",
            spec_m[n - 1],
            mu + ubu
        );
    }
}
