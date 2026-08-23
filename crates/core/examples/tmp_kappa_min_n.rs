//! Does the resolvable-κ floor depend on `n`?
//!
//! `tmp_kappa_min` found the residual contrast follows a signal/(signal+noise)
//! law, `C(κ) = κ / (κ + κ_half)`, with `κ_half ≈ 600·σ` at n = 400 — i.e. the
//! curvature signal must be ~100× the distance noise, not ~1× as a naive
//! "distortion must exceed noise" bound would say.
//!
//! If `κ_half` grows with `n`, then κ_min is not a constant and any hardcoded
//! cap is `n`-dependent — which matters for a thesis that sweeps N.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_kappa_min_n`

use fitting_core::curvature_detection::hyperbolic_residual_at;
use fitting_core::synthetic_data::{generate_uniform_hyperbolic, Rng};

const DATA_SEED: u64 = 42;
const NOISE_SEEDS: [u64; 3] = [11, 12, 13];

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}
fn d_rms_of(d: &[f64], n: usize) -> f64 {
    let s: f64 = d.iter().map(|x| x * x).sum();
    (s / (n as f64 * (n as f64 - 1.0))).sqrt()
}

fn noisy(d: &[f64], n: usize, sigma: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut out = d.to_vec();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = (d[i * n + j] * (1.0 + sigma * rng.normal())).max(1e-12);
            out[i * n + j] = v;
            out[j * n + i] = v;
        }
    }
    out
}

/// Contrast C and the implied κ_half = κ(1−C)/C.
fn contrast(d: &[f64], n: usize, rho: f64) -> f64 {
    let r_inf = 1000.0 * d_max_of(d);
    let rf = hyperbolic_residual_at(d, n, 2, r_inf);
    let rt = hyperbolic_residual_at(d, n, 2, rho);
    (rf - rt) / rf
}

fn main() {
    println!("══ κ_half vs n, at fixed extent (κ_true held ~constant) ══");
    println!("  exact H²(ρ=1), extent 0.4;  κ_half = κ(1−C)/C from C = κ/(κ+κ_half)\n");
    println!(
        "  {:>6} {:>9} {:>9}  {:>9} {:>10}  {:>9} {:>10}",
        "n", "d_rms", "κ_true", "C(1e-4)", "κ½(1e-4)", "C(1e-3)", "κ½(1e-3)"
    );
    for &n in &[100usize, 200, 400, 800, 1600] {
        let base = generate_uniform_hyperbolic(n, DATA_SEED, 0.4).distances;
        let dr = d_rms_of(&base, n);
        let k = dr * dr;
        print!("  {n:>6} {dr:>9.4} {k:>9.4}");
        for &s in &[1e-4_f64, 1e-3] {
            let c: f64 = NOISE_SEEDS
                .iter()
                .map(|&sd| contrast(&noisy(&base, n, s, sd), n, 1.0))
                .sum::<f64>()
                / NOISE_SEEDS.len() as f64;
            print!("  {c:>9.4} {:>10.5}", k * (1.0 - c) / c);
        }
        println!();
    }

    println!("\n══ κ_half vs σ, at n = 400 — is it linear in σ? ══");
    let base = generate_uniform_hyperbolic(400, DATA_SEED, 0.4).distances;
    let dr = d_rms_of(&base, 400);
    let k = dr * dr;
    println!("  κ_true = {k:.4}\n");
    println!(
        "  {:>10} {:>10} {:>11} {:>12}",
        "σ", "C", "κ_half", "κ_half / σ"
    );
    for &s in &[3e-6_f64, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3] {
        let c: f64 = NOISE_SEEDS
            .iter()
            .map(|&sd| contrast(&noisy(&base, 400, s, sd), 400, 1.0))
            .sum::<f64>()
            / NOISE_SEEDS.len() as f64;
        let kh = k * (1.0 - c) / c;
        println!("  {s:>10.1e} {c:>10.4} {kh:>11.5} {:>12.1}", kh / s);
    }

    println!("\n══ the mirror problem: noise amplification at SMALL r ══");
    println!("  conjecture: the fit breaks when σ·cosh(d_max/r) ≳ 1, i.e. d_max/r ≳ ln(2/σ)\n");
    println!(
        "  {:>8} {:>8} {:>9} {:>10} {:>12} {:>10}",
        "extent", "d_max", "d_max/ρ", "σ", "σ·cosh(·)", "C"
    );
    for &e in &[1.0_f64, 2.5, 4.0, 6.0, 8.0] {
        let b = generate_uniform_hyperbolic(400, DATA_SEED, e).distances;
        let dm = d_max_of(&b);
        for &s in &[1e-5_f64, 1e-4, 1e-3] {
            let c: f64 = NOISE_SEEDS
                .iter()
                .map(|&sd| contrast(&noisy(&b, 400, s, sd), 400, 1.0))
                .sum::<f64>()
                / NOISE_SEEDS.len() as f64;
            println!(
                "  {e:>8.1} {dm:>8.3} {dm:>9.2} {s:>10.1e} {:>12.3e} {c:>10.4}",
                s * dm.cosh()
            );
        }
    }
}
