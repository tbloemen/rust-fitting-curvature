//! Does the Wilson objective's *units* hide a good small-r fit?
//!
//! `Z(r) = −r² cosh(D/r)` has entries growing like `r² e^{d_max/r}` as `r → 0`,
//! so the raw residual `Σ|λ_res|` diverges there **no matter how good the fit
//! is in relative terms**.  A tree metric embeds in H^d with distortion → 1 as
//! `r → 0` (Sarkar), so if the objective were scale-free the tree's residual
//! should *fall* toward small `r`.  This compares
//!
//!   raw       R(r)                        — what fit_hyperbolic minimises
//!   gauged    R(r) / (n · r² cosh(d_max/r))  — R as a fraction of Z's own scale
//!
//! Run: `cargo run --release -p fitting-core --example tmp_tree_relative`

use fitting_core::curvature_detection::hyperbolic_residual_at;
use fitting_core::data::load_wordnet_mammals;
use fitting_core::synthetic_data::generate_uniform_hyperbolic;

const N: usize = 400;
const SEED: u64 = 42;

fn bary_tree_metric(n: usize, b: usize) -> Vec<f64> {
    let mut depth = vec![0usize; n];
    for i in 1..n {
        depth[i] = depth[(i - 1) / b] + 1;
    }
    let parent = |i: usize| if i == 0 { 0 } else { (i - 1) / b };
    let mut d = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (mut a, mut c, mut s) = (i, j, 0usize);
            while depth[a] > depth[c] {
                a = parent(a);
                s += 1;
            }
            while depth[c] > depth[a] {
                c = parent(c);
                s += 1;
            }
            while a != c {
                a = parent(a);
                c = parent(c);
                s += 2;
            }
            d[i * n + j] = s as f64;
            d[j * n + i] = s as f64;
        }
    }
    d
}

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}

fn main() {
    let data_root = std::env::var("DATA_ROOT").unwrap_or_else(|_| "www/public/data".into());
    let mut cases: Vec<(String, Vec<f64>, usize)> = vec![
        ("binary tree BFS".into(), bary_tree_metric(N, 2), N),
        (
            "H² control (ρ = 1)".into(),
            generate_uniform_hyperbolic(N, SEED, 5.0).distances,
            N,
        ),
    ];
    if let Ok(dp) = load_wordnet_mammals(&format!("{data_root}/wordnet"), N) {
        cases.insert(1, ("wordnet_mammals".into(), dp.distances, dp.n_points));
    }

    for (name, d, n) in &cases {
        let dm = d_max_of(d);
        println!("══ {name}   (dim = 2 model, d_max = {dm:.3}) ══");
        println!(
            "  {:>10} {:>10} {:>13} {:>13}   {}",
            "r", "r/d_max", "raw R(r)", "gauged", "gauged = R / (n r² cosh(d_max/r))"
        );
        for &frac in &[
            0.02_f64, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5, 0.7, 1.0, 2.0, 5.0, 20.0,
        ] {
            let r = dm * frac;
            let raw = hyperbolic_residual_at(d, *n, 2, r);
            let scale = *n as f64 * r * r * (dm / r).cosh();
            println!(
                "  {r:>10.4} {frac:>10.3} {raw:>13.4e} {:>13.4e}",
                raw / scale
            );
        }
        println!();
    }
}
