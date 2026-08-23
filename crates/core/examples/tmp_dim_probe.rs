//! Targeted probe: does the `tree` fixture really have no residual zero at
//! ρ = 1, or did the coarse log grid straddle a cusp (< 1% wide in r)?
//!
//! Evaluates the hyperbolic residual on a *fine* grid around r = 1 for the
//! H⁹ fixtures at their true dimension, and reports the effective rank of the
//! Wilson Gram matrix at r = 1 so the dimension the data actually needs is
//! measured rather than assumed.
//!
//! Run: `cargo run --release -p fitting-core --example tmp_dim_probe`

use fitting_core::curvature_detection::hyperbolic_residual_at;
use fitting_core::synthetic_data::{
    generate_hd_hyperbolic_shells, generate_hd_tree, generate_uniform_hyperbolic,
};

const N: usize = 400;
const SEED: u64 = 42;

fn main() {
    let cases: [(&str, Vec<f64>); 3] = [
        ("tree (H⁹ + 0.05 noise)", generate_hd_tree(N, 10, SEED).distances),
        (
            "hyp_shells (H⁹)",
            generate_hd_hyperbolic_shells(N, 10, SEED).distances,
        ),
        (
            "control (H²)",
            generate_uniform_hyperbolic(N, SEED, 5.0).distances,
        ),
    ];

    println!("══ residual at the TRUE radius r = 1, by model dimension ══");
    println!("  (if the fixture is an exact H^d cloud, this hits ~1e-6 at d = its dimension)\n");
    println!("  {:<26} {:>10} {:>12}", "fixture", "dim", "R(r = 1)");
    for (name, d) in &cases {
        for dim in [2usize, 4, 6, 8, 9, 10, 12, 16, 24, 40] {
            println!(
                "  {:<26} {dim:>10} {:>12.4e}",
                if dim == 2 { *name } else { "" },
                hyperbolic_residual_at(d, N, dim, 1.0)
            );
        }
        println!();
    }

    println!("══ fine scan of r near 1 (dim = 9), to rule out a straddled cusp ══");
    println!("  {:>9} {:>14} {:>14} {:>14}", "r", "tree", "hyp_shells", "control");
    for i in 0..=40 {
        let r = 0.5 * (2.0_f64).powf(i as f64 / 20.0); // 0.5 → 2.0, 0.17% steps near 1
        println!(
            "  {r:>9.5} {:>14.4e} {:>14.4e} {:>14.4e}",
            hyperbolic_residual_at(&cases[0].1, N, 9, r),
            hyperbolic_residual_at(&cases[1].1, N, 9, r),
            hyperbolic_residual_at(&cases[2].1, N, 9, r),
        );
    }
}
