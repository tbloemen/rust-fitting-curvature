//! Is the Wilson radius fit *wrong*, or *out of scope*?
//!
//! Motivating puzzle: `wordnet_mammals` is a tree metric, so its true curvature
//! magnitude is unbounded (a tree embeds in H^d with distortion → 1 only as
//! ρ → 0).  Gromov reports κ = ∞ — correct.  Wilson reports κ* = 0.398, pinned
//! at the flat end of its window — maximally wrong, and wrong in the *opposite*
//! direction.
//!
//! Four sections:
//!   A — calibrate the Gromov δ → curvature constant on exact H²(ρ=1).
//!   B — does Wilson recover ρ when the model is correctly specified?
//!   C — the dimension trap: `tree`/`hyperbolic_shells` are exact H⁹ with ρ=1.
//!       Fit them at dim = 2..12 and watch r* move.
//!   D — genuine tree metrics (synthetic b-ary BFS + wordnet_mammals): which way
//!       does the residual actually want to go, over a window 10³× wider than
//!       production's?
//!
//! Run: `cargo run --release -p fitting-core --example tmp_wilson_validity`

use fitting_core::curvature_detection::{detect_hyperbolic, hyperbolic_residual_at};
use fitting_core::data::load_wordnet_mammals;
use fitting_core::synthetic_data::{
    generate_hd_hyperbolic_shells, generate_hd_tree, generate_uniform_hyperbolic, DataPoints,
};

const SEED: u64 = 42;
const N: usize = 400;

fn d_max_of(d: &[f64]) -> f64 {
    d.iter().cloned().fold(0.0_f64, f64::max)
}
fn d_rms_of(d: &[f64], n: usize) -> f64 {
    let s: f64 = d.iter().map(|x| x * x).sum();
    (s / (n as f64 * (n as f64 - 1.0))).sqrt()
}
fn scaled(d: &[f64], s: f64) -> Vec<f64> {
    d.iter().map(|x| x * s).collect()
}

// ── minimiser (mirrors signature::minimise_log_spaced, wider grid) ──────────

fn golden(a: f64, b: f64, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64) {
    let phi = 0.618_033_988_749_894_9_f64;
    let (mut a, mut b) = (a, b);
    let mut r1 = a + (1.0 - phi) * (b - a);
    let mut r2 = a + phi * (b - a);
    let (mut f1, mut f2) = (f(r1), f(r2));
    for _ in 0..60 {
        if f1 < f2 {
            b = r2;
            r2 = r1;
            f2 = f1;
            r1 = a + (1.0 - phi) * (b - a);
            f1 = f(r1);
        } else {
            a = r1;
            r1 = r2;
            f1 = f2;
            r2 = a + phi * (b - a);
            f2 = f(r2);
        }
        if (b - a) / (a + b).max(1e-12) < 1e-9 {
            break;
        }
    }
    if f1 < f2 { (r1, f1) } else { (r2, f2) }
}

/// Returns `(r*, value, where)` with `where` ∈ {-1 lower-pinned, 0 interior,
/// +1 upper-pinned}.
fn fit_window(lo: f64, hi: f64, g: usize, f: &mut dyn FnMut(f64) -> f64) -> (f64, f64, i32) {
    let step = (hi.ln() - lo.ln()) / (g - 1) as f64;
    let gr: Vec<f64> = (0..g).map(|i| (lo.ln() + i as f64 * step).exp()).collect();
    let gv: Vec<f64> = gr.iter().map(|&r| f(r)).collect();
    let mut mins: Vec<usize> = Vec::new();
    if gv[0] < gv[1] {
        mins.push(0);
    }
    for i in 1..g - 1 {
        if gv[i] < gv[i - 1] && gv[i] < gv[i + 1] {
            mins.push(i);
        }
    }
    if gv[g - 1] < gv[g - 2] {
        mins.push(g - 1);
    }
    if mins.is_empty() {
        mins.push(0);
    }
    let (mut br, mut bv, mut bi) = (gr[mins[0]], gv[mins[0]], mins[0]);
    for &i in &mins {
        let a = gr[i.saturating_sub(1)];
        let b = gr[(i + 1).min(g - 1)];
        let (r, v) = if a < b { golden(a, b, f) } else { (gr[i], gv[i]) };
        if v < bv {
            bv = v;
            br = r;
            bi = i;
        }
    }
    let side = if bi == 0 {
        -1
    } else if bi == g - 1 {
        1
    } else {
        0
    };
    (br, bv, side)
}

fn side_str(s: i32) -> &'static str {
    match s {
        -1 => "LOW-pin",
        1 => "HIGH-pin",
        _ => "interior",
    }
}

fn distances_of(d: &DataPoints) -> Vec<f64> {
    d.distances.clone()
}

/// Exact BFS shortest-path metric on a complete `b`-ary tree, truncated to `n`
/// nodes (breadth-first numbering, parent of `i` is `(i-1)/b`).
fn bary_tree_metric(n: usize, b: usize) -> Vec<f64> {
    let depth: Vec<usize> = {
        let mut v = vec![0usize; n];
        for i in 1..n {
            v[i] = v[(i - 1) / b] + 1;
        }
        v
    };
    let parent = |i: usize| if i == 0 { 0 } else { (i - 1) / b };
    let mut d = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (mut a, mut c) = (i, j);
            let mut steps = 0usize;
            while depth[a] > depth[c] {
                a = parent(a);
                steps += 1;
            }
            while depth[c] > depth[a] {
                c = parent(c);
                steps += 1;
            }
            while a != c {
                a = parent(a);
                c = parent(c);
                steps += 2;
            }
            d[i * n + j] = steps as f64;
            d[j * n + i] = steps as f64;
        }
    }
    d
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ln_c = (1.0 + 2.0_f64.sqrt()).ln();

    // ── A ──────────────────────────────────────────────────────────────────
    println!("══ A — Gromov δ constant on EXACT H²(ρ = 1) ══");
    println!("  theory (four-point condition, δ = (S_max − S_mid)/2):  δ_∞ = ln(1+√2) = {ln_c:.4}");
    println!("  quad_delta() in gromov.rs returns S_max − S_mid, i.e. 2δ, with no ½.\n");
    println!(
        "  {:>7} {:>6} {:>9} {:>9} {:>10} {:>10} {:>10}",
        "extent", "n", "d_max", "δ_sat", "ρ̂_code", "ρ̂_half", "ρ̂_half/ρ"
    );
    for &(mr, n) in &[
        (5.0_f64, 200usize),
        (5.0, 400),
        (5.0, 800),
        (8.0, 400),
        (12.0, 400),
        (3.0, 400),
    ] {
        let d = distances_of(&generate_uniform_hyperbolic(n, SEED, mr));
        let v = detect_hyperbolic(&d, n);
        let rho_code = v.saturated_delta / ln_c;
        let rho_half = (v.saturated_delta / 2.0) / ln_c;
        println!(
            "  {mr:>7.1} {n:>6} {:>9.4} {:>9.4} {rho_code:>10.4} {rho_half:>10.4} {rho_half:>10.4}",
            d_max_of(&d),
            v.saturated_delta,
        );
    }
    // scale-invariance: distances × s is exactly H²(ρ = s)
    let d1 = distances_of(&generate_uniform_hyperbolic(N, SEED, 5.0));
    println!("\n  scale check — distances × s is exactly H²(ρ = s):");
    println!(
        "  {:>7} {:>10} {:>12} {:>12}",
        "s (=ρ)", "δ_sat", "ρ̂_code/ρ", "ρ̂_half/ρ"
    );
    for &s in &[0.25_f64, 1.0, 4.0, 20.0] {
        let ds = scaled(&d1, s);
        let v = detect_hyperbolic(&ds, N);
        println!(
            "  {s:>7.2} {:>10.4} {:>12.4} {:>12.4}",
            v.saturated_delta,
            v.saturated_delta / ln_c / s,
            (v.saturated_delta / 2.0) / ln_c / s,
        );
    }

    // ── B ──────────────────────────────────────────────────────────────────
    println!("\n══ B — Wilson on CORRECTLY SPECIFIED data (H², dim = 2 model) ══");
    println!("  window = production's [d_max/20, d_max]\n");
    println!(
        "  {:>7} {:>6} {:>8} {:>10} {:>10} {:>12} {:>10}",
        "extent", "n", "ρ_true", "r*", "r*/ρ", "residual", "where"
    );
    for &(mr, n, s) in &[
        (5.0_f64, 200usize, 1.0_f64),
        (5.0, 400, 1.0),
        (5.0, 800, 1.0),
        (5.0, 400, 7.0),
        (8.0, 400, 1.0),
        (12.0, 400, 1.0),
        (3.0, 400, 1.0),
        (1.5, 400, 1.0),
    ] {
        let d = scaled(&distances_of(&generate_uniform_hyperbolic(n, SEED, mr)), s);
        let dm = d_max_of(&d);
        let mut f = |r: f64| hyperbolic_residual_at(&d, n, 2, r);
        let (r, v, side) = fit_window(dm / 20.0, dm, 30, &mut f);
        println!(
            "  {mr:>7.1} {n:>6} {s:>8.2} {r:>10.4} {:>10.4} {v:>12.3e} {:>10}",
            r / s,
            side_str(side)
        );
    }

    // ── C ──────────────────────────────────────────────────────────────────
    println!("\n══ C — THE DIMENSION TRAP ══");
    println!("  `tree` and `hyperbolic_shells` are exact H⁹ point clouds, K = −1 (ρ = 1),");
    println!("  built by generate_hd_*(n, dim=10, ..) → Poincaré ball of dim 9.");
    println!("  Same *nature* as the control; only the model dimension is wrong.");
    println!("  Window widened to [d_max/1000, 1000·d_max] so nothing pins artificially.\n");
    for (label, d) in [
        ("tree (H⁹, ρ=1)", distances_of(&generate_hd_tree(N, 10, SEED))),
        (
            "hyp_shells (H⁹, ρ=1)",
            distances_of(&generate_hd_hyperbolic_shells(N, 10, SEED)),
        ),
        (
            "control (H², ρ=1)",
            distances_of(&generate_uniform_hyperbolic(N, SEED, 5.0)),
        ),
    ] {
        let dm = d_max_of(&d);
        let dr = d_rms_of(&d, N);
        println!("  ── {label}   d_max = {dm:.3}, d_rms = {dr:.3}");
        println!(
            "     {:>5} {:>11} {:>10} {:>12} {:>12} {:>10}",
            "dim", "r*", "r*/ρ_true", "κ* = (dr/r)²", "residual", "where"
        );
        for dim in [2usize, 3, 4, 5, 6, 7, 8, 9, 10, 11] {
            let mut f = |r: f64| hyperbolic_residual_at(&d, N, dim, r);
            let (r, v, side) = fit_window(dm / 1000.0, dm * 1000.0, 90, &mut f);
            println!(
                "     {dim:>5} {r:>11.4} {r:>10.3} {:>12.4} {v:>12.3e} {:>10}",
                (dr / r).powi(2),
                side_str(side)
            );
        }
        println!();
    }

    // ── D ──────────────────────────────────────────────────────────────────
    println!("══ D — GENUINE TREE METRICS (true κ = ∞, i.e. r* should → 0) ══\n");
    let mut cases: Vec<(String, Vec<f64>, usize)> = vec![
        ("binary tree BFS (b=2)".into(), bary_tree_metric(N, 2), N),
        ("ternary tree BFS (b=3)".into(), bary_tree_metric(N, 3), N),
    ];
    let data_root = std::env::var("DATA_ROOT").unwrap_or_else(|_| "www/public/data".into());
    match load_wordnet_mammals(&format!("{data_root}/wordnet"), N) {
        Ok(dp) => cases.push(("wordnet_mammals".into(), dp.distances.clone(), dp.n_points)),
        Err(e) => println!("  (wordnet_mammals unavailable: {e})\n"),
    }

    for (label, d, n) in &cases {
        let (dm, dr) = (d_max_of(d), d_rms_of(d, *n));
        let v = detect_hyperbolic(d, *n);
        println!(
            "  ── {label}   n = {n}, d_max = {dm:.3}, d_rms = {dr:.3}, δ_sat = {:.4}, slope = {:.3}",
            v.saturated_delta, v.tail_slope
        );
        println!(
            "     {:>5} {:>12} {:>12} {:>12} {:>10}",
            "dim", "r*", "κ* = (dr/r)²", "residual", "where"
        );
        for dim in [2usize, 3, 5, 8, 12, 20] {
            let mut f = |r: f64| hyperbolic_residual_at(d, *n, dim, r);
            let (r, val, side) = fit_window(dm / 1000.0, dm * 1000.0, 90, &mut f);
            println!(
                "     {dim:>5} {r:>12.5} {:>12.4} {val:>12.3e} {:>10}",
                (dr / r).powi(2),
                side_str(side)
            );
        }
        // Which way does the residual slope at production's window?
        let (lo, hi) = (dm / 20.0, dm);
        let (rl, rh) = (
            hyperbolic_residual_at(d, *n, 2, lo),
            hyperbolic_residual_at(d, *n, 2, hi),
        );
        println!(
            "     dim=2 across production window: R({:.3}) = {rl:.4e} → R({:.3}) = {rh:.4e}   [{}]",
            lo,
            hi,
            if rh < rl {
                "descends toward FLAT — wrong way for a tree"
            } else {
                "descends toward CURVED"
            }
        );
        println!();
    }

    Ok(())
}
