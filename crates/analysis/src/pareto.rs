//! Pareto non-domination in the oriented objective space.

use crate::objectives::{oriented_matrix, N_OBJECTIVES};
use crate::records::TrialRecord;

/// Boolean mask of Pareto-non-dominated rows of an oriented matrix *m*.
///
/// Row `i` is dominated when some row `j` is `>=` it in every objective and
/// strictly greater in at least one. Exact duplicates are all kept (no row
/// strictly dominates an identical one).
pub fn pareto_front_mask(m: &[[f64; N_OBJECTIVES]]) -> Vec<bool> {
    let n = m.len();
    let mut keep = vec![true; n];
    for j in 0..n {
        if !keep[j] {
            continue;
        }
        for i in 0..n {
            if i == j || !keep[i] {
                continue;
            }
            if dominates(&m[j], &m[i]) {
                keep[i] = false;
            }
        }
    }
    keep
}

/// Does `a` dominate `b`: weakly better in every objective, strictly in one.
fn dominates(a: &[f64; N_OBJECTIVES], b: &[f64; N_OBJECTIVES]) -> bool {
    let mut strict = false;
    for (x, y) in a.iter().zip(b.iter()) {
        if y > x {
            return false;
        }
        if y < x {
            strict = true;
        }
    }
    strict
}

/// The non-dominated subset of *records* (in the 10-objective space).
pub fn pareto_front_records(records: &[TrialRecord]) -> Vec<TrialRecord> {
    let m = oriented_matrix(records);
    let keep = pareto_front_mask(&m);
    records
        .iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(r, _)| r.clone())
        .collect()
}

/// Indices of the 2D Pareto front over `(x, y)`, sorted by x ascending.
///
/// `x_up` / `y_up` say whether larger is better on each axis. Used for the
/// per-cell front cross-sections in the Exp 2 figure.
pub fn slice_front_2d(x: &[f64], y: &[f64], x_up: bool, y_up: bool) -> Vec<usize> {
    let n = x.len().min(y.len());
    let sign = |up: bool, v: f64| if up { v } else { -v };
    let xs: Vec<f64> = x[..n].iter().map(|&v| sign(x_up, v)).collect();
    let ys: Vec<f64> = y[..n].iter().map(|&v| sign(y_up, v)).collect();

    let mut keep = vec![true; n];
    for j in 0..n {
        if !keep[j] {
            continue;
        }
        for i in 0..n {
            if i == j || !keep[i] {
                continue;
            }
            // j dominates i: weakly better in both, strictly better in one.
            if xs[j] >= xs[i] && ys[j] >= ys[i] && (xs[j] > xs[i] || ys[j] > ys[i]) {
                keep[i] = false;
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// The front's step (attainment) polyline, for drawing it as the staircase it
/// is rather than as straight segments between its points.
///
/// Input is a front sorted by x ascending. Demanding any x beyond a point costs
/// the *next* point's y immediately, so each segment is a riser at the earlier
/// x followed by a tread at the later y:
/// `[(x0,y0), (x1,y1)]` → `[(x0,y0), (x0,y1), (x1,y1)]`. Empty and single-point
/// fronts pass through unchanged.
pub fn step_polyline(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(points.len().saturating_mul(2).saturating_sub(1));
    for (i, &(x, y)) in points.iter().enumerate() {
        if i > 0 {
            out.push((points[i - 1].0, y));
        }
        out.push((x, y));
    }
    out
}
