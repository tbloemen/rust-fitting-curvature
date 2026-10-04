//! Pareto non-domination in the oriented objective space.

use crate::objectives::{oriented_matrix, ObjectiveSpace, Row};
use crate::records::TrialRecord;

/// Boolean mask of Pareto-non-dominated rows of an oriented matrix *m*.
///
/// Row `i` is dominated when some row `j` is `>=` it in every objective and
/// strictly greater in at least one. Exact duplicates are all kept (no row
/// strictly dominates an identical one).
#[must_use]
pub fn pareto_front_mask(m: &[Row]) -> Vec<bool> {
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
///
/// Rows shorter than each other compare only over the overlap, which cannot
/// happen within one [`ObjectiveSpace`] and is why every caller builds both
/// sides from the same one.
fn dominates(a: &[f64], b: &[f64]) -> bool {
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

/// The non-dominated subset of *records*, scored in *space*.
#[must_use]
pub fn pareto_front_records(records: &[TrialRecord], space: ObjectiveSpace) -> Vec<TrialRecord> {
    let m = oriented_matrix(records, space);
    let keep = pareto_front_mask(&m);
    records
        .iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(r, _)| r.clone())
        .collect()
}
