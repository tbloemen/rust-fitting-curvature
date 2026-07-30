//! Seeded Monte-Carlo hypervolume in the oriented unit box `[0, 1]^10`.

use fitting_core::rng::Rng;
use serde::{Deserialize, Serialize};

use crate::objectives::{oriented_matrix, N_OBJECTIVES};
use crate::pareto::pareto_front_mask;
use crate::records::TrialRecord;

/// Hypervolume summary for one cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HvSummary {
    pub n_trials: usize,
    pub n_front: usize,
    pub hv: f64,
    pub hv_se: f64,
}

/// Seeded Monte-Carlo hypervolume of a point set in `[0, 1]^10`.
///
/// The reference point is the origin and the reference box has volume 1, so the
/// hypervolume dominated by the point set equals the fraction of uniform samples
/// that the set dominates. Dominated points contribute nothing, so passing the
/// Pareto front (rather than all trials) gives an identical estimate faster.
///
/// Returns `(hv, se)` where `se` is the Monte-Carlo standard error of the
/// estimate, `sqrt(p (1 - p) / n_mc)` from the dominated-fraction binomial. This
/// is *estimation* error of the integral, not run-to-run optimizer variance —
/// there is a single optimization run per cell, so the latter is not estimated
/// here.
///
/// Two exact accelerations over the numpy original, neither of which changes the
/// estimator: samples outside the front's bounding box are rejected on the first
/// coordinate that exceeds the box (a sample can only be dominated inside it),
/// and the scan over front points stops at the first dominator. Front points are
/// visited in descending coordinate-sum order so the likeliest dominator is
/// tried first.
pub fn monte_carlo_hypervolume(front: &[[f64; N_OBJECTIVES]], n_mc: u64, seed: u64) -> (f64, f64) {
    if front.is_empty() || n_mc == 0 {
        return (0.0, 0.0);
    }

    // Per-dimension maximum: the bounding box of the dominated region.
    let mut box_max = [0.0f64; N_OBJECTIVES];
    for f in front {
        for (m, v) in box_max.iter_mut().zip(f.iter()) {
            if *v > *m {
                *m = *v;
            }
        }
    }

    let mut order: Vec<usize> = (0..front.len()).collect();
    order.sort_by(|&a, &b| {
        let sa: f64 = front[a].iter().sum();
        let sb: f64 = front[b].iter().sum();
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted: Vec<[f64; N_OBJECTIVES]> = order.into_iter().map(|i| front[i]).collect();

    let mut rng = Rng::new(seed);
    let mut dominated: u64 = 0;
    for _ in 0..n_mc {
        let mut u = [0.0f64; N_OBJECTIVES];
        let mut outside = false;
        for (slot, m) in u.iter_mut().zip(box_max.iter()) {
            *slot = rng.uniform();
            if *slot > *m {
                outside = true;
                // Keep drawing so the RNG stream advances by exactly
                // N_OBJECTIVES per sample regardless of where the sample lands —
                // the estimate must not depend on the rejection order.
            }
        }
        if outside {
            continue;
        }
        // u is dominated iff some front point f has f >= u in every dimension.
        if sorted
            .iter()
            .any(|f| f.iter().zip(u.iter()).all(|(a, b)| b <= a))
        {
            dominated += 1;
        }
    }

    let p = dominated as f64 / n_mc as f64;
    let se = ((p * (1.0 - p)).max(0.0) / n_mc as f64).sqrt();
    (p, se)
}

/// Hypervolume summary for one cell's trial records.
///
/// Reduces to the Pareto front first (HV-invariant, much faster), then runs the
/// Monte-Carlo estimate.
pub fn cell_hypervolume(records: &[TrialRecord], n_mc: u64, seed: u64) -> HvSummary {
    let m_all = oriented_matrix(records);
    let keep = pareto_front_mask(&m_all);
    let front: Vec<[f64; N_OBJECTIVES]> = m_all
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(row, _)| *row)
        .collect();
    let (hv, hv_se) = monte_carlo_hypervolume(&front, n_mc, seed);
    HvSummary {
        n_trials: m_all.len(),
        n_front: front.len(),
        hv,
        hv_se,
    }
}

/// Deterministic per-cell RNG seed, independent of shard/worker layout.
///
/// Mirrors the Python `crc32(stem) ^ seed` so a cell's estimate depends only on
/// its name and the base seed — sharding never changes a result, it only changes
/// which task computes it.
pub fn cell_seed(stem: &str, base_seed: u64) -> u64 {
    (crc32(stem.as_bytes()) ^ (base_seed as u32)) as u64
}

/// CRC-32 (IEEE), matching Python's `zlib.crc32`.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}
