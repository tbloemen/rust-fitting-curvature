//! Shared low-level primitives for the Gromov 4-point δ estimators.
//!
//! Both δ estimators build on these:
//!
//! - [`old_detection`](super::old_detection) — the global 90th-percentile
//!   scalar `gromov_hyperbolicity`.
//! - [`gromov_ball_curve`](super::gromov_ball_curve) — the NeTS-proposal
//!   growing-ball δ(k) curve and saturation test.
//!
//! The four-point δ of a quadruple `{w, x, y, z}` orders the three
//! pair-sums `S₁ ≤ S₂ ≤ S₃`; the value is `S₃ − S₂` (the amount by which
//! the largest sum exceeds the middle — zero for a tree).

/// Gromov 4-point δ of one quadruple: `S_max − S_mid` over the three
/// pair-sums (the NeTS-proposal four-point condition, no ½ factor).
pub fn quad_delta(distances: &[f64], n: usize, a: usize, b: usize, c: usize, d: usize) -> f64 {
    let s1 = distances[a * n + b] + distances[c * n + d];
    let s2 = distances[a * n + c] + distances[b * n + d];
    let s3 = distances[a * n + d] + distances[b * n + c];
    let mut s = [s1, s2, s3];
    s.sort_by(|x, y| x.partial_cmp(y).unwrap());
    s[2] - s[1]
}

/// Median of the `n(n−1)/2` upper-triangle pairwise distances.
pub fn median_pairwise_distance(distances: &[f64], n: usize) -> f64 {
    let mut v = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            v.push(distances[i * n + j]);
        }
    }
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Four distinct indices in `[0, range)`, drawn by rejection from `next`
/// (which yields raw `usize`s; this applies the `% range`).
pub fn four_distinct(range: usize, next: &mut impl FnMut() -> usize) -> [usize; 4] {
    let a = next() % range;
    let mut b = next() % range;
    while b == a {
        b = next() % range;
    }
    let mut c = next() % range;
    while c == a || c == b {
        c = next() % range;
    }
    let mut d = next() % range;
    while d == a || d == b || d == c {
        d = next() % range;
    }
    [a, b, c, d]
}
