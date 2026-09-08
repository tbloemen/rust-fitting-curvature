//! Tests for embedding quality metrics.
//! Ported from Python test/test_metrics.py

use fitting_core::metrics::*;
use fitting_core::synthetic_data::Rng;

// ---------------------------------------------------------------------------
// Helpers shared by multiple tests
// ---------------------------------------------------------------------------

/// Build 2D point clusters: `n_clusters` groups of `per_cluster` points,
/// placed on a wide grid so clusters are well-separated.
fn make_clustered_2d(
    n_clusters: usize,
    per_cluster: usize,
    spread: f64,
    seed: u64,
) -> (Vec<f64>, Vec<u32>) {
    let mut rng = Rng::new(seed);
    let n = n_clusters * per_cluster;
    let mut pts = vec![0.0f64; n * 2];
    let mut labels = vec![0u32; n];
    for c in 0..n_clusters {
        let cx = (c as f64) * 20.0;
        for i in 0..per_cluster {
            let idx = c * per_cluster + i;
            pts[idx * 2] = cx + rng.normal() * spread;
            pts[idx * 2 + 1] = rng.normal() * spread;
            labels[idx] = c as u32;
        }
    }
    (pts, labels)
}

fn make_distance_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let val = rng.uniform() * 5.0 + 0.1;
            d[i * n + j] = val;
            d[j * n + i] = val;
        }
    }
    d
}

#[test]
fn test_cluster_density_measure_separated() {
    // Two well-separated clusters should have high ClDM
    let n = 100;
    let mut pts = vec![0.0; n * 2];
    let mut labels = vec![0u32; n];

    // Cluster 0 around (-5, 0), Cluster 1 around (5, 0)
    let mut rng = Rng::new(42);
    for i in 0..n / 2 {
        pts[i * 2] = -5.0 + rng.normal() * 0.5;
        pts[i * 2 + 1] = rng.normal() * 0.5;
        labels[i] = 0;
    }
    for i in n / 2..n {
        pts[i * 2] = 5.0 + rng.normal() * 0.5;
        pts[i * 2 + 1] = rng.normal() * 0.5;
        labels[i] = 1;
    }

    let cldm = cluster_density_measure(&pts, &labels, n);
    assert!(
        cldm > 1.0,
        "Separated clusters should have high ClDM, got {cldm}"
    );
}

#[test]
fn test_cluster_density_measure_overlapping() {
    // Two overlapping clusters should have lower ClDM
    let n = 100;
    let mut rng = Rng::new(42);
    let pts: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();
    let labels: Vec<u32> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

    let cldm = cluster_density_measure(&pts, &labels, n);
    // Not a strict bound, but should be relatively low
    assert!(cldm >= 0.0);
}

#[test]
fn test_dunn_index_well_separated() {
    let n = 60;
    let mut d = vec![0.0; n * n];
    let mut labels = vec![0u32; n];

    // 3 clusters of 20 points each
    // Within cluster: small distances (0.1-0.5)
    // Between clusters: large distances (5-10)
    let mut rng = Rng::new(42);
    for c in 0..3 {
        for i in 0..20 {
            labels[c * 20 + i] = c as u32;
            for j in (i + 1)..20 {
                let dist = 0.1 + rng.uniform() * 0.4;
                d[(c * 20 + i) * n + (c * 20 + j)] = dist;
                d[(c * 20 + j) * n + (c * 20 + i)] = dist;
            }
        }
    }
    // Between-cluster distances
    for ci in 0..3 {
        for cj in (ci + 1)..3 {
            for i in 0..20 {
                for j in 0..20 {
                    let dist = 5.0 + rng.uniform() * 5.0;
                    d[(ci * 20 + i) * n + (cj * 20 + j)] = dist;
                    d[(cj * 20 + j) * n + (ci * 20 + i)] = dist;
                }
            }
        }
    }

    let di = dunn_index(&d, &labels, n);
    assert!(
        di > 1.0,
        "Well-separated clusters should have Dunn > 1, got {di}"
    );
}

// ---------------------------------------------------------------------------
// Normalized stress
// ---------------------------------------------------------------------------

#[test]
fn test_normalized_stress_zero_for_identical() {
    let d = make_distance_matrix(20, 42);
    let stress = normalized_stress(&d, &d, 20);
    assert!(
        stress.abs() < 1e-10,
        "Identical matrices should give 0 stress, got {stress}"
    );
}

#[test]
fn test_normalized_stress_positive_for_different() {
    let d1 = make_distance_matrix(20, 42);
    let d2 = make_distance_matrix(20, 99);
    let stress = normalized_stress(&d1, &d2, 20);
    assert!(
        stress > 0.0,
        "Different matrices should give positive stress"
    );
}

#[test]
fn test_normalized_stress_range() {
    let d1 = make_distance_matrix(30, 1);
    let d2 = make_distance_matrix(30, 2);
    let stress = normalized_stress(&d1, &d2, 30);
    assert!(
        (0.0..=1.0).contains(&stress),
        "Stress out of [0,1]: {stress}"
    );
}

// ---------------------------------------------------------------------------
// Neighborhood hit
// ---------------------------------------------------------------------------

#[test]
fn test_neighborhood_hit_perfect_separation() {
    let (pts, labels) = make_clustered_2d(3, 20, 0.3, 42);
    let n = pts.len() / 2;
    let d = euclidean_dist_2d(&pts, n);
    let nh = neighborhood_hit(&d, &labels, n, 7);
    assert!(
        nh > 0.95,
        "Well-separated clusters should have NH > 0.95, got {nh}"
    );
}

#[test]
fn test_neighborhood_hit_random_labels() {
    // Random labels on random points: NH should be around 1/num_classes
    let mut rng = Rng::new(42);
    let n = 120;
    let num_classes = 3usize;
    let pts: Vec<f64> = (0..n * 2).map(|_| rng.normal()).collect();
    let labels: Vec<u32> = (0..n).map(|i| (i % num_classes) as u32).collect();
    let d = euclidean_dist_2d(&pts, n);
    let nh = neighborhood_hit(&d, &labels, n, 7);
    // With random points + balanced labels, expected NH ≈ 1/3
    assert!(nh < 0.6, "Random labels should give low NH, got {nh}");
}

#[test]
fn test_neighborhood_hit_range() {
    let (pts, labels) = make_clustered_2d(2, 15, 1.0, 7);
    let n = pts.len() / 2;
    let d = euclidean_dist_2d(&pts, n);
    let nh = neighborhood_hit(&d, &labels, n, 5);
    assert!(
        (0.0..=1.0).contains(&nh),
        "Neighborhood hit out of [0,1]: {nh}"
    );
}

// ---------------------------------------------------------------------------
// Shepard goodness
// ---------------------------------------------------------------------------

#[test]
fn test_shepard_goodness_perfect() {
    let d = make_distance_matrix(25, 42);
    let sg = shepard_goodness(&d, &d, 25);
    assert!(
        (sg - 1.0).abs() < 1e-10,
        "Identical matrices should give shepard_goodness = 1, got {sg}"
    );
}

#[test]
fn test_shepard_goodness_range() {
    let d1 = make_distance_matrix(25, 42);
    let d2 = make_distance_matrix(25, 99);
    let sg = shepard_goodness(&d1, &d2, 25);
    assert!(
        (0.0..=1.0).contains(&sg),
        "Shepard goodness out of [0,1]: {sg}"
    );
}

#[test]
fn test_shepard_goodness_lower_for_uncorrelated() {
    // Reversed rank order should give poor Shepard goodness
    let n = 20;
    // Make d1 increasing, d2 decreasing (perfectly anti-correlated ranks)
    let mut d1 = vec![0.0f64; n * n];
    let mut d2 = vec![0.0f64; n * n];
    let mut val = 1.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            d1[i * n + j] = val;
            d1[j * n + i] = val;
            d2[i * n + j] = 1000.0 - val;
            d2[j * n + i] = 1000.0 - val;
            val += 1.0;
        }
    }
    let sg = shepard_goodness(&d1, &d2, n);
    // Anti-correlated ranks → raw r_s = -1, which (r_s + 1) / 2 maps to 0, the
    // bottom of the normalised scale.
    assert!(
        sg < 1e-10,
        "Anti-correlated distances should give shepard goodness 0, got {sg}"
    );
}

#[test]
fn test_shepard_goodness_collapsed_embedding_is_no_skill() {
    // Every pairwise distance identical: the embedding carries no rank
    // information whatsoever, so the score must be the no-skill value of the
    // normalised scale, 0.5 — the image of r_s = 0 under (r_s + 1) / 2, and
    // strictly worse than any embedding with real rank agreement.
    let n = 30;
    let d_high = make_distance_matrix(n, 7);
    let collapsed = vec![0.25f64; n * n];
    let sg = shepard_goodness(&d_high, &collapsed, n);
    assert!(
        (sg - 0.5).abs() < 1e-12,
        "Collapsed embedding should score 0.5, got {sg}"
    );
    // ...and it must not beat a genuinely correlated embedding.
    assert!(
        sg < shepard_goodness(&d_high, &d_high, n),
        "Collapsed embedding scored no worse than a perfect one"
    );
}

#[test]
fn test_shepard_goodness_ties_match_reference_spearman() {
    // Heavily tied integer distances on both sides (the wordnet_mammals shape:
    // BFS hop counts take a handful of distinct values). Compare against
    // r_s computed independently as Pearson on fractional ranks.
    let n = 24;
    let mut d1 = vec![0.0f64; n * n];
    let mut d2 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let a = ((i + j) % 4) as f64 + 1.0;
            let b = ((i * j) % 3) as f64 + 1.0;
            d1[i * n + j] = a;
            d1[j * n + i] = a;
            d2[i * n + j] = b;
            d2[j * n + i] = b;
        }
    }
    let sg = shepard_goodness(&d1, &d2, n);

    let mut u1 = Vec::new();
    let mut u2 = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            u1.push(d1[i * n + j]);
            u2.push(d2[i * n + j]);
        }
    }
    let expected = (pearson_on_fractional_ranks(&u1, &u2) + 1.0) / 2.0;
    assert!(
        (sg - expected).abs() < 1e-10,
        "Tied inputs should give r_s = {expected}, got {sg}"
    );
}

#[test]
fn test_shepard_goodness_tie_order_is_irrelevant() {
    // Permuting points must not change the score. Under ordinal ranking it
    // does, because ties are then resolved by array position.
    let n = 20;
    let mut d1 = vec![0.0f64; n * n];
    let mut d2 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let a = ((i + j) % 3) as f64;
            let b = ((i + 2 * j) % 5) as f64;
            d1[i * n + j] = a;
            d1[j * n + i] = a;
            d2[i * n + j] = b;
            d2[j * n + i] = b;
        }
    }
    let sg = shepard_goodness(&d1, &d2, n);

    // Reverse the point order in both matrices: same distance multiset, same
    // pairing, different array layout.
    let perm: Vec<usize> = (0..n).rev().collect();
    let mut p1 = vec![0.0f64; n * n];
    let mut p2 = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            p1[i * n + j] = d1[perm[i] * n + perm[j]];
            p2[i * n + j] = d2[perm[i] * n + perm[j]];
        }
    }
    let sg_perm = shepard_goodness(&p1, &p2, n);
    assert!(
        (sg - sg_perm).abs() < 1e-12,
        "Score changed under point relabelling: {sg} vs {sg_perm}"
    );
}

/// Reference `r_s`: the Pearson correlation coefficient of the rank variables
/// `R[X]`, `R[Y]`, with ties given fractional (average) ranks.
fn pearson_on_fractional_ranks(a: &[f64], b: &[f64]) -> f64 {
    fn fractional_ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&x, &y| v[x].total_cmp(&v[y]));
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0;
            for &k in &idx[i..=j] {
                r[k] = avg;
            }
            i = j + 1;
        }
        r
    }
    let (ra, rb) = (fractional_ranks(a), fractional_ranks(b));
    let m = ra.len() as f64;
    let ma = ra.iter().sum::<f64>() / m;
    let mb = rb.iter().sum::<f64>() / m;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in ra.iter().zip(rb.iter()) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    cov / (va * vb).sqrt()
}

// ---------------------------------------------------------------------------
// Before vs after projection distinction
// ---------------------------------------------------------------------------

#[test]
fn test_euclidean_dist_2d_is_symmetric() {
    let (pts, _) = make_clustered_2d(2, 10, 1.0, 42);
    let n = pts.len() / 2;
    let d = euclidean_dist_2d(&pts, n);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (d[i * n + j] - d[j * n + i]).abs() < 1e-12,
                "Distance matrix not symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_before_after_projection_differ() {
    // Simulate before/after: use two different distance matrices for the
    // same set of points (analogous to manifold geodesic vs. projected 2D).
    // Metrics computed on them should generally differ, demonstrating that
    // both variants are worth storing.
    let (pts, labels) = make_clustered_2d(3, 15, 0.5, 42);
    let n = pts.len() / 2;

    // "before": distances in some ambient space (here: a scaled version)
    let d_before = euclidean_dist_2d(&pts, n);
    // "after": distances in a distorted 2D space (simulate projection distortion)
    let pts_distorted: Vec<f64> = pts
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v * 2.0 } else { v * 0.5 })
        .collect();
    let d_after = euclidean_dist_2d(&pts_distorted, n);

    let t_before = trustworthiness(&d_before, &d_before, n, 7);
    let _t_after = trustworthiness(&d_before, &d_after, n, 7);
    let nh_before = neighborhood_hit(&d_before, &labels, n, 7);
    let nh_after = neighborhood_hit(&d_after, &labels, n, 7);
    let sg_before = shepard_goodness(&d_before, &d_before, n);
    let sg_after = shepard_goodness(&d_before, &d_after, n);

    // Before (self-comparison) should be perfect / better
    assert!(
        (t_before - 1.0).abs() < 1e-10,
        "Trustworthiness before should be 1.0, got {t_before}"
    );
    assert!(
        sg_before > sg_after,
        "Shepard goodness before ({sg_before}) should exceed after ({sg_after})"
    );
    // NH may be high in both cases since clusters are well-separated
    let _ = nh_before;
    let _ = nh_after;
}

#[test]
fn test_davies_bouldin_separated() {
    // Create well-separated clusters with a distance matrix
    let n = 40;
    let mut d = vec![0.0; n * n];
    let mut labels = vec![0u32; n];
    let mut rng = Rng::new(42);

    for c in 0..2 {
        for i in 0..20 {
            labels[c * 20 + i] = c as u32;
            for j in (i + 1)..20 {
                let dist = 0.1 + rng.uniform() * 0.3;
                d[(c * 20 + i) * n + (c * 20 + j)] = dist;
                d[(c * 20 + j) * n + (c * 20 + i)] = dist;
            }
        }
    }
    for i in 0..20 {
        for j in 0..20 {
            let dist = 8.0 + rng.uniform() * 2.0;
            d[i * n + (20 + j)] = dist;
            d[(20 + j) * n + i] = dist;
        }
    }

    let db = davies_bouldin(&d, &labels, n);
    assert!(db > 0.0, "DB should be positive");
    assert!(
        db < 1.0,
        "Well-separated clusters should have low DB, got {db}"
    );
}

// ---------------------------------------------------------------------------
// compute_snapshot
// ---------------------------------------------------------------------------

#[test]
fn test_compute_snapshot_without_labels_gives_none() {
    let (pts_2d, _) = make_clustered_2d(2, 12, 1.0, 42);
    let n = pts_2d.len() / 2;
    let d = make_distance_matrix(n, 42);
    let snap = compute_snapshot(&d, &d, &pts_2d, None, n, 5);
    assert!(snap.neighborhood_hit_manifold.is_none());
    assert!(snap.neighborhood_hit_2d.is_none());
    assert!(snap.cluster_density_measure.is_none());
    assert!(snap.davies_bouldin_ratio.is_none());
}

#[test]
fn test_compute_snapshot_with_labels_gives_some() {
    let (pts_2d, labels) = make_clustered_2d(3, 10, 0.5, 42);
    let n = pts_2d.len() / 2;
    let d = euclidean_dist_2d(&pts_2d, n);
    let snap = compute_snapshot(&d, &d, &pts_2d, Some(&labels), n, 7);
    assert!(snap.neighborhood_hit_manifold.is_some());
    assert!(snap.neighborhood_hit_2d.is_some());
    assert!(snap.cluster_density_measure.is_some());
    assert!(snap.davies_bouldin_ratio.is_some());
}

#[test]
fn test_compute_snapshot_perfect_embedding_scores() {
    // When embed_dist == high_dim_dist: trustworthiness/continuity
    // should all be 1.0 and normalized_stress/shepard_goodness should be 0/1.
    let n = 20;
    let d = make_distance_matrix(n, 42);
    let pts_2d = vec![0.0f64; n * 2]; // dummy 2D coords for the 2D half
    let snap = compute_snapshot(&d, &d, &pts_2d, None, n, 5);
    assert!(
        (snap.trustworthiness_manifold - 1.0).abs() < 1e-10,
        "trustworthiness_manifold should be 1.0, got {}",
        snap.trustworthiness_manifold
    );
    assert!(
        (snap.continuity_manifold - 1.0).abs() < 1e-10,
        "continuity_manifold should be 1.0, got {}",
        snap.continuity_manifold
    );
    assert!(
        snap.normalized_stress_manifold.abs() < 1e-10,
        "normalized_stress_manifold should be 0, got {}",
        snap.normalized_stress_manifold
    );
    assert!(
        (snap.shepard_goodness_manifold - 1.0).abs() < 1e-10,
        "shepard_goodness_manifold should be 1.0, got {}",
        snap.shepard_goodness_manifold
    );
}

#[test]
fn test_compute_snapshot_all_values_in_range() {
    let (pts_2d, labels) = make_clustered_2d(3, 10, 0.5, 99);
    let n = pts_2d.len() / 2;
    let d_high = make_distance_matrix(n, 1);
    let d_embed = euclidean_dist_2d(&pts_2d, n);
    let snap = compute_snapshot(&d_high, &d_embed, &pts_2d, Some(&labels), n, 5);
    assert!((0.0..=1.0).contains(&snap.trustworthiness_manifold));
    assert!((0.0..=1.0).contains(&snap.trustworthiness_2d));
    assert!((0.0..=1.0).contains(&snap.continuity_manifold));
    assert!((0.0..=1.0).contains(&snap.continuity_2d));
    assert!(snap.normalized_stress_manifold >= 0.0);
    assert!(snap.normalized_stress_2d >= 0.0);
    assert!((0.0..=1.0).contains(&snap.shepard_goodness_manifold));
    assert!((0.0..=1.0).contains(&snap.shepard_goodness_2d));
    assert!((0.0..=1.0).contains(&snap.neighborhood_hit_manifold.unwrap()));
    assert!((0.0..=1.0).contains(&snap.neighborhood_hit_2d.unwrap()));
}

#[test]
fn test_normalized_stress_scale_invariant() {
    // SNS must be invariant to uniform scaling of the embedded distances.
    // If it is not, the manifold and 2D variants will differ for Euclidean
    // embeddings where project_to_2d rescales coords for display.
    let n = 20;
    let d_high = make_distance_matrix(n, 1);
    let d_embed = make_distance_matrix(n, 2);
    let scale = 7.3_f64;
    let d_embed_scaled: Vec<f64> = d_embed.iter().map(|&v| v * scale).collect();
    let s1 = normalized_stress(&d_high, &d_embed, n);
    let s2 = normalized_stress(&d_high, &d_embed_scaled, n);
    assert!(
        (s1 - s2).abs() < 1e-10,
        "normalized_stress must be scale-invariant: {s1} != {s2}"
    );
}

#[test]
fn test_compute_snapshot_manifold_and_2d_can_differ() {
    // When the 2D distances differ from embed_dist the two variants must differ.
    let (pts_2d, _) = make_clustered_2d(2, 15, 1.0, 42);
    let n = pts_2d.len() / 2;
    let d = euclidean_dist_2d(&pts_2d, n);
    // Scaled 2D: shrink x, stretch y — creates different pairwise distances.
    let pts_scaled: Vec<f64> = pts_2d
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v * 3.0 } else { v * 0.3 })
        .collect();
    let snap = compute_snapshot(&d, &d, &pts_scaled, None, n, 5);
    // Manifold = self-comparison → stress 0; 2D uses different distances → stress > 0.
    assert!(
        snap.normalized_stress_2d > snap.normalized_stress_manifold,
        "2D stress ({}) should exceed manifold stress ({})",
        snap.normalized_stress_2d,
        snap.normalized_stress_manifold
    );
}
