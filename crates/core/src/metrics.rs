//! Embedding quality metrics.
//!
//! Metrics grouped into:
//! - A. Local structure preservation (trustworthiness, continuity, knn_overlap)
//! - B. Global geometry preservation (geodesic_distortion)
//! - C. Space efficiency (radial_distribution)
//! - D. Perceptual evaluation (class_density_measure, cluster_density_measure,
//!   davies_bouldin, davies_bouldin_ratio)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute ranks: for each point i, ranks[i*n + j] = rank of j sorted by
/// distance from i (0 = self, 1 = nearest neighbor, etc.).
fn compute_ranks(distances: &[f64], n: usize) -> Vec<usize> {
    let mut ranks = vec![0usize; n * n];
    for i in 0..n {
        let mut indices: Vec<usize> = (0..n).collect();
        // total_cmp gives a proper total order (NaN/inf sort last), so sort_by
        // cannot panic when a diverged embedding produces non-finite distances.
        indices.sort_by(|&a, &b| distances[i * n + a].total_cmp(&distances[i * n + b]));
        for (rank, &j) in indices.iter().enumerate() {
            ranks[i * n + j] = rank;
        }
    }
    ranks
}

/// Compute k-nearest neighbor index sets (excluding self).
fn knn_index_sets(dist: &[f64], n: usize, k: usize) -> Vec<Vec<usize>> {
    (0..n)
        .map(|i| {
            let mut indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            indices.sort_by(|&a, &b| dist[i * n + a].total_cmp(&dist[i * n + b]));
            indices.truncate(k);
            indices
        })
        .collect()
}

/// Compute pairwise Euclidean distance matrix from 2D points (flat [x,y] pairs).
///
/// Use this to obtain "after-projection" distances from the output of
/// `visualisation::project_to_2d`, so that any metric can be evaluated on
/// what the viewer actually sees rather than on the manifold geometry.
pub fn euclidean_dist_2d(pts_2d: &[f64], n: usize) -> Vec<f64> {
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pts_2d[i * 2] - pts_2d[j * 2];
            let dy = pts_2d[i * 2 + 1] - pts_2d[j * 2 + 1];
            let d = (dx * dx + dy * dy).sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// A. Local structure preservation
// ---------------------------------------------------------------------------

/// Trustworthiness (Venna & Kaski 2006).
///
/// Measures whether points that appear as neighbors in the embedding are also
/// neighbors in the original space. Penalizes "false neighbors" in the embedding.
/// Returns a value in [0, 1], higher is better.
pub fn trustworthiness(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    let k = k.min(n - 2);
    if k == 0 || n < 3 {
        return 1.0;
    }

    let ranks_high = compute_ranks(high_dim_distances, n);
    let embed_knn = knn_index_sets(embedded_distances, n, k);

    let denom = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0);
    if denom < 1e-12 {
        return 1.0;
    }

    let mut penalty = 0.0;
    for i in 0..n {
        for &j in &embed_knn[i] {
            // ranks_high uses 0-based ranks where 0 = self, so rank > k means
            // j is NOT among i's k nearest in high-dim space
            let r = ranks_high[i * n + j];
            if r > k {
                penalty += (r - k) as f64;
            }
        }
    }

    1.0 - (2.0 / denom) * penalty
}

/// Continuity (Venna & Kaski 2006).
///
/// Measures whether points that are neighbors in the original space remain
/// neighbors in the embedding. Penalizes "missed neighbors" from the original.
/// Returns a value in [0, 1], higher is better.
pub fn continuity(
    high_dim_distances: &[f64],
    embedded_distances: &[f64],
    n: usize,
    k: usize,
) -> f64 {
    let k = k.min(n - 2);
    if k == 0 || n < 3 {
        return 1.0;
    }

    let ranks_embed = compute_ranks(embedded_distances, n);
    let high_knn = knn_index_sets(high_dim_distances, n, k);

    let denom = n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0);
    if denom < 1e-12 {
        return 1.0;
    }

    let mut penalty = 0.0;
    for i in 0..n {
        for &j in &high_knn[i] {
            let r = ranks_embed[i * n + j];
            if r > k {
                penalty += (r - k) as f64;
            }
        }
    }

    1.0 - (2.0 / denom) * penalty
}

// ---------------------------------------------------------------------------
// B. Global geometry preservation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// C. Space efficiency
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// D. Perceptual evaluation
// ---------------------------------------------------------------------------

/// Cluster Density Measure (ClDM) from Albuquerque et al. (2010).
///
/// Uses the label-based cluster formula on 2D projected coordinates:
/// ClDM = (1/K) * sum_{k<l} d²_{k,l} / (r_k * r_l)
///
/// Measures how well-separated and compact the clusters are.
/// Higher values = better separated clusters.
pub fn cluster_density_measure(pts_2d: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    // Compute centroids and radii per cluster
    let mut centroids = vec![(0.0f64, 0.0f64); k];
    let mut counts = vec![0usize; k];
    let mut radii = vec![0.0f64; k];

    for i in 0..n {
        let label_idx = unique_labels.iter().position(|&l| l == labels[i]).unwrap();
        centroids[label_idx].0 += pts_2d[i * 2];
        centroids[label_idx].1 += pts_2d[i * 2 + 1];
        counts[label_idx] += 1;
    }
    for ci in 0..k {
        if counts[ci] > 0 {
            centroids[ci].0 /= counts[ci] as f64;
            centroids[ci].1 /= counts[ci] as f64;
        }
    }

    // Compute average radius per cluster
    for i in 0..n {
        let label_idx = unique_labels.iter().position(|&l| l == labels[i]).unwrap();
        let dx = pts_2d[i * 2] - centroids[label_idx].0;
        let dy = pts_2d[i * 2 + 1] - centroids[label_idx].1;
        radii[label_idx] += (dx * dx + dy * dy).sqrt();
    }
    for ci in 0..k {
        radii[ci] = if counts[ci] > 0 {
            (radii[ci] / counts[ci] as f64).max(1e-12)
        } else {
            1e-12
        };
    }

    let mut cldm = 0.0;
    for ki in 0..k {
        for kj in (ki + 1)..k {
            let dx = centroids[ki].0 - centroids[kj].0;
            let dy = centroids[ki].1 - centroids[kj].1;
            let d_sq = dx * dx + dy * dy;
            cldm += d_sq / (radii[ki] * radii[kj]);
        }
    }
    cldm / k as f64
}

/// Davies-Bouldin index from precomputed distance matrix.
/// Lower = better separated, more compact clusters.
pub fn davies_bouldin(distances: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    let cluster_indices: Vec<Vec<usize>> = unique_labels
        .iter()
        .map(|&lbl| (0..n).filter(|&i| labels[i] == lbl).collect())
        .collect();

    // Compute scatters and medoids
    let mut scatters = vec![0.0f64; k];
    let mut medoid_indices = vec![0usize; k];

    for (ci, indices) in cluster_indices.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        // Find medoid (point with smallest total distance to others)
        let mut best_total = f64::INFINITY;
        for &candidate in indices {
            let total: f64 = indices.iter().map(|&j| distances[candidate * n + j]).sum();
            if total < best_total {
                best_total = total;
                medoid_indices[ci] = candidate;
            }
        }
        // Scatter: mean distance to medoid
        let medoid = medoid_indices[ci];
        scatters[ci] = indices
            .iter()
            .map(|&j| distances[medoid * n + j])
            .sum::<f64>()
            / indices.len() as f64;
    }

    // DB index
    let mut db = 0.0;
    for i in 0..k {
        let mut max_ratio = 0.0f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let d_ij = distances[medoid_indices[i] * n + medoid_indices[j]];
            if d_ij < 1e-12 {
                continue;
            }
            let ratio = (scatters[i] + scatters[j]) / d_ij;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        db += max_ratio;
    }
    db / k as f64
}

/// Davies-Bouldin ratio: DB_high / DB_projected.
///
/// Computes the DB index on both the high-dimensional data distances and
/// the 2D projected Euclidean distances. A higher ratio indicates the
/// projection preserves or improves cluster separation relative to the
/// original data (Di Caro et al. 2010).
pub fn davies_bouldin_ratio(
    high_dim_distances: &[f64],
    pts_2d: &[f64],
    labels: &[u32],
    n: usize,
) -> f64 {
    let dist_2d = euclidean_dist_2d(pts_2d, n);
    let db_high = davies_bouldin(high_dim_distances, labels, n);
    let db_proj = davies_bouldin(&dist_2d, labels, n);
    if db_proj < 1e-12 {
        return 0.0;
    }
    db_high / db_proj
}

// ---------------------------------------------------------------------------
// E. Distance-rank preservation
// ---------------------------------------------------------------------------

/// The rank variable `R[X]` of `values`, using **fractional ranks**: identical
/// values "are each assigned fractional ranks equal to the average of their
/// positions" (Spearman's rank correlation coefficient, *Definition and
/// calculation*). 0-based here, so rank 0 is the smallest value.
///
/// Ties are not exotic in this codebase, which is why the fractional
/// convention matters rather than being a formality.
fn fractional_rank_vector(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < indices.len() {
        let mut end = start;
        while end + 1 < indices.len() && values[indices[end + 1]] == values[indices[start]] {
            end += 1;
        }
        let fractional_rank = (start + end) as f64 / 2.0;
        for &idx in &indices[start..=end] {
            ranks[idx] = fractional_rank;
        }
        start = end + 1;
    }
    ranks
}

/// Scale-normalized stress (SNS) from Damrich & Hamprecht 2022.
///
/// Standard normalized stress is scale-sensitive: embeddings that preserve
/// distance shape but differ in overall scale appear poor. SNS removes this
/// by finding the optimal scale factor α before computing stress:
///
/// `α  = Σ_{i,j} d(x,x') · ‖y−y'‖ / Σ_{i,j} d(x,x')²`
/// `SNS = Σ_{i,j} [d(x,x') − α·‖y−y'‖]² / Σ_{i,j} d(x,x')²`
///
/// Returns a value in [0, 1], with **0 being best**. Because α is optimised
/// out, manifold and 2D variants are identical for Euclidean embeddings where
/// `project_to_2d` only rescales coordinates for display.
pub fn normalized_stress(high_dim_distances: &[f64], embedded_distances: &[f64], n: usize) -> f64 {
    let mut cross = 0.0;
    let mut embed_sq = 0.0;
    let mut high_sq = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_h = high_dim_distances[i * n + j];
            let d_e = embedded_distances[i * n + j];
            cross += d_h * d_e;
            embed_sq += d_e * d_e;
            high_sq += d_h * d_h;
        }
    }
    if high_sq < 1e-12 {
        return 0.0;
    }
    let alpha = if embed_sq < 1e-12 {
        1.0
    } else {
        cross / embed_sq
    };

    let mut numerator = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d_h = high_dim_distances[i * n + j];
            let d_e = embedded_distances[i * n + j];
            let diff = d_h - alpha * d_e;
            numerator += diff * diff;
        }
    }
    (numerator / high_sq).sqrt()
}

/// Neighborhood hit (van der Maaten 2009).
///
/// Measures how well the k-nearest-neighbor structure in the embedding aligns
/// with class labels: for each point, the fraction of its k-NN in the
/// embedding that share the same label, averaged over all points.
///
/// `M_NH = (1/N) * sum_i |{j ∈ kNN_embed(i) : label[j] = label[i]}| / k`
///
/// Returns a value in [0, 1], with **1 being best** (all k-NN same-class).
/// Requires labeled data. Pass manifold geodesic or 2D Euclidean distances
/// for the before/after projection distinction.
pub fn neighborhood_hit(embedded_distances: &[f64], labels: &[u32], n: usize, k: usize) -> f64 {
    let k = k.min(n - 1);
    if k == 0 {
        return 1.0;
    }

    let knn = knn_index_sets(embedded_distances, n, k);
    let mut total = 0.0;
    for i in 0..n {
        let same = knn[i].iter().filter(|&&j| labels[j] == labels[i]).count();
        total += same as f64 / k as f64;
    }
    total / n as f64
}

/// Shepard goodness `M_shep` (Espadoto et al.): Spearman's rank correlation
/// coefficient `r_s` between the pairwise distances of the original space and
/// those of the embedding.
///
/// The observations `(X_i, Y_i)` are the `m = n(n−1)/2` upper-triangle point
/// pairs — note that the statistical sample size is `m`, not this function's
/// `n`, which counts *points*, not observations. `r_s` is then a scalar measure
/// of how well the global rank-order of distances is preserved.
///
/// Computed from the general definition, as the Pearson correlation
/// coefficient of the two rank variables:
///
/// `r_s = ρ(R[X], R[Y]) = cov(R[X], R[Y]) / (σ_R[X] · σ_R[Y])`
///
/// and deliberately **not** via the familiar shortcut
///
/// `r_s = 1 − 6·Σd_i² / (m(m²−1))`,  `d_i = R[X_i] − R[Y_i]`
///
/// which "applies only when all n ranks are distinct integers (no ties)".
/// These distance vectors are routinely full of ties (see
/// [`fractional_rank_vector`]), so the standard guidance holds: with ties
/// present that formula "should not be used" and "the Pearson correlation
/// coefficient should be calculated on the ranks" instead. Using it anyway
/// with ordinal ranks scored a fully collapsed embedding — every pairwise
/// distance identical, so no rank information at all — at 0.63 on
/// `tree_structured` and 0.51 on `hyperbolic_shells`, purely from tie-breaking
/// by point index.
///
/// Two project-specific deviations from textbook `r_s ∈ [−1, 1]`:
///
/// - The result is **normalised onto [0, 1]** by the order-preserving affine
///   map `(r_s + 1) / 2`, so that every reported metric shares one range.
///   1 is perfect rank-order preservation, 0.5 is rank-order independence
///   (the no-skill value of a rank correlation), and 0 would be exact rank
///   reversal. The map is strictly monotone, so no information in `r_s` is
///   lost and the induced ordering of embeddings is unchanged; note that it
///   moves the no-skill point off zero, so **0.5, not 0, is the score of an
///   embedding that preserves nothing**.
/// - Degenerate input — either side constant, so `σ_R = 0` and `r_s` is
///   undefined (`scipy.stats.spearmanr` returns NaN here) — returns 0.5, the
///   image of "no rank information", rather than NaN, because this feeds a
///   Pareto objective where NaN is a hazard.
pub fn shepard_goodness(high_dim_distances: &[f64], embedded_distances: &[f64], n: usize) -> f64 {
    let m = n * (n - 1) / 2;
    if m < 2 {
        return 1.0;
    }

    let mut d_high = Vec::with_capacity(m);
    let mut d_embed = Vec::with_capacity(m);
    for i in 0..n {
        for j in (i + 1)..n {
            d_high.push(high_dim_distances[i * n + j]);
            d_embed.push(embedded_distances[i * n + j]);
        }
    }

    let r_x = fractional_rank_vector(&d_high);
    let r_y = fractional_rank_vector(&d_embed);

    // cov(R[X], R[Y]) / (σ_R[X] · σ_R[Y]), with the 1/m factors cancelling.
    // Both mean ranks are (m-1)/2 whatever the tie pattern, since fractional
    // ranks redistribute 0..m-1 without changing their sum; the σ do change
    // — ties shrink them — which is exactly what the shortcut cannot see.
    let mean_rank = (m - 1) as f64 / 2.0;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&rx, &ry) in r_x.iter().zip(r_y.iter()) {
        let (dx, dy) = (rx - mean_rank, ry - mean_rank);
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    // A constant distance vector collapses every rank onto `mean_rank`, so
    // σ_R = 0 and r_s is undefined. That is a total loss of rank structure,
    // which on this scale is the no-skill value 0.5 — not 1, and not the 0
    // that exact rank *reversal* would earn.
    let sigma_product = (var_x * var_y).sqrt();
    if sigma_product < 1e-12 {
        return 0.5;
    }

    // (r_s + 1) / 2, clamped only against floating-point overshoot at the ends.
    ((cov / sigma_product + 1.0) / 2.0).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// All quality metrics for a completed embedding, in both manifold and
/// 2D-projected variants (the "before" / "after projecting" distinction).
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    // A. Local structure preservation
    pub trustworthiness_manifold: f64,
    pub trustworthiness_2d: f64,
    pub continuity_manifold: f64,
    pub continuity_2d: f64,
    // B. Distance preservation
    pub normalized_stress_manifold: f64,
    pub normalized_stress_2d: f64,
    pub shepard_goodness_manifold: f64,
    pub shepard_goodness_2d: f64,
    // C. Label-dependent (None when no labels provided)
    pub neighborhood_hit_manifold: Option<f64>,
    pub neighborhood_hit_2d: Option<f64>,
    // D. Class separation — 2D only, label-dependent
    pub cluster_density_measure: Option<f64>,
    pub davies_bouldin_ratio: Option<f64>,
}

/// Compute a full metrics snapshot.
///
/// `high_dim_dist` — pairwise distances in input space.
/// `embed_dist`    — manifold geodesic distances (before projection).
/// `pts_2d`        — flat (x,y) pairs from `project_to_2d` (after projection).
/// `labels`        — optional class labels; label-dependent metrics are `None` when absent.
/// `k`             — neighbourhood size used for kNN metrics.
pub fn compute_snapshot(
    high_dim_dist: &[f64],
    embed_dist: &[f64],
    pts_2d: &[f64],
    labels: Option<&[u32]>,
    n: usize,
    k: usize,
) -> MetricsSnapshot {
    let dist_2d = euclidean_dist_2d(pts_2d, n);

    let (neighborhood_hit_manifold, neighborhood_hit_2d, cluster_density, db_ratio) =
        if let Some(lbl) = labels {
            (
                Some(neighborhood_hit(embed_dist, lbl, n, k)),
                Some(neighborhood_hit(&dist_2d, lbl, n, k)),
                Some(cluster_density_measure(pts_2d, lbl, n)),
                Some(davies_bouldin_ratio(high_dim_dist, pts_2d, lbl, n)),
            )
        } else {
            (None, None, None, None)
        };

    MetricsSnapshot {
        trustworthiness_manifold: trustworthiness(high_dim_dist, embed_dist, n, k),
        trustworthiness_2d: trustworthiness(high_dim_dist, &dist_2d, n, k),
        continuity_manifold: continuity(high_dim_dist, embed_dist, n, k),
        continuity_2d: continuity(high_dim_dist, &dist_2d, n, k),
        normalized_stress_manifold: normalized_stress(high_dim_dist, embed_dist, n),
        normalized_stress_2d: normalized_stress(high_dim_dist, &dist_2d, n),
        shepard_goodness_manifold: shepard_goodness(high_dim_dist, embed_dist, n),
        shepard_goodness_2d: shepard_goodness(high_dim_dist, &dist_2d, n),
        neighborhood_hit_manifold,
        neighborhood_hit_2d,
        cluster_density_measure: cluster_density,
        davies_bouldin_ratio: db_ratio,
    }
}

/// Dunn index: ratio of minimum inter-cluster distance to maximum intra-cluster diameter.
/// Higher = better clustering.
pub fn dunn_index(embedded_distances: &[f64], labels: &[u32], n: usize) -> f64 {
    let mut unique_labels: Vec<u32> = labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let k = unique_labels.len();
    if k < 2 {
        return 0.0;
    }

    // Cluster indices
    let cluster_indices: Vec<Vec<usize>> = unique_labels
        .iter()
        .map(|&lbl| (0..n).filter(|&i| labels[i] == lbl).collect())
        .collect();

    // Max intra-cluster diameter
    let mut max_intra = 0.0f64;
    for indices in &cluster_indices {
        for &a in indices {
            for &b in indices {
                let d = embedded_distances[a * n + b];
                if d > max_intra {
                    max_intra = d;
                }
            }
        }
    }
    if max_intra < 1e-12 {
        return 0.0;
    }

    // Min inter-cluster distance
    let mut min_inter = f64::INFINITY;
    for ci in 0..k {
        for cj in (ci + 1)..k {
            for &a in &cluster_indices[ci] {
                for &b in &cluster_indices[cj] {
                    let d = embedded_distances[a * n + b];
                    if d < min_inter {
                        min_inter = d;
                    }
                }
            }
        }
    }

    min_inter / max_intra
}
