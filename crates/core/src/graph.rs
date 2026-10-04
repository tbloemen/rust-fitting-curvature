//! Unweighted graphs and the shortest-path metric they carry.
//!
//! `data::load_wordnet_mammals` reads a real hierarchy off disk and turns its
//! adjacency list into an all-pairs hop-distance matrix with the BFS here.
//!
//! Nothing here touches the filesystem, so unlike `data` it compiles for wasm.

use crate::cast::count_to_f64;

/// Unweighted all-pairs BFS distances over an adjacency list.
///
/// Returns a flat `n × n` row-major matrix. Pairs with no path between them get
/// `2n` — large but finite, which is the convention `load_wordnet_mammals` has
/// always used: an infinity would poison every downstream mean and the graphs
/// this is applied to are connected in practice.
#[must_use]
pub fn all_pairs_bfs_distances(adj: &[Vec<usize>], n: usize) -> Vec<f64> {
    use std::collections::VecDeque;

    let mut dist_matrix = vec![f64::INFINITY; n * n];
    for src in 0..n {
        dist_matrix[src * n + src] = 0.0;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(src);
        while let Some(u) = queue.pop_front() {
            let d_u = dist_matrix[src * n + u];
            for &v in &adj[u] {
                if dist_matrix[src * n + v] == f64::INFINITY {
                    dist_matrix[src * n + v] = d_u + 1.0;
                    queue.push_back(v);
                }
            }
        }
        for j in 0..n {
            if dist_matrix[src * n + j] == f64::INFINITY {
                dist_matrix[src * n + j] = count_to_f64(n) * 2.0;
            }
        }
    }
    dist_matrix
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "hop counts are small integers held in f64; the equality is exact by construction and an epsilon would hide an off-by-one"
)]
mod tests {
    use super::*;

    #[test]
    fn bfs_counts_hops_along_a_path() {
        // 0 - 1 - 2 - 3
        let adj = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let d = all_pairs_bfs_distances(&adj, 4);
        for u in 0..4 {
            for v in 0..4 {
                assert_eq!(d[u * 4 + v], count_to_f64(u.abs_diff(v)), "pair ({u}, {v})");
            }
        }
    }

    #[test]
    fn unreachable_pairs_get_a_finite_distance() {
        // Two isolated nodes, no edges.
        let d = all_pairs_bfs_distances(&[Vec::new(), Vec::new()], 2);
        assert_eq!(d[1], 4.0);
        assert_eq!(d[0], 0.0);
    }
}
