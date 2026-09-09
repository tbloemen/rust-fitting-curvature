//! Unweighted graphs and rooted trees, and the shortest-path metric they carry.
//!
//! Two consumers share this: `data::load_wordnet_mammals`, which reads a real
//! hierarchy off disk, and `synthetic_data::generate_tree_graph`, which builds
//! one. Both want the same thing — an adjacency list turned into an all-pairs
//! hop-distance matrix — so the BFS lives here rather than being written twice.
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

/// A rooted `branching`-ary tree on `n` nodes, filled breadth-first.
///
/// Node 0 is the root and node `i`'s parent is `(i − 1) / branching`, so the
/// level-order numbering makes the parent map a closed form and every prefix of
/// `0..n` is itself a valid tree. That matters for the generator: asking for
/// 1000 nodes gives a complete tree with one partially-filled bottom level
/// rather than a truncation that orphans anything.
pub struct RootedTree {
    parent: Vec<usize>,
    depth: Vec<u32>,
    n: usize,
}

impl RootedTree {
    /// Complete `branching`-ary tree on `n` nodes.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` or `branching < 2`.
    #[must_use]
    pub fn complete(n: usize, branching: usize) -> Self {
        assert!(n > 0, "a tree needs at least a root");
        assert!(branching >= 2, "branching must be at least 2");

        let mut parent = vec![0usize; n];
        let mut depth = vec![0u32; n];
        for i in 1..n {
            let p = (i - 1) / branching;
            parent[i] = p;
            depth[i] = depth[p] + 1;
        }
        Self { parent, depth, n }
    }

    /// Undirected adjacency list: each non-root node is joined to its parent.
    #[must_use]
    pub fn adjacency(&self) -> Vec<Vec<usize>> {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        for i in 1..self.n {
            adj[self.parent[i]].push(i);
            adj[i].push(self.parent[i]);
        }
        adj
    }

    /// Depth of every node, root at 0.
    ///
    /// Kept separate from the labels on purpose: depth is the radial coordinate
    /// of the hierarchy and branch membership is the categorical one, and
    /// collapsing them into a single label vector is what made the old `tree`
    /// dataset put 98.5% of its points in one class.
    #[must_use]
    pub fn depths(&self) -> &[u32] {
        &self.depth
    }

    /// The number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false — [`RootedTree::complete`] rejects `n == 0`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Branch membership: which subtree rooted at depth `level` a node lies in.
    ///
    /// Labels are `1..=k` in node order at that depth; nodes shallower than
    /// `level` — the trunk — get label 0. A complete binary tree at `level = 3`
    /// therefore yields 8 branches plus a 7-node trunk.
    #[must_use]
    pub fn branch_labels(&self, level: u32) -> Vec<u32> {
        // parent[i] < i for every i > 0, so one forward pass resolves the whole
        // ancestor chain.
        let mut ancestor: Vec<Option<usize>> = vec![None; self.n];
        let mut next_label = 1u32;
        let mut label_of: Vec<u32> = vec![0; self.n];

        for i in 0..self.n {
            match self.depth[i].cmp(&level) {
                std::cmp::Ordering::Less => ancestor[i] = None,
                std::cmp::Ordering::Equal => {
                    ancestor[i] = Some(i);
                    label_of[i] = next_label;
                    next_label += 1;
                }
                std::cmp::Ordering::Greater => ancestor[i] = ancestor[self.parent[i]],
            }
        }

        ancestor
            .iter()
            .map(|a| a.map_or(0, |root| label_of[root]))
            .collect()
    }

    /// The tree metric: `d(u,v) = depth(u) + depth(v) − 2·depth(lca(u,v))`.
    ///
    /// Equal to the BFS hop distance by construction; used by the tests to
    /// check [`all_pairs_bfs_distances`] against the closed form.
    #[must_use]
    pub fn hop_distance(&self, mut u: usize, mut v: usize) -> u32 {
        let mut d = 0u32;
        while self.depth[u] > self.depth[v] {
            u = self.parent[u];
            d += 1;
        }
        while self.depth[v] > self.depth[u] {
            v = self.parent[v];
            d += 1;
        }
        while u != v {
            u = self.parent[u];
            v = self.parent[v];
            d += 2;
        }
        d
    }
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "hop counts are small integers held in f64; the equality is exact by construction and an epsilon would hide an off-by-one"
)]
mod tests {
    use super::*;

    #[test]
    fn complete_binary_tree_parents() {
        let t = RootedTree::complete(7, 2);
        assert_eq!(t.depths(), &[0, 1, 1, 2, 2, 2, 2]);
        assert_eq!(t.parent, vec![0, 0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn bfs_matches_the_closed_form() {
        let t = RootedTree::complete(31, 2);
        let d = all_pairs_bfs_distances(&t.adjacency(), 31);
        for u in 0..31 {
            for v in 0..31 {
                assert_eq!(
                    d[u * 31 + v],
                    f64::from(t.hop_distance(u, v)),
                    "pair ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn branch_labels_partition_the_leaves() {
        let t = RootedTree::complete(31, 2);
        let labels = t.branch_labels(2);
        // Depth 0 and 1 are the trunk: 3 nodes labelled 0.
        assert_eq!(labels.iter().filter(|&&l| l == 0).count(), 3);
        // Four subtrees of 7 nodes each.
        for branch in 1..=4 {
            assert_eq!(labels.iter().filter(|&&l| l == branch).count(), 7);
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
