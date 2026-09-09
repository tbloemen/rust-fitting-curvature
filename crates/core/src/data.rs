//! Data loading utilities for non-synthetic datasets.
//!
//! Note: filesystem access is not available in WASM; these functions
//! are only compiled for native targets.

#[cfg(not(target_arch = "wasm32"))]
use crate::cast::count_to_f64;

#[cfg(not(target_arch = "wasm32"))]
use crate::synthetic_data::DataPoints;

#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs::File,
    io::{BufReader, Read},
};

/// Load MNIST training images and labels from the IDX binary format.
///
/// `path` is the directory containing `train-images-idx3-ubyte` and
/// `train-labels-idx1-ubyte`. Returns a `DataPoints` with pixel values
/// normalised to [0, 1] and `distances` left empty (not precomputed).
///
/// # Errors
///
/// Returns `Err` if the IDX files are missing, have an invalid magic number, or cannot be read.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_mnist(path: &str, n_samples: usize) -> Result<DataPoints, String> {
    let images_path = format!("{path}/train-images-idx3-ubyte");
    let labels_path = format!("{path}/train-labels-idx1-ubyte");

    let images = read_idx3_ubyte(&images_path)?;
    let labels_raw = read_idx1_ubyte(&labels_path)?;

    let n_features = 28 * 28;
    let actual_samples = n_samples.min(images.len() / n_features);

    let x = images[..actual_samples * n_features]
        .chunks(n_features)
        .flat_map(|row| row.iter().map(|&p| f64::from(p) / 255.0))
        .collect();
    let labels = labels_raw[..actual_samples]
        .iter()
        .map(|&l| u32::from(l))
        .collect();

    Ok(DataPoints {
        x,
        labels,
        n_points: actual_samples,
        ambient_dim: n_features,
        distances: Vec::new(), // not precomputed; callers compute as needed
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn read_idx3_ubyte(path: &str) -> Result<Vec<u8>, String> {
    let mut reader =
        BufReader::new(File::open(path).map_err(|e| format!("Failed to open {path}: {e}"))?);
    let magic = read_u32(&mut reader)?;
    if magic != 2051 {
        return Err(format!("Invalid magic number for idx3: {magic}"));
    }
    let num_images = read_u32(&mut reader)? as usize;
    let _rows = read_u32(&mut reader)?;
    let _cols = read_u32(&mut reader)?;
    let mut data = vec![0u8; num_images * 28 * 28];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("Failed to read image data: {e}"))?;
    Ok(data)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_idx1_ubyte(path: &str) -> Result<Vec<u8>, String> {
    let mut reader =
        BufReader::new(File::open(path).map_err(|e| format!("Failed to open {path}: {e}"))?);
    let magic = read_u32(&mut reader)?;
    if magic != 2049 {
        return Err(format!("Invalid magic number for idx1: {magic}"));
    }
    let num_items = read_u32(&mut reader)? as usize;
    let mut data = vec![0u8; num_items];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("Failed to read label data: {e}"))?;
    Ok(data)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_u32(reader: &mut impl Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read u32: {e}"))?;
    Ok(u32::from_be_bytes(buf))
}

/// Load Fashion-MNIST from the IDX binary format.
///
/// Fashion-MNIST uses the exact same file format and filenames as MNIST.
/// `path` is the directory containing `train-images-idx3-ubyte` and
/// `train-labels-idx1-ubyte`.
///
/// # Errors
///
/// Returns `Err` if the IDX files are missing, have an invalid magic number, or cannot be read.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_fashion_mnist(path: &str, n_samples: usize) -> Result<DataPoints, String> {
    load_mnist(path, n_samples)
}

/// Load the `WordNet` mammal subtree from a pre-generated edge list.
///
/// `path` is the directory containing:
/// - `mammals_edges.tsv`: tab-separated `parent_id\tchild_id` pairs (integer IDs, no header).
/// - `mammals_labels.tsv` (optional): one integer label per line (line i = label for node i).
///   If absent, labels are derived from each node's depth-2 ancestor from the root (node 0).
///
/// Returns a `DataPoints` where `distances` holds the all-pairs BFS shortest-path
/// distance matrix (flat n × n), suitable for use with `EmbeddingState::from_distances`.
///
/// # Errors
///
/// Returns `Err` if edge/label files are missing, contain non-numeric IDs, or have structural mismatches.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_wordnet_mammals(path: &str, n_samples: usize) -> Result<DataPoints, String> {
    use std::collections::VecDeque;

    // --- Parse edge list ---
    let (edges, n_nodes_full) = parse_edge_list(path)?;

    // --- Build adjacency list (undirected for BFS) ---
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_nodes_full];
    for &(p, c) in &edges {
        adj[p].push(c);
        adj[c].push(p);
    }

    // --- BFS order from root (node 0) to select n_samples nodes ---
    let n = n_samples.min(n_nodes_full);
    let mut bfs_order: Vec<usize> = Vec::with_capacity(n);
    let mut visited = vec![false; n_nodes_full];
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(0);
    visited[0] = true;
    while let Some(node) = queue.pop_front() {
        bfs_order.push(node);
        if bfs_order.len() >= n {
            break;
        }
        for &nb in &adj[node] {
            if !visited[nb] {
                visited[nb] = true;
                queue.push_back(nb);
            }
        }
    }
    let n = bfs_order.len();

    // Map original IDs → compact indices [0, n).
    let mut id_to_idx = vec![usize::MAX; n_nodes_full];
    for (idx, &orig) in bfs_order.iter().enumerate() {
        id_to_idx[orig] = idx;
    }

    // Compact adjacency list for selected nodes.
    let mut compact_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(p, c) in &edges {
        let pi = id_to_idx[p];
        let ci = id_to_idx[c];
        if pi != usize::MAX && ci != usize::MAX {
            compact_adj[pi].push(ci);
            compact_adj[ci].push(pi);
        }
    }

    // --- BFS from every node to compute all-pairs distances ---
    let dist_matrix = compute_all_pairs_bfs_distances(&compact_adj, n);

    // --- Labels ---
    let labels = load_or_derive_labels(path, &bfs_order, &compact_adj, n);

    Ok(DataPoints {
        // No feature representation for graph data; the distance matrix drives
        // affinities and evaluation via the `distances` field.
        x: Vec::new(),
        labels,
        n_points: n,
        ambient_dim: 0,
        distances: dist_matrix,
    })
}

/// Parse the edge-list TSV into a list of parent→child edges and the total
/// number of nodes.
#[cfg(not(target_arch = "wasm32"))]
fn parse_edge_list(path: &str) -> Result<(Vec<(usize, usize)>, usize), String> {
    use std::io::BufRead;

    let edges_path = format!("{path}/mammals_edges.tsv");
    let edges_file =
        File::open(&edges_path).map_err(|e| format!("Failed to open {edges_path}: {e}"))?;
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut max_id = 0usize;
    for (line_no, line) in BufReader::new(edges_file).lines().enumerate() {
        let line = line.map_err(|e| format!("Read error in {edges_path}: {e}"))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let parse_id = |s: Option<&str>, field: &str| -> Result<usize, String> {
            s.ok_or_else(|| format!("{edges_path}:{}: missing {field}", line_no + 1))?
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("{edges_path}:{}: bad {field}: {e}", line_no + 1))
        };
        let parent = parse_id(parts.next(), "parent_id")?;
        let child = parse_id(parts.next(), "child_id")?;
        max_id = max_id.max(parent).max(child);
        edges.push((parent, child));
    }
    Ok((edges, max_id + 1))
}

/// Compute all-pairs BFS distances on the compact adjacency list. Returns a
/// flat `n × n` distance matrix (row-major). Unreachable nodes get a large
/// but finite distance.
#[cfg(not(target_arch = "wasm32"))]
fn compute_all_pairs_bfs_distances(compact_adj: &[Vec<usize>], n: usize) -> Vec<f64> {
    use std::collections::VecDeque;

    let mut dist_matrix = vec![f64::INFINITY; n * n];
    for src in 0..n {
        dist_matrix[src * n + src] = 0.0;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(src);
        while let Some(u) = queue.pop_front() {
            let d_u = dist_matrix[src * n + u];
            for &v in &compact_adj[u] {
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

/// Load labels from the labels TSV if present, otherwise derive them from the
/// tree structure (depth-2 ancestor grouping).
#[cfg(not(target_arch = "wasm32"))]
fn load_or_derive_labels(
    path: &str,
    bfs_order: &[usize],
    compact_adj: &[Vec<usize>],
    n: usize,
) -> Vec<u32> {
    use std::collections::VecDeque;
    use std::io::{BufRead, BufReader};

    let labels_path = format!("{path}/mammals_labels.tsv");
    if let Ok(file) = File::open(&labels_path) {
        let raw: Vec<u32> = BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .filter_map(|l| l.trim().parse::<u32>().ok())
            .collect();
        bfs_order
            .iter()
            .map(|&orig| raw.get(orig).copied().unwrap_or(0))
            .collect()
    } else {
        let mut depth2_label = vec![0u32; n];
        let mut group_queue: VecDeque<(usize, u32, usize)> = VecDeque::new();
        let mut assigned = vec![false; n];
        group_queue.push_back((0, 0, 0));
        assigned[0] = true;
        let mut next_group = 1u32;
        while let Some((node, group, depth)) = group_queue.pop_front() {
            depth2_label[node] = group;
            for &nb in &compact_adj[node] {
                if !assigned[nb] {
                    assigned[nb] = true;
                    let child_group = if depth < 2 {
                        let g = next_group;
                        next_group += 1;
                        g
                    } else {
                        group
                    };
                    group_queue.push_back((nb, child_group, depth + 1));
                }
            }
        }
        depth2_label
    }
}

/// Load pre-processed PBMC single-cell RNA-seq data from a TSV file.
///
/// `path` is the directory containing `pbmc_pca.tsv`.
/// Expected format (tab-separated):
/// - Optional comment lines starting with `#` are skipped.
/// - Optional header row (detected when the first field is non-numeric).
/// - Each data row: optional string label column followed by numeric PCA features.
///
/// Labels are mapped from strings to integers (sorted for reproducibility).
/// Returns up to `n_samples` rows. `distances` is left empty.
///
/// # Errors
///
/// Returns `Err` if `pbmc_pca.tsv` is missing, empty, or has non-numeric features.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_pbmc(path: &str, n_samples: usize) -> Result<DataPoints, String> {
    use std::collections::BTreeMap;
    use std::io::{BufRead, BufReader};

    let file_path = format!("{path}/pbmc_pca.tsv");
    let file = File::open(&file_path).map_err(|e| format!("Failed to open {file_path}: {e}"))?;

    let mut data_lines: Vec<String> = BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
        .collect();

    if data_lines.is_empty() {
        return Err(format!("{file_path}: file is empty"));
    }

    // Detect and skip header: if first field of first line is non-numeric, it's a header.
    let first_fields: Vec<&str> = data_lines[0].split('\t').collect();
    let has_header = first_fields[0].parse::<f64>().is_err();
    if has_header {
        data_lines.remove(0);
    }

    if data_lines.is_empty() {
        return Err(format!("{file_path}: no data rows after header"));
    }

    // Detect label column: if the first field of any row is non-numeric, treat col 0 as labels.
    let has_label_col = data_lines.iter().any(|l| {
        l.split('\t')
            .next()
            .is_some_and(|f| f.parse::<f64>().is_err())
    });

    let mut raw_labels: Vec<String> = Vec::new();
    let mut x: Vec<f64> = Vec::new();
    let mut n_features = 0usize;

    for (row_idx, line) in data_lines.iter().take(n_samples).enumerate() {
        let fields: Vec<&str> = line.split('\t').collect();
        let feature_start = usize::from(has_label_col);
        let features: Result<Vec<f64>, _> = fields[feature_start..]
            .iter()
            .map(|f| f.trim().parse::<f64>())
            .collect();
        let features =
            features.map_err(|e| format!("{file_path}: row {}: parse error: {e}", row_idx + 1))?;

        if row_idx == 0 {
            n_features = features.len();
        } else if features.len() != n_features {
            return Err(format!(
                "{file_path}: row {} has {} features, expected {n_features}",
                row_idx + 1,
                features.len()
            ));
        }

        if has_label_col {
            raw_labels.push(fields[0].trim().to_string());
        }
        x.extend(features);
    }

    let n_points = x.len() / n_features.max(1);

    // Map string labels to sorted integer IDs.
    let labels: Vec<u32> = if has_label_col {
        let mut label_map: BTreeMap<String, u32> = BTreeMap::new();
        let mut next_id = 0u32;
        let mut ids = Vec::with_capacity(raw_labels.len());
        for lbl in &raw_labels {
            let id = *label_map.entry(lbl.clone()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            ids.push(id);
        }
        ids
    } else {
        vec![0u32; n_points]
    };

    Ok(DataPoints {
        x,
        labels,
        n_points,
        ambient_dim: n_features,
        distances: Vec::new(),
    })
}
