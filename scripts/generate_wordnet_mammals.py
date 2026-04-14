"""
Generate the WordNet mammal subtree edge list for use with the Rust loader.

Usage:
    uv run python scripts/generate_wordnet_mammals.py --output www/public/data/wordnet/

Output files (written to --output directory):
    mammals_edges.tsv   — tab-separated parent_id<TAB>child_id pairs (integer IDs)
    mammals_labels.tsv  — one integer label per line (line i = label for node i)

Node IDs are assigned in BFS order starting from the root (mammal.n.01 = node 0).
Labels encode which direct child of mammal.n.01 each node descends from
(the root itself gets label 0).
"""

import argparse
import os
from collections import deque

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    raise SystemExit("NLTK is required: pip install nltk")


def get_hyponyms(synset, visited: set) -> list:
    """Return direct hyponyms not already visited."""
    children = []
    for hypo in synset.hyponyms():
        if hypo not in visited:
            visited.add(hypo)
            children.append(hypo)
    return children


def build_mammal_tree():
    """BFS over mammal.n.01 subtree. Returns (bfs_order, edges, depth2_label)."""
    root = wn.synset("mammal.n.01")
    visited = {root}
    queue = deque([(root, None, 0)])  # (synset, parent_synset, depth)
    bfs_order = []       # list of synsets in BFS order
    edges = []           # list of (parent_idx, child_idx)
    synset_to_idx = {}

    # For depth-2 labelling: every node gets the label of its depth-2 ancestor.
    depth2_ancestor = {}  # synset -> synset at depth 2 (or the node itself if depth <= 2)

    while queue:
        node, parent, depth = queue.popleft()
        idx = len(bfs_order)
        bfs_order.append(node)
        synset_to_idx[node] = idx

        if parent is not None:
            parent_idx = synset_to_idx[parent]
            edges.append((parent_idx, idx))

        # Depth-2 ancestor labelling
        if depth <= 2:
            depth2_ancestor[node] = node
        else:
            depth2_ancestor[node] = depth2_ancestor[parent]

        for child in get_hyponyms(node, visited):
            queue.append((child, node, depth + 1))

    # Map depth-2 ancestors to integer labels
    d2_synsets = sorted({depth2_ancestor[s] for s in bfs_order}, key=lambda s: synset_to_idx[s])
    d2_label_map = {s: i for i, s in enumerate(d2_synsets)}
    labels = [d2_label_map[depth2_ancestor[s]] for s in bfs_order]

    return bfs_order, edges, labels


def main():
    parser = argparse.ArgumentParser(description="Generate WordNet mammal subtree files.")
    parser.add_argument(
        "--output",
        default="www/public/data/wordnet",
        help="Output directory (default: www/public/data/wordnet)",
    )
    args = parser.parse_args()

    # Ensure WordNet is available
    try:
        wn.synset("mammal.n.01")
    except Exception:
        import nltk
        print("Downloading WordNet...")
        nltk.download("wordnet")

    print("Building mammal subtree...")
    bfs_order, edges, labels = build_mammal_tree()
    print(f"  {len(bfs_order)} synsets, {len(edges)} edges, {max(labels) + 1} label groups")

    os.makedirs(args.output, exist_ok=True)

    edges_path = os.path.join(args.output, "mammals_edges.tsv")
    with open(edges_path, "w") as f:
        for parent_id, child_id in edges:
            f.write(f"{parent_id}\t{child_id}\n")
    print(f"Wrote {edges_path}")

    labels_path = os.path.join(args.output, "mammals_labels.tsv")
    with open(labels_path, "w") as f:
        for lbl in labels:
            f.write(f"{lbl}\n")
    print(f"Wrote {labels_path}")

    print(f"\nDone. {len(bfs_order)} nodes total.")
    print(f"To use: --dataset wordnet_mammals --data-path <dir containing 'wordnet/'> --n-samples {len(bfs_order)}")


if __name__ == "__main__":
    main()
