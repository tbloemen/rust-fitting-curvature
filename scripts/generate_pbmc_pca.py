"""
Download and preprocess PBMC 3k, then export PCA coordinates + cell-type labels.

Usage:
    uv run python scripts/generate_pbmc_pca.py --output www/public/data/pbmc/

Output:
    pbmc_pca.tsv  — tab-separated file with columns:
                    cell_type  PC1  PC2  ...  PC50
                    (one row per cell, cell type as first column)

The script uses scanpy's built-in PBMC 3k dataset loader, which downloads the
filtered gene-barcode matrix (~6 MB) from the 10x Genomics website on first run.
Subsequent runs use the scanpy cache (~/.cache/scanpy/).

Cell-type annotations follow the standard Seurat/Scanpy PBMC 3k tutorial:
Leiden clusters are mapped to the canonical 8 cell types.
"""

import argparse
import os

try:
    import scanpy as sc
except ImportError:
    raise SystemExit(
        "scanpy and numpy are required.\n"
        "Run: uv add scanpy numpy leidenalg\n"
        "or:  pip install scanpy numpy leidenalg"
    )

# Canonical Leiden→cell-type mapping from the Scanpy PBMC 3k tutorial.
# Cluster numbers may vary slightly depending on random state; adjust if needed.
LEIDEN_TO_CELLTYPE = {
    "0": "CD4 T",
    "1": "CD14 Monocytes",
    "2": "B cells",
    "3": "CD8 T",
    "4": "NK cells",
    "5": "FCGR3A Monocytes",
    "6": "Dendritic",
    "7": "Megakaryocytes",
}


def preprocess(adata):
    """Standard PBMC 3k preprocessing pipeline (Scanpy tutorial)."""
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50)
    return adata


def annotate(adata):
    """Run Leiden clustering and map clusters to cell types."""
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata, resolution=0.5, random_state=42, flavor="igraph", n_iterations=2, directed=False)
    # .astype(str) converts the Categorical leiden column to plain strings so
    # fillna("Unknown") doesn't fail when "Unknown" isn't already a category.
    adata.obs["cell_type"] = adata.obs["leiden"].astype(str).map(LEIDEN_TO_CELLTYPE).fillna("Unknown")
    return adata


def main():
    parser = argparse.ArgumentParser(description="Generate PBMC 3k PCA file.")
    parser.add_argument(
        "--output",
        default="www/public/data/pbmc",
        help="Output directory (default: www/public/data/pbmc)",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of PCA components to export (default: 50)",
    )
    args = parser.parse_args()

    sc.settings.verbosity = 2

    print("Loading PBMC 3k dataset (downloads ~6 MB on first run)...")
    adata = sc.datasets.pbmc3k()
    print(f"  Raw: {adata.n_obs} cells × {adata.n_vars} genes")

    print("Preprocessing...")
    adata = preprocess(adata)
    print(f"  After QC: {adata.n_obs} cells")

    print("Clustering and annotating cell types...")
    adata = annotate(adata)
    counts = adata.obs["cell_type"].value_counts()
    print("  Cell type counts:")
    for ct, n in counts.items():
        print(f"    {ct}: {n}")

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "pbmc_pca.tsv")

    n_pcs = min(args.n_pcs, adata.obsm["X_pca"].shape[1])
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]

    print(f"\nWriting {out_path} ({adata.n_obs} cells × {n_pcs} PCs + cell_type)...")
    with open(out_path, "w") as f:
        # Header
        f.write("cell_type\t" + "\t".join(pc_cols) + "\n")
        pca = adata.obsm["X_pca"]
        cell_types = adata.obs["cell_type"].values
        for i in range(adata.n_obs):
            ct = cell_types[i]
            coords = "\t".join(f"{pca[i, j]:.6f}" for j in range(n_pcs))
            f.write(f"{ct}\t{coords}\n")

    print(f"Done. {adata.n_obs} cells written.")
    print(f"\nUnique cell types: {sorted(adata.obs['cell_type'].unique())}")


if __name__ == "__main__":
    main()
