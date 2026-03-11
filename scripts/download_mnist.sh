#!/bin/bash
# Download MNIST test set (10k images) for the web frontend.
# Files are decompressed and served as static assets by Vite from www/public/data/.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../www/public/data"
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

mkdir -p "$DATA_DIR"

echo "Downloading MNIST test set..."
curl -# -o "$DATA_DIR/t10k-images-idx3-ubyte.gz" "$BASE_URL/t10k-images-idx3-ubyte.gz"
curl -# -o "$DATA_DIR/t10k-labels-idx1-ubyte.gz" "$BASE_URL/t10k-labels-idx1-ubyte.gz"

echo "Decompressing..."
gunzip -f "$DATA_DIR/t10k-images-idx3-ubyte.gz"
gunzip -f "$DATA_DIR/t10k-labels-idx1-ubyte.gz"

echo "Done. Files saved to www/public/data/"
ls -lh "$DATA_DIR"/t10k-*
