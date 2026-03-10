#!/usr/bin/env bash
set -e

# Build the WASM package and start the dev server.
#
# Prerequisites:
#   cargo install wasm-pack
#   Node.js (for npm/vite)

rustup target add wasm32-unknown-unknown

if cargo install --list | grep -q wasm-pack; then
  cargo install wasm-pack
fi

echo "==> Building WASM package..."
wasm-pack build crates/web --target web --out-dir ../../www/pkg

echo "==> Installing npm dependencies..."
cd www
npm install
npm run dev
