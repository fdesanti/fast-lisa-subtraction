#!/usr/bin/env bash
set -euo pipefail

echo "==> Building Fast Lisa Subtraction documentation"

# Paths
DOCS_DIR="docs"
SRC_DIR="src/fast_lisa_subtraction"
API_DIR="$DOCS_DIR/source/api"
BUILD_DIR="$DOCS_DIR/build"

echo "==> Ensuring Sphinx directories exist"
mkdir -p docs/source/_static
mkdir -p docs/source/_templates
mkdir -p docs/build

echo "==> Cleaning old builds"
rm -rf "$BUILD_DIR" "$API_DIR"/*

echo "==> Recreating API directory"
mkdir -p "$API_DIR"

echo "==> Generating API stubs with sphinx-apidoc"
sphinx-apidoc -o "$API_DIR" "$SRC_DIR" --separate

echo "==> Organizing API structure and indexes"
python "$DOCS_DIR/generate_api_indexes.py"

echo "==> Building HTML documentation"
sphinx-build --keep-going -b html "$DOCS_DIR/source" "$BUILD_DIR/html"

echo "==> DONE ✅"
echo "HTML available at: $BUILD_DIR/html/index.html"

echo "To visualize locally..."
echo "cd docs/build/html; python -m http.server 8000"
