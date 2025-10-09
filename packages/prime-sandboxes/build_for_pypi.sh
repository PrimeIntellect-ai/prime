#!/bin/bash
# Build script for creating PyPI-ready wheels
# This script bundles prime-core code and removes it from dependencies

set -e

echo "🔨 Building prime-sandboxes for PyPI..."

# Store original pyproject.toml
cp pyproject.toml pyproject.toml.backup

# Remove prime-core from dependencies using sed
# This creates a version without the workspace dependency
sed '/prime-core/d' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml

# Remove the workspace source declaration
sed '/\[tool\.uv\.sources\]/,/prime-core = { workspace = true }/d' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml

echo "📦 Building wheel (prime-core code will be bundled via force-include)..."
uv build

# Restore original pyproject.toml
mv pyproject.toml.backup pyproject.toml

echo "✅ Build complete! Wheel in dist/ includes bundled prime-core code"
echo "   Users installing from PyPI will get a self-contained package"
