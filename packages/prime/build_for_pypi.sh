#!/bin/bash
# Build script for creating PyPI-ready wheels
# This script bundles prime-core code and removes it from dependencies

set -e

echo "🔨 Building prime for PyPI..."

# Store original pyproject.toml
cp pyproject.toml pyproject.toml.backup

# Remove prime-core from dependencies (keep prime-sandboxes as-is)
sed '/^    "prime-core",/d' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml

# Remove the entire [tool.uv.sources] section (not needed for build, causes issues)
sed '/^\[tool\.uv\.sources\]/,/^$/d' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml

echo "📦 Building wheel (prime-core code will be bundled via force-include)..."
uv build

# Restore original pyproject.toml
mv pyproject.toml.backup pyproject.toml

echo "✅ Build complete! Wheel in dist/ includes bundled prime-core code"
echo "   Users installing from PyPI will get:"
echo "   - prime (with bundled prime-core)"
echo "   - prime-sandboxes from PyPI (as dependency)"
