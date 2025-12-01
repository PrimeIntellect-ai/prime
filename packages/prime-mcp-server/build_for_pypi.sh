#!/bin/bash
set -e

echo "Building prime-mcp-server for PyPI..."

echo "Cleaning previous builds..."
rm -rf dist/ build/ src/prime_mcp.egg-info/

echo "Building wheel..."
uv build --wheel --out-dir dist

echo "Build complete!"
echo "Built files:"
ls -la dist/

