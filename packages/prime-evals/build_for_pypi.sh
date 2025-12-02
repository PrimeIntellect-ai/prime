#!/bin/bash
set -e

echo "Building prime-evals for PyPI..."
uv build --wheel --out-dir dist

echo "Build complete!"
echo "Built files:"
ls -la dist/
