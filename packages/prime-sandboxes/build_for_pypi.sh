#!/bin/bash
set -e

echo "Building prime-sandboxes for PyPI..."

cp pyproject.toml pyproject.toml.backup

echo "Copying prime-core into source tree..."
cp -r ../prime-core/src/prime_core src/prime_core

sed '/prime-core/d' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml
sed '/\[tool\.uv\.sources\]/,/prime-core = { workspace = true }/d' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml
sed 's|"../prime-core/src/prime_core"|"src/prime_core"|' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml

echo "Building wheel..."
uv build --wheel

rm -rf src/prime_core
mv pyproject.toml.backup pyproject.toml

echo "Build complete!"
