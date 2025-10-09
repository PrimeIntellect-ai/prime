#!/bin/bash
set -e

cleanup() {
    rm -rf src/prime_core
    if [ -f pyproject.toml.backup ]; then
        mv pyproject.toml.backup pyproject.toml
    fi
}

trap cleanup EXIT

echo "Building prime for PyPI..."

cp pyproject.toml pyproject.toml.backup

echo "Copying prime-core into source tree..."
rm -rf src/prime_core
cp -r ../prime-core/src/prime_core src/prime_core

sed '/^    "prime-core",/d' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml
sed '/^\[tool\.uv\.sources\]/,/^$/d' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml
sed 's|"../prime-core/src/prime_core"|"src/prime_core"|' pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml

echo "Building wheel..."
uv build --wheel

echo "Build complete!"
