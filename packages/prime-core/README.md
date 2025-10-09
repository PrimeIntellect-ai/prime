# prime-core

**Internal development package - NOT published to PyPI**

This package contains shared HTTP client and configuration code used by both `prime-sandboxes` and `prime` packages during development.

## Purpose

- **Single source of truth** for `APIClient` and `Config` during development
- **Used via uv workspace** - packages reference this locally
- **Not published** - published packages include their own copies of this code

## Contents

- `client.py` - HTTP client with error handling
- `config.py` - Configuration management

## Development

When making changes to core functionality:
1. Edit files in `prime-core`
2. Changes are immediately available to all workspace packages
3. Before releasing, ensure changes are synced to published packages
