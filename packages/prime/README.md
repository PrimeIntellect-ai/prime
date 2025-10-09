<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8">
    <img alt="Prime Intellect" src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8" width="312" style="max-width: 100%;">
  </picture>
</p>

---

<h3 align="center">
Prime Intellect CLI & SDKs
</h3>

---

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/prime?cacheSeconds=60)](https://pypi.org/project/prime/)
[![Python versions](https://img.shields.io/pypi/pyversions/prime?cacheSeconds=60)](https://pypi.org/project/prime/)
[![Downloads](https://img.shields.io/pypi/dm/prime)](https://pypi.org/project/prime/)

Command line interface and SDKs for managing Prime Intellect GPU resources, sandboxes, and environments.
</div>

## Installation

```bash
uv tool install prime
```

Or with pip:
```bash
pip install prime
```

## Quick Start

```bash
# Authenticate
prime login

# List available GPUs
prime availability list

# Create a pod
prime pods create

# SSH into a pod
prime pods ssh <pod-id>
```

## Full Documentation

For complete usage documentation, configuration options, and development setup, see the [main repository README](https://github.com/PrimeIntellect-ai/prime-cli).

## Links

- [Full Documentation](https://github.com/PrimeIntellect-ai/prime-cli)
- [Website](https://primeintellect.ai)
- [Dashboard](https://app.primeintellect.ai)
- [API Docs](https://api.primeintellect.ai/docs)
- [Discord Community](https://discord.gg/primeintellect)
