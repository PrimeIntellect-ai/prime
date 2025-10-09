# Prime Intellect - CLI & SDKs

Monorepo for Prime Intellect's Python packages: CLI tools and SDKs for managing GPU compute resources, sandboxes, and inference.

## 📦 Packages

### [`prime-sandboxes`](./packages/prime-sandboxes/)
[![PyPI](https://img.shields.io/pypi/v/prime-sandboxes)](https://pypi.org/project/prime-sandboxes/)

**Lightweight standalone SDK for managing remote code execution environments (sandboxes).**

```bash
uv pip install prime-sandboxes
```

```python
from prime_sandboxes import APIClient, SandboxClient, CreateSandboxRequest

client = APIClient(api_key="...")
sandbox = SandboxClient(client)
```

**Use this if you only need sandboxes** - minimal dependencies, ~50KB installed.

[→ Full Documentation](./packages/prime-sandboxes/)

---

### [`prime`](./packages/prime/)
[![PyPI](https://img.shields.io/pypi/v/prime)](https://pypi.org/project/prime/)

**Full-featured CLI + SDK with pods, sandboxes, inference, and availability APIs.**

```bash
uv tool install prime
```

```bash
prime login
prime availability list
prime sandbox create --image python:3.11
```

**Use this for the complete experience** - CLI commands + all SDKs.

[→ Full Documentation](./packages/prime/)

---

## 🚀 Quick Start

### For Sandbox SDK Users

```python
# Lightweight - install only sandboxes SDK
uv pip install prime-sandboxes

from prime_sandboxes import APIClient, SandboxClient, CreateSandboxRequest

client = APIClient(api_key="your-api-key")
sandbox_client = SandboxClient(client)

sandbox = sandbox_client.create(CreateSandboxRequest(
    name="my-sandbox",
    docker_image="python:3.11-slim",
    cpu_cores=2,
    memory_gb=4,
))

sandbox_client.wait_for_creation(sandbox.id)
result = sandbox_client.execute_command(sandbox.id, "python --version")
print(result.stdout)
```

### For CLI Users

```bash
# Install full CLI
uv tool install prime

# Authenticate
prime login

# Manage resources
prime availability list --gpu H100
prime sandbox create --image python:3.11 --name my-sandbox
prime sandbox exec <id> "python --version"
```

## 🏗️ Repository Structure

```
prime-cli/
├── packages/
│   ├── prime-sandboxes/     # Standalone sandboxes SDK
│   └── prime/               # Full CLI + all SDKs
├── examples/                # Example scripts
├── .github/workflows/       # CI/CD
└── README.md               # This file
```

## 📚 Documentation

- [prime-sandboxes SDK Docs](./packages/prime-sandboxes/README.md)
- [prime CLI Docs](./packages/prime/README.md)
- [Examples](./examples/)

## 🤝 Contributing

Contributions welcome! This is a monorepo with independent packages.

**Development setup:**

```bash
# Clone repository
git clone https://github.com/PrimeIntellect-ai/prime-cli
cd prime-cli

# Install prime-sandboxes package
cd packages/prime-sandboxes
uv sync --all-extras
uv run pytest

# Install prime package
cd packages/prime
uv sync --all-extras
uv run pytest
```

## 📄 License

MIT License - see [LICENSE](./LICENSE) file for details.

## 🔗 Links

- [Website](https://primeintellect.ai)
- [Dashboard](https://app.primeintellect.ai)
- [API Docs](https://api.primeintellect.ai/docs)
- [Discord Community](https://discord.gg/primeintellect)
