from pathlib import Path

def init_sandbox_environment(name: str, path: str) -> Path:
    env_id_underscore = name.replace("-", "_")
    local_dir = Path(path) / env_id_underscore
    local_dir.mkdir(parents=True, exist_ok=True)

    sandbox_dir = local_dir / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    pyproject_content = f'''[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["verifiers"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["{env_id_underscore}.py", "pyproject.toml", "sandbox/**/*"]
'''
    (local_dir / "pyproject.toml").write_text(pyproject_content)

    readme_content = f'''# {name}

A remote sandbox environment.

## Structure

- `{env_id_underscore}.py` - Environment definition using RemoteEnv
- `sandbox/setup.sh` - Setup script that runs in the sandbox

## Usage

Edit `sandbox/setup.sh` to install dependencies and start your service.
The last command in setup.sh should start your long-running process.
'''
    (local_dir / "README.md").write_text(readme_content)

    env_py_content = f'''from pathlib import Path
from verifiers.envs.experimental.remote_envs import RemoteEnv


def load_environment(**kwargs):
    return RemoteEnv(
        sandbox_path=Path(__file__).parent / "sandbox",
        **kwargs
    )
'''
    (local_dir / f"{env_id_underscore}.py").write_text(env_py_content)

    setup_sh_content = '''#!/bin/bash
set -e

echo "Setup complete. Add your start command here."
'''
    (sandbox_dir / "setup.sh").write_text(setup_sh_content)

    return local_dir


def init_ts_environment(name: str, path: str) -> Path:
    env_id_underscore = name.replace("-", "_")
    local_dir = Path(path) / env_id_underscore
    local_dir.mkdir(parents=True, exist_ok=True)

    sandbox_dir = local_dir / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    src_dir = sandbox_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    pyproject_content = f'''[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["verifiers"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["{env_id_underscore}.py", "pyproject.toml", "sandbox/**/*"]
'''
    (local_dir / "pyproject.toml").write_text(pyproject_content)

    readme_content = f'''# {name}

A TypeScript sandbox environment with REST API tool/reward discovery.

## Structure

- `{env_id_underscore}.py` - Environment definition using TypeScriptEnv
- `sandbox/setup.sh` - Installs Node.js and starts the server
- `sandbox/src/index.ts` - REST API with tool and reward endpoints

## REST API Contract

Your TypeScript server must implement:

- `GET /tools` - Returns tool definitions
- `POST /tools/:name` - Executes a tool
- `GET /rewards` - Returns reward function definitions
- `POST /rewards/:name` - Calculates a reward

## Usage

Edit `sandbox/src/index.ts` to define your tools and rewards.
'''
    (local_dir / "README.md").write_text(readme_content)

    env_py_content = f'''from pathlib import Path
from verifiers.envs.experimental.remote_envs import TypeScriptEnv


def load_environment(**kwargs):
    return TypeScriptEnv(
        sandbox_path=Path(__file__).parent / "sandbox",
        **kwargs
    )
'''
    (local_dir / f"{env_id_underscore}.py").write_text(env_py_content)

    setup_sh_content = '''#!/bin/bash
set -e

apt-get update && apt-get install -y curl unzip
curl -fsSL https://bun.sh/install | bash
export PATH="$HOME/.bun/bin:$PATH"

bun install
bun run src/index.ts
'''
    (sandbox_dir / "setup.sh").write_text(setup_sh_content)

    package_json_content = f'''{{
  "name": "{name}",
  "version": "1.0.0",
  "dependencies": {{
    "zod": "^3.23.0",
    "zod-to-json-schema": "^3.23.0"
  }}
}}
'''
    (sandbox_dir / "package.json").write_text(package_json_content)

    index_ts_content = '''import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const PORT = 3000;

// =============================================================================
// Tools - Define your tools here
// =============================================================================

const EchoArgs = z.object({
  message: z.string().describe("Message to echo back"),
});

function echo(args: z.infer<typeof EchoArgs>): string {
  return args.message;
}

const AddArgs = z.object({
  x: z.number().describe("First number"),
  y: z.number().describe("Second number"),
});

function add(args: z.infer<typeof AddArgs>): string {
  return String(args.x + args.y);
}

const tools: Record<string, { description: string; schema: z.ZodObject<any>; fn: (args: any) => string }> = {
  echo: {
    description: "Echoes back the input message",
    schema: EchoArgs,
    fn: echo,
  },
  add: {
    description: "Adds two numbers together",
    schema: AddArgs,
    fn: add,
  },
};

// =============================================================================
// Rewards - Define your reward functions here
// =============================================================================

function correctness(prompt: any, completion: any, answer: any, state: any): number {
  const lastMessage = completion[completion.length - 1];
  const content = lastMessage?.content || "";
  return content.includes(answer) ? 1.0 : 0.0;
}

const rewards: Record<string, { weight: number; fn: (prompt: any, completion: any, answer: any, state: any) => number }> = {
  correctness: {
    weight: 1.0,
    fn: correctness,
  },
};

// =============================================================================
// Server - No need to modify below
// =============================================================================

function getToolList() {
  return Object.entries(tools).map(([name, tool]) => ({
    type: "function",
    function: {
      name,
      description: tool.description,
      parameters: zodToJsonSchema(tool.schema, { $refStrategy: "none" }),
    },
  }));
}

function getRewardList() {
  return Object.entries(rewards).map(([name, reward]) => ({
    name,
    weight: reward.weight,
  }));
}

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;

    if (path === "/tools" && req.method === "GET") {
      return Response.json({ tools: getToolList() });
    }

    if (path.startsWith("/tools/") && req.method === "POST") {
      const name = path.slice("/tools/".length);
      const tool = tools[name];
      if (!tool) {
        return Response.json({ error: `Tool ${name} not found` }, { status: 404 });
      }
      const { args } = await req.json();
      const parsed = tool.schema.parse(args);
      const result = tool.fn(parsed);
      return Response.json({ result });
    }

    if (path === "/rewards" && req.method === "GET") {
      return Response.json({ rewards: getRewardList() });
    }

    if (path.startsWith("/rewards/") && req.method === "POST") {
      const name = path.slice("/rewards/".length);
      const reward = rewards[name];
      if (!reward) {
        return Response.json({ error: `Reward ${name} not found` }, { status: 404 });
      }
      const { prompt, completion, answer, state } = await req.json();
      const score = reward.fn(prompt, completion, answer, state);
      return Response.json({ score });
    }

    return Response.json({ error: "Not found" }, { status: 404 });
  },
});

console.log(`Server running on port ${PORT}`);
'''
    (src_dir / "index.ts").write_text(index_ts_content)

    return local_dir