"""The frozen v0 verifiers eval CLI argv surface, vendored for hosted evals.

Hosted-eval sandboxes still emit the classic v0 `vf-eval` argv, so prime must parse
and validate that argument surface — to build hosted requests, and to convert local
runs onto the v1 CLI. Verifiers has removed the v0 CLI itself; the frozen parser
lives here (vendored from verifiers v0.1.15 `verifiers/scripts/eval.py`) while the
config-level compat helpers live in `verifiers.v1.cli.eval.compat`.
"""

import argparse
import json
from typing import Any

from verifiers.v1.cli.eval.compat import (  # noqa: F401  (re-exported for callers)
    PROVIDER_CONFIGS,
    build_extra_headers,
    merge_sampling_args,
)

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_ENV_DIR_PATH = "./environments"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"
DEFAULT_NUM_EXAMPLES = 5
DEFAULT_ROLLOUTS_PER_EXAMPLE = 3
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_CLIENT_TYPE = "openai_chat_completions"
DEFAULT_PROVIDER = "prime"


def v0_argv_to_fields(environment: str, passthrough_args: list[str]) -> dict[str, Any]:
    """Parse a frozen-surface argv into the explicitly-provided v0 fields.

    Returns only flags the caller actually passed (defaults suppressed), plus the
    positional target under ``env_target``. Raises SystemExit on unknown flags,
    exactly like the v0 CLI did.
    """
    parser = build_parser()
    for action in parser._actions:
        if action.option_strings:
            action.default = argparse.SUPPRESS
    namespace = parser.parse_args([environment, *passthrough_args])
    fields = {k: v for k, v in vars(namespace).items() if k != "env_id_or_config"}
    fields["env_target"] = namespace.env_id_or_config
    return fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env_id_or_config",
        type=str,
        default="gsm8k",
        help="Environment module name or path to TOML config file.",
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default={},
        help='Environment module arguments as JSON object (e.g., \'{"key": "value", "num": 42}\')',
    )
    parser.add_argument(
        "--env-dir-path",
        type=str,
        default=DEFAULT_ENV_DIR_PATH,
        help="Path to environments directory",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default=None,
        choices=list(PROVIDER_CONFIGS.keys()),
        help=(
            "Inference provider shorthand. Resolves --api-base-url and --api-key-var "
            "automatically. Explicit --api-base-url / --api-key-var take precedence. "
            "Overrides endpoint registry when model is in registry. "
            "Falls back to 'prime' when model is not in registry."
        ),
    )
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default=DEFAULT_ENDPOINTS_PATH,
        help="Path to API endpoints TOML registry",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help="Name of model to evaluate",
    )
    parser.add_argument(
        "--api-client-type",
        type=str,
        default=None,
        help=(
            "Which client type to use ('openai_completions', 'openai_chat_completions', "
            "'openai_chat_completions_token', 'openai_responses', 'renderer', "
            "'anthropic_messages', 'nemorl_chat_completions')"
        ),
        choices=[
            "openai_completions",
            "openai_chat_completions",
            "openai_chat_completions_token",
            "openai_responses",
            "renderer",
            "anthropic_messages",
            "nemorl_chat_completions",
        ],
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default=None,
        help="Environment variable name for API key (overrides --provider)",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default=None,
        help="Base URL for API (overrides --provider)",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=None,
        help="Extra HTTP header to pass to inference API. 'Name: Value'. Repeatable.",
    )
    parser.add_argument(
        "--header-from-state",
        action="append",
        default=None,
        help=(
            "Per-request HTTP header whose value is read from the rollout state. "
            "'Name: state_key' (e.g. 'X-Session-ID: trajectory_id'). Repeatable. "
            "Defaults to X-Session-ID=trajectory_id if unset."
        ),
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=None,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=None,
        help="Number of rollouts per example",
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="Shuffle the evaluation dataset before selecting examples",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Seed for --shuffle. Defaults to 0 when --shuffle is enabled.",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (unset to use model default)",
    )
    parser.add_argument(
        "--temperature", "-T", type=float, default=None, help="Temperature for sampling"
    )
    parser.add_argument(
        "--sampling-args",
        "-S",
        type=json.loads,
        default=None,
        help=(
            "Sampling arguments as JSON object. Keys here override --max-tokens/--temperature. "
            'Example: \'{"enable_thinking": false, "max_tokens": 256}\''
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Custom output directory for evaluation results and logs",
    )
    parser.add_argument(
        "--verbose", "-v", default=False, action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--no-interleave-scoring",
        "-N",
        default=False,
        action="store_true",
        help="Disable interleaving of scoring",
    )
    parser.add_argument(
        "--state-columns",
        "-C",
        type=lambda t: [s.strip() for s in t.split(",")],
        default=[],
        help="Comma-separated list of state columns to save (e.g., 'turn,timing')",
    )
    parser.add_argument(
        "--save-results",
        "-s",
        default=False,
        action="store_true",
        help="Save results to disk",
    )
    parser.add_argument(
        "--resume",
        "-R",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help=(
            "Resume from a previous run. Optionally provide a PATH; "
            "if omitted, auto-detect the latest incomplete matching run."
        ),
    )
    parser.add_argument(
        "--independent-scoring",
        "-i",
        default=False,
        action="store_true",
        help="Score each rollout individually instead of scoring by group",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        "-H",
        default=False,
        action="store_true",
        help="Save dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf-hub-dataset-name",
        "-D",
        type=str,
        default="",
        help="Name of dataset to save to Hugging Face Hub",
    )
    parser.add_argument(
        "--extra-env-kwargs",
        "-x",
        type=json.loads,
        default={},
        help=(
            'Extra environment as JSON object (e.g., \'{"key": "value", "num": 42}\'). '
            "Passed to environment constructor."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Per-rollout wall-clock timeout in seconds. "
            "Overrides timeout_seconds in --extra-env-kwargs."
        ),
    )
    parser.add_argument(
        "--fullscreen",
        "-f",
        default=False,
        action="store_true",
        help="Use fullscreen (alternate-screen) mode for the Rich live evaluation display",
    )
    parser.add_argument(
        "--disable-tui",
        "-d",
        default=False,
        action="store_true",
        help="Disable Rich display; use normal logging and tqdm progress instead",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for transient infrastructure errors (default: 3)",
    )
    parser.add_argument(
        "--disable-env-server",
        default=False,
        action="store_true",
        help="Do not start env servers when evaluating environments",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        default="auto",
        help='Number of env server worker processes ("auto" = concurrency // 256, or an integer)',
    )
    parser.add_argument(
        "--abbreviated-summary",
        "-A",
        default=False,
        action="store_true",
        help="Abbreviated summary: show settings and stats only, skip example prompts/completions",
    )
    parser.add_argument(
        "--heartbeat-url",
        type=str,
        default=None,
        help="Heartbeat URL for uptime monitoring",
    )
    return parser
