"""Hosted Training commands."""

import json
import os
import re
import time
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import toml
import typer
from click.exceptions import Abort
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError
from rich.markup import escape as rich_escape
from rich.table import Table

from prime_cli.core import Config

from ..api.rl import EnvServerInfo, RLClient, RLRun
from ..client import APIClient, APIError, ValidationError
from ..utils import (
    DefaultCommandGroup,
    PlainTyper,
    get_console,
    json_help,
    json_output_help,
    output_data_as_json,
    validate_output_format,
)
from ..utils.env_metadata import find_environment_metadata
from ..utils.env_vars import EnvParseError, collect_env_vars
from ..utils.formatters import (
    format_file_size,
    format_promo_price,
    strip_ansi,
)
from ..utils.projects import resolve_project_id
from ..utils.prompt import confirm_or_skip
from .feedback import submit_feedback
from .usage import RUN_USAGE_JSON_HELP, run_usage_command

console = get_console()

RL_RUN_JSON_HELP = json_output_help(
    ".run = {id, name?, status, base_model, environments[], "
    "rollouts_per_example, max_steps, batch_size, created_at, updated_at, ...}",
)

RL_MODELS_JSON_HELP = json_output_help(
    ".models[] = {name, at_capacity, training_price_per_mtok, "
    "inference_input_price_per_mtok, inference_output_price_per_mtok, "
    "effective_training_price_per_mtok?, "
    "effective_inference_input_price_per_mtok?, "
    "effective_inference_output_price_per_mtok?, promo_label?}",
)

RL_LIST_JSON_HELP = json_output_help(
    ".runs[] = {id, name?, status, base_model, environments[], max_steps, "
    "batch_size, created_at, updated_at, ...}",
    ".total = number",
    ".page = number",
    ".per_page = number",
)

RL_METRICS_JSON_HELP = json_help(
    ".metrics[] = metric record from the Hosted Training API",
)

RL_ROLLOUTS_JSON_HELP = json_help(
    ". = {run_id, samples[], total, page, limit, total_pages}",
)

RL_PROGRESS_JSON_HELP = json_help(
    ". = {latest_step, steps_with_samples[], steps_with_distributions[], last_updated_at}",
)

RL_DISTRIBUTIONS_JSON_HELP = json_help(
    ". = {bins[], step}",
)

RL_CHECKPOINTS_JSON_HELP = json_output_help(
    ".checkpoints[] = {id, rft_run_id, step, storage_url, status, "
    "size_bytes?, created_at, uploaded_at?}",
)


# Progress bar pattern (tqdm-style progress bars)
PROGRESS_BAR = re.compile(r".*\|[█▏▎▍▌▋▊▉ ]{10,}\|.*")

HOSTED_TRAINING_LOG_STARTUP_POLL_SECONDS = 10
HOSTED_TRAINING_LOG_STARTUP_POLLS = 18
HOSTED_TRAINING_LOG_FOLLOW_POLL_SECONDS = 5

# Log level colors for rich console
LEVEL_STYLES = {
    "DEBUG": "dim",
    "INFO": "blue",
    "WARNING": "yellow",
    "WARN": "yellow",
    "ERROR": "red",
    "CRITICAL": "red bold",
    "SUCCESS": "green",
}


# Sentinel to indicate "this is JSON but should be skipped"
_SKIP_LINE = "__SKIP__"

_MODEL_PARAM_COUNT_PATTERN = re.compile(
    r"-(?P<total>\d+(?:\.\d)?)[bB](?:-[aA](?P<active>\d+(?:\.\d)?)[bB])?(?=$|-)"
)


def _model_name_sort_key(name: str) -> tuple[tuple[Any, ...], ...]:
    """Sort model names lexically, with recognized billion-parameter counts numeric."""
    segments: list[tuple[Any, ...]] = []
    start = 0

    for match in _MODEL_PARAM_COUNT_PATTERN.finditer(name):
        if match.start() > start:
            segments.append((0, name[start : match.start()]))

        total_params = Decimal(match.group("total"))
        active_params = Decimal(match.group("active") or match.group("total"))
        segments.append((1, total_params, active_params))
        start = match.end()

    if start < len(name):
        segments.append((0, name[start:]))

    return tuple(segments)


def format_json_log_line(line: str) -> str | None:
    """Parse a JSON log line and format it for CLI display.

    Returns:
        - Formatted string for displayable JSON logs
        - _SKIP_LINE sentinel for JSON logs that should be filtered (e.g., progress)
        - None if not a valid JSON log (falls back to legacy handling)
    """
    trimmed = line.strip()
    if not trimmed.startswith("{") or not trimmed.endswith("}"):
        return None

    try:
        entry = json.loads(trimmed)
        if not isinstance(entry, dict):
            return None
        if "timestamp" not in entry or "level" not in entry:
            return None

        # Skip progress logs in CLI - they're too spammy
        if entry.get("type") == "progress":
            return _SKIP_LINE

        # Format timestamp (extract time portion)
        timestamp = entry.get("timestamp", "")
        if "T" in timestamp:
            time_part = timestamp.split("T")[1][:8]  # HH:MM:SS
        else:
            time_part = timestamp[:8]

        level = str(entry.get("level", "INFO")).upper()
        message = rich_escape(str(entry.get("message", "")))

        # Get style for level
        style = LEVEL_STYLES.get(level, "")

        if style:
            return f"[dim]{time_part}[/dim] [{style}]\\[{level}][/{style}] {message}"
        else:
            return f"[dim]{time_part}[/dim] \\[{level}] {message}"

    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def clean_logs(text: str) -> list[str]:
    """Clean and format logs, detecting JSON format automatically.

    Returns list of formatted lines.
    """
    lines = text.splitlines()
    formatted_lines = []

    for line in lines:
        if not line.strip():
            continue

        # Try to parse as JSON log
        formatted = format_json_log_line(line)
        if formatted == _SKIP_LINE:
            # Valid JSON but should be skipped (e.g., progress logs)
            continue
        elif formatted is not None:
            formatted_lines.append(formatted)
        else:
            # Fall back to legacy handling
            cleaned = strip_ansi(line)
            # Skip tqdm-style progress bars
            if PROGRESS_BAR.search(cleaned) or re.search(r"\d+%\|", cleaned):
                if "100%" not in cleaned:
                    continue
            if cleaned.strip():
                formatted_lines.append(cleaned)

    return formatted_lines


def generate_rl_config_template(environment: str | None = None) -> str:
    """Generate a TOML config template for Hosted Training."""
    env_value = environment or "primeintellect/reverse-text"

    return f'''\
model = "Qwen/Qwen3.5-0.8B"
loss = "rl" # "rl" | "sft"; OPD is not yet supported on hosted runtimes
max_steps = 100

# env_files = ["secrets.env"] # optional file(s) for secrets

# Training
batch_size = 128
rollouts_per_example = 8
# max_inflight_rollouts = 96 # optional hard cap for in-flight rollout capacity
# learning_rate = 3e-5 # optional; default is 1e-4

# Optional: warm-start from an existing checkpoint
# checkpoint_id = "..."

# Optional: SFT distillation teacher
# To use SFT, change loss to "sft" and uncomment this block. Defaults to Prime Inference.
# [teacher]
# model = "openai/gpt-oss-120b"
#
# [teacher.sampling]
# max_tokens = 2048
# reasoning_effort = "medium"

# Optional: override the Prime Inference default with a custom OAI-compatible endpoint.
# Uncomment this block only when pointing at OpenAI, OpenRouter, self-hosted vLLM, etc.
# [teacher.client]
# base_url = "https://api.openai.com/v1"
# api_key_var = "OPENAI_API_KEY"

[sampling]
max_tokens = 2048
# temperature = 0.7

# Optional: Hosted Training reasoning controls (mutually exclusive)
# enable_thinking = false    # supported models: Qwen3.5, Nemotron
# reasoning_effort = "high"  # supported models: GPT-OSS ("low" | "medium" | "high")

[[env]]
id = "{env_value}"

# [[env]] # add multiple [[env]] sections for multi-env training
# id = "primeintellect/another-env"
# args = {{ split = "train", max_examples = 1000 }}

# Optional: online evaluation
# [eval]
# interval = 100
# # optional: default for all environments
# num_examples = -1
# rollouts_per_example = 1
# skip_first_step = false # set true to skip the pre-training eval of the base model
#
# [[eval.env]]
# id = "primeintellect/eval-env"
# args = {{ split = "test" }}
# # environment-specific overrides
# num_examples = 30
# rollouts_per_example = 4

# Optional: validation during training
# [val]
# num_examples = 64
# rollouts_per_example = 1
# interval = 5

# Optional: rollout filters (replaces the removed difficulty buffer).
# Setting a slot overrides prime-rl's default filter list for that slot.
# [[pre_batch_filters]]
# type = "zero_advantage" # "gibberish" | "repetition" | "zero_advantage"
# enforce = true          # drop flagged rollouts before they fill a batch slot

# Optional: checkpoint configuration
# [checkpoints]
# interval = 100              # Save checkpoint every N steps
# keep_cloud = 5              # Keep N checkpoints in cloud (-1 = keep all)

# Optional: adapter upload configuration
# [adapters]
# interval = 0                # Upload adapter every N steps (0 = only at run end)
# keep_last = 3               # Keep N adapters in cloud (-1 = keep all)
'''


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    args: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int | None = Field(default=None, ge=0)
    version: str | None = None

    @model_validator(mode="after")
    def parse_version_from_id(self) -> "EnvConfig":
        """Extract version from id if specified as 'owner/name@version'."""
        if "@" in self.id:
            id_part, version_part = self.id.rsplit("@", 1)
            self.id = id_part
            if self.version is None and version_part:
                self.version = version_part
        return self

    def to_api_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"id": self.id}
        if self.name is not None:
            result["name"] = self.name
        if self.args:
            result["args"] = self.args
        if self.max_retries is not None:
            result["max_retries"] = self.max_retries
        if self.version is not None:
            result["version"] = self.version
        return result


class EvalEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    args: Dict[str, Any] = Field(default_factory=dict)
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    max_retries: int | None = Field(default=None, ge=0)
    version: str | None = None

    @model_validator(mode="after")
    def parse_version_from_id(self) -> "EvalEnvConfig":
        """Extract version from id if specified as 'owner/name@version'."""
        if "@" in self.id:
            id_part, version_part = self.id.rsplit("@", 1)
            self.id = id_part
            if self.version is None and version_part:
                self.version = version_part
        return self

    def to_api_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"id": self.id}
        if self.name is not None:
            result["name"] = self.name
        if self.args:
            result["args"] = self.args
        if self.num_examples is not None:
            result["num_examples"] = self.num_examples
        if self.rollouts_per_example is not None:
            result["rollouts_per_example"] = self.rollouts_per_example
        if self.max_retries is not None:
            result["max_retries"] = self.max_retries
        if self.version is not None:
            result["version"] = self.version
        return result


class TemperatureSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "linear"  # "linear" or "cosine"
    start_temperature: float
    end_temperature: float
    total_steps: int | None = None


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = None
    temperature: float | None = None
    temp_scheduler: TemperatureSchedulerConfig | None = None
    extra_body: Dict[str, Any] | None = None
    enable_thinking: bool | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _reasoning_controls_mutually_exclusive(self) -> "SamplingConfig":
        if self.enable_thinking is not None and self.reasoning_effort is not None:
            raise ValueError("enable_thinking and reasoning_effort cannot both be set")
        return self


class TeacherSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = None
    temperature: float | None = None
    extra_body: Dict[str, Any] | None = None
    enable_thinking: bool | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _reasoning_controls_mutually_exclusive(self) -> "TeacherSamplingConfig":
        if self.enable_thinking is not None and self.reasoning_effort is not None:
            raise ValueError("enable_thinking and reasoning_effort cannot both be set")
        return self


class TeacherClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., min_length=1)
    api_key_var: str = Field(..., min_length=1)
    headers_from_env: Dict[str, str] | None = None
    skip_model_check: bool | None = None


class TeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    client: TeacherClientConfig | None = None
    sampling: TeacherSamplingConfig | None = None

    def to_api_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"model": {"name": self.model}}
        if self.client is not None:
            result["client"] = self.client.model_dump(exclude_none=True)
        if self.sampling is not None:
            result["sampling"] = self.sampling.model_dump(exclude_none=True)
        return result


class EvalSamplingConfig(BaseModel):
    """Eval-time sampling overrides."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = None
    temperature: float | None = None
    extra_body: Dict[str, Any] | None = None
    enable_thinking: bool | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _reasoning_controls_mutually_exclusive(self) -> "EvalSamplingConfig":
        if self.enable_thinking is not None and self.reasoning_effort is not None:
            raise ValueError("enable_thinking and reasoning_effort cannot both be set")
        return self


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interval: int | None = None
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    skip_first_step: bool | None = None
    """Skip the eval of the base model that otherwise runs before training
    starts. Replaces the deprecated ``eval_base_model`` with inverted meaning
    (``eval_base_model = false`` ≡ ``skip_first_step = true``), matching the
    prime-rl orch v2 rename."""
    env: List[EvalEnvConfig] = Field(default_factory=list)
    sampling: EvalSamplingConfig | None = None

    def to_api_dict(self) -> Dict[str, Any] | None:
        if not self.env:
            return None
        result: Dict[str, Any] = {"environments": [e.to_api_dict() for e in self.env]}
        if self.interval is not None:
            result["interval"] = self.interval
        if self.num_examples is not None:
            result["num_examples"] = self.num_examples
        if self.rollouts_per_example is not None:
            result["rollouts_per_example"] = self.rollouts_per_example
        if self.skip_first_step is not None:
            result["skip_first_step"] = self.skip_first_step
        if self.sampling is not None:
            sampling_dict = self.sampling.model_dump(exclude_none=True)
            if sampling_dict:
                result["sampling"] = sampling_dict
        return result


class ValConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_examples: int | None = None
    rollouts_per_example: int | None = None
    interval: int | None = None

    def to_api_dict(self) -> Dict[str, Any] | None:
        result: Dict[str, Any] = {}
        if self.num_examples is not None:
            result["num_examples"] = self.num_examples
        if self.rollouts_per_example is not None:
            result["rollouts_per_example"] = self.rollouts_per_example
        if self.interval is not None:
            result["interval"] = self.interval
        return result if result else None


class BatchFilterConfig(BaseModel):
    """A single prime-rl rollout filter (one ``[[pre_batch_filters]]`` /
    ``[[post_batch_filters]]`` table).

    Extra keys pass through verbatim so type-specific knobs (e.g.
    repetition's ``window``) reach the trainer without a CLI release.
    Enforcing ``zero_advantage`` pre-batch is the orch v2 replacement for
    the removed ``buffer.online_difficulty_filtering``.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    enforce: bool | None = None


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str | None = None
    project: str | None = None
    name: str | None = None


class CheckpointsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interval: int | None = None  # Save checkpoint every N steps
    keep_cloud: int | None = None  # Keep N checkpoints in cloud (-1 = keep all)

    def to_api_dict(self) -> Dict[str, Any] | None:
        result: Dict[str, Any] = {}
        if self.interval is not None:
            result["interval"] = self.interval
        if self.keep_cloud is not None:
            result["keep_cloud"] = self.keep_cloud
        return result if result else None


class AdaptersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interval: int | None = None  # Upload adapter every N steps (0 = only at run end)
    keep_last: int | None = None  # Keep N adapters in cloud (-1 = keep all)

    def to_api_dict(self) -> Dict[str, Any] | None:
        result: Dict[str, Any] = {}
        if self.interval is not None:
            result["interval"] = self.interval
        if self.keep_last is not None:
            result["keep_last"] = self.keep_last
        return result if result else None


class InfrastructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compute_size: str | None = None

    def to_api_dict(self) -> Dict[str, Any] | None:
        d: Dict[str, Any] = {}
        if self.compute_size is not None:
            d["compute_size"] = self.compute_size
        return d if d else None


class TailscaleConfig(BaseModel):
    """Optional per-run tailscale sidecar (enterprise-only feature).

    When enabled, every env-server (training + eval) for this run joins the
    user's tailnet via a sidecar. The orchestrator and other runs are
    untouched.

    The auth key must be supplied via the ``auth_key`` field or, preferably,
    via the ``TAILSCALE_AUTH_KEY`` environment variable so the secret never
    has to live in ``rl.toml``. Only pre-authenticated keys (``tskey-auth-``)
    are accepted; OAuth client secrets are not supported. Free-form
    ``tailscale up`` flags are also not accepted — the platform always
    boots the sidecar with a locked-down arg set to keep tenant isolation.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    auth_key: str | None = None
    hostname_prefix: str = "prime-hosted-training"

    @field_validator("hostname_prefix")
    @classmethod
    def validate_hostname_prefix(cls, v: str) -> str:
        # Must end in alphanumeric — otherwise the derived sidecar hostname
        # (e.g. f"{prefix}-env-{idx}-{run_id}") would contain consecutive
        # hyphens which Tailscale rejects. Length cap matches the platform
        # (backend/app/packages/rft/k8s_resources/tailscale.py).
        if not re.fullmatch(r"[a-z]([a-z0-9-]{0,28}[a-z0-9])?", v):
            raise ValueError(
                "hostname_prefix must be 1-30 chars, lowercase alphanumeric or "
                "hyphens, starting with a letter and ending with a letter or digit"
            )
        return v

    @field_validator("auth_key")
    @classmethod
    def validate_auth_key(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.startswith("tskey-auth-"):
            raise ValueError(
                "auth_key must be a Tailscale pre-authenticated auth key "
                "(starting with 'tskey-auth-')"
            )
        return v

    @model_validator(mode="after")
    def fill_auth_key_from_env_and_validate(self) -> "TailscaleConfig":
        if not self.enabled:
            return self
        if self.auth_key is None:
            env_value = os.environ.get("TAILSCALE_AUTH_KEY")
            if env_value:
                if not env_value.startswith("tskey-auth-"):
                    raise ValueError("TAILSCALE_AUTH_KEY must start with 'tskey-auth-'")
                self.auth_key = env_value
        if not self.auth_key:
            raise ValueError(
                "auth_key is required when tailscale.enabled is true. "
                "Set [tailscale] auth_key in your config or export TAILSCALE_AUTH_KEY."
            )
        return self

    def to_api_dict(self) -> Dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "enabled": True,
            "auth_key": self.auth_key,
            "hostname_prefix": self.hostname_prefix,
        }


class RLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    model: str
    loss: Literal["rl", "sft", "opd"] = "rl"
    teacher: TeacherConfig | None = None
    max_steps: int = 100
    batch_size: int = 128
    rollouts_per_example: int = 8
    max_inflight_rollouts: int | None = Field(default=None, ge=1)
    learning_rate: float | None = None
    lora_alpha: int | None = None
    oversampling_factor: float | None = Field(default=None, gt=0)
    checkpoint_id: str | None = None  # Warm-start from an existing checkpoint
    cluster_name: str | None = None  # Admin-only: target a specific cluster by name
    env: List[EnvConfig] = Field(default_factory=list)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    val: ValConfig = Field(default_factory=ValConfig)
    pre_batch_filters: List[BatchFilterConfig] | None = None
    post_batch_filters: List[BatchFilterConfig] | None = None
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    adapters: AdaptersConfig = Field(default_factory=AdaptersConfig)
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    tailscale: TailscaleConfig = Field(default_factory=TailscaleConfig)
    run_config: Dict[str, Any] = Field(default_factory=dict)
    env_file: List[str] = Field(default_factory=list)  # deprecated, use env_files
    env_files: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "RLConfig":
        if self.max_inflight_rollouts is not None and self.oversampling_factor is not None:
            raise ValueError("Only one of max_inflight_rollouts and oversampling_factor can be set")
        if (
            self.max_inflight_rollouts is not None
            and self.max_inflight_rollouts < self.rollouts_per_example
        ):
            raise ValueError("max_inflight_rollouts must be at least rollouts_per_example")
        if self.loss == "rl" and self.teacher is not None:
            raise ValueError("teacher can only be set when loss is 'sft' or 'opd'")
        if self.loss == "sft" and self.teacher is None:
            raise ValueError("teacher is required when loss is 'sft'")
        if self.loss == "opd":
            raise ValueError(
                "loss='opd' is not supported for hosted runs yet; OPD requires "
                "teacher logprob scoring support in the hosted runtime"
            )
        return self


def _format_validation_errors(errors: list[Any]) -> list[str]:
    """Format Pydantic validation errors into user-friendly messages."""
    messages = []
    for error in errors:
        loc = ".".join(str(x) for x in error["loc"])
        msg = error["msg"]
        # Clean up common Pydantic message prefixes
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        messages.append(f"{loc}: {msg}")
    return messages


def _remove_deprecated_config_keys(data: Dict[str, Any]) -> None:
    """Remove deprecated config keys while warning users."""
    removed = False
    for key in ("trajectory_strategy", "trajectoryStrategy"):
        if key in data:
            data.pop(key, None)
            removed = True

    if removed:
        console.print("[yellow]Warning:[/yellow] `trajectory_strategy` is deprecated and ignored.")

    if "buffer" in data:
        data.pop("buffer", None)
        console.print(
            "[yellow]Warning:[/yellow] `[buffer]` is deprecated and ignored: "
            "the difficulty-filtering buffer was removed from the trainer."
        )

    eval_section = data.get("eval")
    if isinstance(eval_section, dict) and "eval_base_model" in eval_section:
        legacy = eval_section.pop("eval_base_model")
        if "skip_first_step" in eval_section:
            console.print(
                "[yellow]Warning:[/yellow] `eval.eval_base_model` is deprecated and "
                "ignored because `eval.skip_first_step` is also set. Remove "
                "`eval_base_model` from your config."
            )
        else:
            translated = (not legacy) if isinstance(legacy, bool) else legacy
            eval_section["skip_first_step"] = translated
            console.print(
                "[yellow]Warning:[/yellow] `eval.eval_base_model` is deprecated: "
                "prime-rl renamed it to `skip_first_step` with inverted meaning "
                "(`eval_base_model = false` ≡ `skip_first_step = true`). "
                f"Translating to `skip_first_step = {str(translated).lower()}` — "
                "update your config to use `skip_first_step` directly."
            )


def _peek_toml(config_path: str) -> Dict[str, Any]:
    """Parse the TOML once for the dispatch decision. Returns {} on missing
    or malformed file — the strict load_config call below produces the
    user-facing error in that case so we don't double-report here."""
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        data = toml.load(p)
    except toml.TomlDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _is_full_finetune(cfg: Dict[str, Any]) -> bool:
    """A config is full-FT iff it carries one of:

    - top-level `type = "full_finetune"` (mirrors prime-rl's discriminator)
    - a `[deployment]` table with a sizing field — either single-node
      (`num_train_gpus` / `num_infer_gpus`) or multi-node
      (`num_train_nodes` / `num_infer_nodes`) — or an explicit
      `[deployment].type` of `single_node` / `multi_node`.

    Previously only the single-node sizing fields triggered detection, so
    canonical multi-node prime-rl shapes (e.g. `qwen30b_math/rl.toml`,
    which uses `num_train_nodes` / `num_infer_nodes`) silently fell
    through to the LoRA dispatch path.
    """
    if cfg.get("type") == "full_finetune":
        return True
    deploy = cfg.get("deployment")
    if isinstance(deploy, dict):
        if deploy.get("type") in ("single_node", "multi_node"):
            return True
        sizing_keys = (
            "num_train_gpus",
            "num_infer_gpus",
            "num_train_nodes",
            "num_infer_nodes",
        )
        if any(k in deploy for k in sizing_keys):
            return True
    return False


def _dispatch_full_finetune_run(
    *,
    raw_cfg: Dict[str, Any],
    config_path: str,
    env: Optional[List[str]],
    env_file: Optional[List[str]],
    output: str,
    yes: bool,
    image_tag: Optional[str] = None,
) -> None:
    """Hand off to /api/v1/training/runs (full-FT prime-rl on a registered
    PrimeCluster). Reuses the shared env-file plumbing for WANDB / HF
    secrets so the user experience matches the LoRA path.

    Backend always auto-picks the first uncordoned PrimeCluster — the
    CLI never threads a cluster id, so a config that targets the wrong
    cluster can't be a footgun."""
    from ..api.training import HostedTrainingClient, build_payload_from_toml

    name = raw_cfg.get("name")

    # Link the dispatched run to the user's active team (same convention
    # as the LoRA path at line 1216). Without this, the RFTRun row gets
    # team_id=None and lands on the caller's personal account even when
    # the CLI is configured for a team — confusing for billing + access
    # scoping. Backend's `CreateDedicatedRunRequest.team_id` is Optional,
    # so omitting it is silently wrong rather than a 400.
    app_config = Config()
    team_id = app_config.team_id

    # Resolve env files relative to the config dir, same convention as the
    # LoRA path. We don't validate WANDB presence here — the chart wires
    # WANDB_API_KEY via secretKeyRef only if `[wandb]` is configured AND
    # the user passed --wandb-api-key, mirroring the platform's existing
    # admin-dialog UX.
    #
    # `env_file` (deprecated, singular) is loaded first so `env_files`
    # (canonical, plural) overrides it on key collision. Mirrors the
    # LoRA path at line 947 ("env_files takes precedence").
    config_dir = Path(config_path).parent
    cfg_env_files: List[str] = []
    for key in ("env_file", "env_files"):
        val = raw_cfg.get(key)
        if isinstance(val, list):
            cfg_env_files.extend(str(config_dir / p) for p in val)
        elif isinstance(val, str):
            cfg_env_files.append(str(config_dir / val))
    all_env_files = cfg_env_files + (env_file or [])

    def _warn(msg: str) -> None:
        console.print(f"[yellow]Warning:[/yellow] {msg}")

    try:
        secrets = collect_env_vars(
            env_args=env,
            env_files=all_env_files if all_env_files else None,
            on_warning=_warn,
        )
    except EnvParseError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    secrets = secrets or {}

    # The full-FT backend schema only knows about WANDB_API_KEY and
    # HF_TOKEN (see platform CreateDedicatedRunRequestBase). Silently
    # dropping anything else here would launch the run without the
    # credentials the orchestrator's env needs — the LoRA path forwards
    # the whole map via env-server pods, but full-FT runs the env inline
    # in the orchestrator and has no comparable surface today. Reject
    # loudly instead so the user knows up-front rather than discovering
    # via a runtime auth error inside a $$$ hosted pod.
    supported_secret_keys = {"WANDB_API_KEY", "HF_TOKEN"}
    unsupported = sorted(k for k in secrets if k not in supported_secret_keys)
    if unsupported:
        console.print(
            "[red]Error:[/red] full-FT runs only forward "
            f"{', '.join(sorted(supported_secret_keys))} today; got "
            f"unsupported secret(s): {', '.join(unsupported)}."
        )
        console.print(
            "[dim]Drop these from --env-var/--env-file, or wait for "
            "generic secret forwarding on the dedicated training endpoint.[/dim]"
        )
        raise typer.Exit(1)

    # `image_tag` is chart-level — it picks which prime-rl container build
    # the run pulls. Two ways to set it: `--image-tag` (CLI flag, useful
    # for ad-hoc pins / CI overrides) or top-level `image_tag = "..."` in
    # the TOML (lives with the run config, the form most teams want to
    # check in). CLI flag wins so an explicit override on the command line
    # always beats the file. Backend defaults to "main" when neither is
    # set, so falling through to None is the right "use platform default"
    # signal.
    config_image_tag = raw_cfg.get("image_tag")
    if config_image_tag is not None and not isinstance(config_image_tag, str):
        console.print(
            f"[red]Error:[/red] image_tag in {config_path} must be a string, "
            f"got {type(config_image_tag).__name__}."
        )
        raise typer.Exit(1)
    resolved_image_tag = image_tag or config_image_tag

    # Same deprecation pass as the LoRA path. In the prime-rl-native shape
    # the deprecated keys live one level down, under `[orchestrator]` — and
    # this config ships verbatim to the dedicated training endpoint, where
    # orch v2 rejects the removed keys as "Extra inputs are not permitted".
    orch_section = raw_cfg.get("orchestrator")
    if isinstance(orch_section, dict):
        _remove_deprecated_config_keys(orch_section)

    # Strip CLI-only secret-loading keys before shipping the TOML to the
    # backend. `env_file` / `env_files` only meaningfully exist on the
    # caller's filesystem — the hosted pod doesn't see them, and prime-rl
    # has no use for the paths once we've already materialized the secrets
    # into the per-run k8s Secret above. Leaving them in the config either
    # confuses prime-rl (unknown field) or silently sends it chasing
    # phantom paths. `image_tag` is similarly chart-level — the backend
    # parses it off the request body, not the embedded prime-rl config.
    config_payload = {
        k: v for k, v in raw_cfg.items() if k not in ("env_file", "env_files", "image_tag")
    }

    payload = build_payload_from_toml(
        config_payload,
        name=name,
        team_id=team_id,
        image_tag=resolved_image_tag,
        wandb_api_key=secrets.get("WANDB_API_KEY"),
        hf_token=secrets.get("HF_TOKEN"),
    )

    # `--output json` is a formatting switch: still dispatch the run,
    # then print the result as JSON. Same contract as the LoRA path
    # ("create then format"), which automation relies on to parse back
    # run_id from the response. Confirmation gating is purely on `--yes`
    # so the JSON output path doesn't accidentally launch an expensive
    # training job without an explicit ack — matches confirm_or_skip
    # in the LoRA path.
    if not confirm_or_skip("Launch this Hosted Training run?", yes, default=True):
        console.print("\nRun cancelled")
        raise typer.Exit(0)

    api_client = APIClient()
    client = HostedTrainingClient(api_client)
    # Skip the spinner for --output json: PrimeConsole.status() falls back
    # to a plain print in --plain mode, which would emit "Creating Hosted
    # Training run..." on stdout ahead of the JSON payload and break
    # automation parsing of run_id.
    status_ctx = (
        console.status("[bold blue]Creating Hosted Training run...", spinner="dots")
        if output != "json"
        else nullcontext()
    )
    try:
        with status_ctx:
            result = client.create_run(payload)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if output == "json":
        # Don't expose token_value in JSON output either — the chart
        # binds it via secretKeyRef and printing it leaks credentials
        # into automation logs.
        output_data_as_json(
            {"run": {"runId": result.run_id}},
            console,
        )
        return

    # Don't print result.token_value: the platform wires it into the
    # per-run k8s Secret automatically (orchestrator's PRIME_API_KEY env
    # via secretKeyRef). Surfacing it on stdout makes it easy to leak into
    # shared shell history/CI logs without buying anything for the user.
    console.print(f"[green]Dispatched[/green] hosted run [bold]{result.run_id}[/bold]")


def load_config(path: str) -> RLConfig:
    """Load config from TOML file."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] Config file not found: {path}")
        raise typer.Exit(1)
    try:
        data = toml.load(p)
    except toml.TomlDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid TOML in {path}: {e}")
        raise typer.Exit(1)

    if isinstance(data, dict):
        _remove_deprecated_config_keys(data)

    try:
        return RLConfig.model_validate(data)
    except PydanticValidationError as e:
        console.print(f"[red]Error:[/red] Invalid config in {path}:\n")
        for msg in _format_validation_errors(e.errors()):
            console.print(f"  [red]•[/red] {msg}")
        console.print()
        raise typer.Exit(1)


# Status color mapping
RUN_STATUS_COLORS = {
    "QUEUED": "white",
    "PENDING": "yellow",
    "RUNNING": "green",
    "COMPLETED": "cyan",
    "FAILED": "red",
    "STOPPED": "magenta",
}


def _get_status_color(status: str) -> str:
    return RUN_STATUS_COLORS.get(status.upper(), "white")


def _render_failure_analysis(analysis: Dict[str, Any]) -> None:
    """Render the automated failure-classifier output below run details.

    The backend only returns this for RUN_ISSUE classifications, so we don't
    surface the internal classifier tag — just the diagnosis itself.
    """
    root_cause = analysis.get("rootCause")
    evidence = analysis.get("evidence")
    if not root_cause and not (isinstance(evidence, list) and evidence):
        return

    header = "\n[bold]Failure Analysis[/bold]"
    confidence = analysis.get("confidence")
    if isinstance(confidence, (int, float)):
        header += f" [dim](confidence {confidence * 100:.0f}%)[/dim]"
    console.print(header)

    if root_cause:
        console.print(f"  Root Cause: {rich_escape(str(root_cause))}")

    if isinstance(evidence, list) and evidence:
        console.print("  Evidence:")
        for line in evidence:
            console.print(f"    - [dim]{rich_escape(str(line))}[/dim]")


def _format_run_for_display(run: RLRun) -> Dict[str, Any]:
    created_at = run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else ""
    env_names = [
        env.get("slug") or env.get("name") or env.get("id") or "?" for env in run.environments
    ]
    envs_display = ", ".join(env_names[:3])
    if len(env_names) > 3:
        envs_display += f" (+{len(env_names) - 3})"

    return {
        "id": run.id,
        "status": run.status,
        "model": run.base_model or "-",
        "environments": envs_display,
        "steps": "-" if run.max_steps is None else f"{run.max_steps}",
        "rollouts": "-" if run.rollouts_per_example is None else str(run.rollouts_per_example),
        "created_at": created_at,
        "team_id": run.team_id,
        "project_id": run.project_id,
    }


class DefaultGroup(DefaultCommandGroup):
    """Makes 'run' the default command when a config file is passed."""

    def __init__(self, *args, default_cmd_name: str = "run", **kwargs):
        super().__init__(*args, default_cmd_name=default_cmd_name, **kwargs)
        self._show_default_command_params = False

    def format_usage(self, ctx, formatter):
        formatter.write_usage(
            ctx.command_path,
            "[OPTIONS] CONFIG_PATH [ARGS]... | COMMAND [ARGS]...",
        )

    def get_params(self, ctx):
        params = super().get_params(ctx)
        if not self._show_default_command_params:
            return params

        default_command = self.commands.get(self.default_cmd_name)
        if default_command is None:
            return params

        seen = {p.name for p in default_command.params}
        return [*default_command.params, *(p for p in params if p.name not in seen)]

    def format_help(self, ctx, formatter):
        self._show_default_command_params = True
        try:
            return super().format_help(ctx, formatter)
        finally:
            self._show_default_command_params = False

    def invoke(self, ctx):
        if ctx.info_name == "rl":
            typer.echo(
                "[DEPRECATED] The 'rl' command is deprecated. Use 'prime train' instead.",
                err=True,
            )
        return super().invoke(ctx)


app = PlainTyper(
    cls=DefaultGroup,
    help=(
        "Launch and manage Hosted Training runs. Pass a config path directly to start a new run."
    ),
    no_args_is_help=True,
)


@app.command("run", rich_help_panel="Commands", hidden=True, epilog=RL_RUN_JSON_HELP)
def create_run(
    config_path: str = typer.Argument(
        ...,
        help="Path to a TOML config file to launch as a Hosted Training run.",
    ),
    env: Optional[List[str]] = typer.Option(
        None,
        "-e",
        "--env-var",
        help=(
            "Environment variable/secret to pass to the training container. "
            "Accepts: KEY=VALUE (direct value), KEY (reads from $KEY), "
            "or path/to/file.env (loads env file)."
        ),
    ),
    env_file: Optional[List[str]] = typer.Option(
        None,
        "--env-file",
        help="Path to .env file containing secrets. Supports ${VAR} expansion from local env.",
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
    skip_action_check: bool = typer.Option(
        False,
        "--skip-action-check",
        help="Skip action status check and run even if environment action failed.",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="Project ID or slug to attach to this run.",
    ),
    no_project: bool = typer.Option(
        False,
        "--no-project",
        help="Do not attach this run to a project.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    image_tag: Optional[str] = typer.Option(
        None,
        "--image-tag",
        help=(
            "prime-rl container image tag for the run (full-FT only). "
            'Falls back to a top-level `image_tag = "..."` in the TOML '
            "if unset, then to the backend default."
        ),
    ),
) -> None:
    """Launch a Hosted Training run from a config file.

    Example:

        prime train config.toml
    """
    validate_output_format(output, console)

    # Peek at the raw TOML BEFORE the strict-schema RLConfig parse so a
    # full-FT config (`type = "full_finetune"` or a `[deployment]` block
    # with num_train_gpus/num_infer_gpus) can bypass the LoRA-shared
    # validators and dispatch on the hosted full-FT endpoint instead.
    # Backwards-compatible: configs without these markers take the LoRA
    # path exactly as before.
    raw_cfg = _peek_toml(config_path)
    if _is_full_finetune(raw_cfg):
        _dispatch_full_finetune_run(
            raw_cfg=raw_cfg,
            config_path=config_path,
            env=env,
            env_file=env_file,
            output=output,
            yes=yes,
            image_tag=image_tag,
        )
        return

    console.print(f"[dim]Loading config from {config_path}[/dim]\n")
    cfg = load_config(config_path)

    # Collect secrets from all sources
    def warn(msg: str) -> None:
        console.print(f"[yellow]Warning:[/yellow] {msg}")

    # Resolve config env file paths relative to config file directory
    config_dir = Path(config_path).parent
    config_env_files = cfg.env_file + cfg.env_files  # support both, env_files takes precedence
    resolved_config_env_files = [str(config_dir / p) for p in config_env_files]

    # Merge config and CLI env files (CLI takes precedence)
    all_env_files = resolved_config_env_files + (env_file or [])

    try:
        secrets = collect_env_vars(
            env_args=env,
            env_files=all_env_files if all_env_files else None,
            on_warning=warn,
        )
    except EnvParseError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Validate WANDB_API_KEY is present when W&B monitoring is configured
    wandb_configured = cfg.wandb.entity or cfg.wandb.project
    has_wandb_key = secrets and "WANDB_API_KEY" in secrets
    if wandb_configured and not has_wandb_key:
        console.print("[red]Configuration Error:[/red]")
        console.print("  WANDB_API_KEY is required when W&B monitoring is configured.\n")
        console.print("Provide it via:")
        console.print('  - env_files in your config: env_files = ["secrets.env"]')
        console.print("  - CLI flag: --env-file secrets.env")
        console.print("  - CLI flag: -e WANDB_API_KEY=your-key")
        console.print(
            "  - Environment variable: export WANDB_API_KEY=... && prime train ... -e WANDB_API_KEY"
        )
        raise typer.Exit(1)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)
        app_config = Config()
        project_id = resolve_project_id(
            project,
            no_project=no_project,
            config=app_config,
            client=api_client,
        )

        # Kick off pricing fetch in the background so it overlaps with summary
        # rendering and the action-status checks below. Daemon thread so a slow
        # /rft/models can't outlive the command (ThreadPoolExecutor workers are
        # joined at interpreter exit, which would defeat the timeout cap below).
        import threading

        pricing_state: Dict[str, Any] = {"models": None, "error": None}
        pricing_done = threading.Event()

        def _fetch_pricing() -> None:
            try:
                pricing_state["models"] = rl_client.list_models(team_id=app_config.team_id)
            except Exception as exc:
                pricing_state["error"] = exc
            finally:
                pricing_done.set()

        threading.Thread(target=_fetch_pricing, daemon=True).start()

        # Show configuration in organized sections
        console.print("[white]Configuration:[/white]\n")

        # Model & Environment
        console.print("[cyan]Model & Environment[/cyan]")
        console.print(f"  Model:        {cfg.model}")
        console.print(f"  Loss:         {cfg.loss}")
        if cfg.teacher is not None:
            console.print(f"  Teacher:      {cfg.teacher.model}")
        console.print(f"  Environments: {', '.join(e.id for e in cfg.env)}")
        if app_config.team_id:
            console.print(f"  Team:         {app_config.team_id}")
        if project_id:
            console.print(f"  Project:      {project_id}")

        # Training
        console.print("\n[cyan]Training[/cyan]")
        console.print(f"  Max Steps:           {cfg.max_steps}")
        console.print(f"  Batch Size:          {cfg.batch_size}")
        console.print(f"  Rollouts per Example: {cfg.rollouts_per_example}")
        if cfg.max_inflight_rollouts is not None:
            console.print(f"  Max Inflight Rollouts: {cfg.max_inflight_rollouts}")
        if cfg.learning_rate is not None:
            console.print(f"  Learning Rate:       {cfg.learning_rate}")
        if cfg.lora_alpha is not None:
            console.print(f"  LoRA Alpha:          {cfg.lora_alpha}")
        if cfg.oversampling_factor is not None:
            console.print(f"  Oversampling Factor: {cfg.oversampling_factor}")
        if cfg.run_config:
            console.print(f"  Run Config:          {cfg.run_config}")

        # Sampling
        has_sampling = (
            cfg.sampling.max_tokens
            or cfg.sampling.temperature is not None
            or cfg.sampling.temp_scheduler is not None
            or cfg.sampling.extra_body is not None
            or cfg.sampling.enable_thinking is not None
            or cfg.sampling.reasoning_effort is not None
        )
        if has_sampling:
            console.print("\n[cyan]Sampling[/cyan]")
            if cfg.sampling.max_tokens:
                console.print(f"  Max Tokens:          {cfg.sampling.max_tokens}")
            if cfg.sampling.temperature is not None:
                console.print(f"  Temperature:         {cfg.sampling.temperature}")
            if cfg.sampling.temp_scheduler is not None:
                ts = cfg.sampling.temp_scheduler
                sched = f"{ts.type} ({ts.start_temperature} → {ts.end_temperature})"
                console.print(f"  Temp Scheduler:      {sched}")
            if cfg.sampling.extra_body is not None:
                console.print(f"  Extra Body:          {cfg.sampling.extra_body}")
            if cfg.sampling.enable_thinking is not None:
                console.print(f"  Enable Thinking:     {cfg.sampling.enable_thinking}")
            if cfg.sampling.reasoning_effort is not None:
                console.print(f"  Reasoning Effort:    {cfg.sampling.reasoning_effort}")

        # W&B
        if cfg.wandb.entity or cfg.wandb.project:
            console.print("\n[cyan]Weights & Biases[/cyan]")
            console.print(f"  Project: {cfg.wandb.entity or '?'}/{cfg.wandb.project or '?'}")
            if cfg.wandb.name:
                console.print(f"  Run Name: {cfg.wandb.name}")

        # Eval
        if cfg.eval.env:
            console.print("\n[cyan]Evaluation[/cyan]")
            console.print(f"  Environments: {', '.join(e.id for e in cfg.eval.env)}")
            if cfg.eval.interval:
                console.print(f"  Interval:     {cfg.eval.interval}")

        # Validation
        if cfg.val.num_examples is not None:
            console.print("\n[cyan]Validation[/cyan]")
            console.print(f"  Num Examples: {cfg.val.num_examples}")
            if cfg.val.interval:
                console.print(f"  Interval:     {cfg.val.interval}")

        # Infrastructure
        if cfg.infrastructure.compute_size:
            console.print("\n[cyan]Infrastructure[/cyan]")
            console.print(f"  Compute Size: {cfg.infrastructure.compute_size}")

        # Checkpoint warm-start
        if cfg.checkpoint_id:
            console.print(f"\n[cyan]Warm-start from checkpoint:[/cyan] {cfg.checkpoint_id}")

        # Secrets
        if secrets:
            console.print("\n[cyan]Secrets[/cyan]")
            console.print(f"  Keys: {', '.join(secrets.keys())}")

        # Pricing (best-effort — skipped silently if /rft/models is unreachable,
        # times out, or doesn't include the chosen model. 8s cap so a slow
        # endpoint can't gate the confirmation prompt.)
        if pricing_done.wait(timeout=8) and pricing_state["error"] is None:
            available = pricing_state["models"] or []
        else:
            available = []
        priced = next((m for m in available if m.name == cfg.model), None)
        if priced is not None:
            list_train, eff_train = priced.resolve_prices("training")
            list_input, eff_input = priced.resolve_prices("inference_input")
            list_output, eff_output = priced.resolve_prices("inference_output")
            console.print(
                "\n[cyan]Pricing[/cyan] [dim](per 1M tokens, charged on actual usage)[/dim]"
            )

            def _format(list_p: Any, eff_p: Any) -> str:
                charged = eff_p if eff_p is not None else list_p
                if charged is None:
                    return "-"
                if float(charged) == 0:
                    if list_p is not None and float(list_p) > float(charged):
                        return format_promo_price(list_p, eff_p) or "[bold green]Free[/bold green]"
                    return "[bold green]Free[/bold green]"
                return format_promo_price(list_p, eff_p) or "-"

            console.print(f"  Training:         {_format(list_train, eff_train)}")
            console.print(f"  Inference Input:  {_format(list_input, eff_input)}")
            console.print(f"  Inference Output: {_format(list_output, eff_output)}")
            if priced.promo_label:
                console.print(f"  [bold yellow]{rich_escape(priced.promo_label)}[/bold yellow]")

        console.print()

        # Check action status for hub environments
        hub_envs = [e for e in cfg.env if "/" in e.id]
        if hub_envs and not skip_action_check:
            console.print("[dim]Checking Environment Actions...[/dim]")
            failed_envs = []

            for env_config in hub_envs:
                env_id_base = env_config.id.split("@")[0]
                owner, name = env_id_base.split("/", 1)
                try:
                    status_resp = rl_client.get_environment_status(owner, name)
                    action = status_resp.get("action") or {}
                    action_status = action.get("status")

                    if action_status == "FAILED":
                        console.print(f"  [red]✗[/red] {env_config.id} [dim](failed)[/dim]")
                        failed_envs.append(env_config.id)
                    elif action_status == "SUCCESS":
                        console.print(f"  [green]✓[/green] {env_config.id} [dim](success)[/dim]")
                    elif action_status in ("RUNNING", "PENDING"):
                        console.print(
                            f"  [yellow]○[/yellow] {env_config.id} [dim](in progress)[/dim]"
                        )
                    else:
                        console.print(f"  [dim]-[/dim] {env_config.id} [dim](no action)[/dim]")
                except APIError:
                    console.print(f"  [dim]-[/dim] {env_config.id} [dim](could not check)[/dim]")

            if failed_envs:
                console.print("\n[red]Error: Action failed for environments:[/red]\n")
                for env_id in failed_envs:
                    env_id_base = env_id.split("@")[0]
                    owner, name = env_id_base.split("/", 1)
                    url = f"{app_config.frontend_url}/dashboard/environments/{owner}/{name}/actions"
                    console.print(f"  [red]✗[/red] {env_id}")
                    console.print(f"    [dim]Details: prime env action list {env_id_base}[/dim]")
                    console.print(f"    [dim]View at: [link={url}]{url}[/link][/dim]\n")

                console.print(
                    "[yellow]This usually means the environment doesn't compile or run, "
                    "or is using an unsupported version of verifiers, so the Hosted Training run "
                    "will fail.[/yellow]"
                )
                console.print("[dim]To proceed anyway, use --skip-action-check[/dim]")
                raise typer.Exit(1)

            console.print()

        if not confirm_or_skip("Launch this Hosted Training run?", yes, default=True):
            console.print("\nRun cancelled")
            return

        console.print("[dim]Creating Hosted Training run...[/dim]\n")

        # Create the run
        run = rl_client.create_run(
            model_name=cfg.model,
            environments=[e.to_api_dict() for e in cfg.env],
            rollouts_per_example=cfg.rollouts_per_example,
            max_steps=cfg.max_steps,
            max_tokens=cfg.sampling.max_tokens,
            temperature=cfg.sampling.temperature,
            temp_scheduler=cfg.sampling.temp_scheduler.model_dump(exclude_none=True)
            if cfg.sampling.temp_scheduler
            else None,
            extra_body=cfg.sampling.extra_body,
            batch_size=cfg.batch_size,
            name=cfg.name,
            wandb_entity=cfg.wandb.entity,
            wandb_project=cfg.wandb.project,
            wandb_run_name=cfg.wandb.name,
            secrets=secrets if secrets else None,
            team_id=app_config.team_id,
            project_id=project_id,
            eval_config=cfg.eval.to_api_dict(),
            val_config=cfg.val.to_api_dict(),
            # `is not None` rather than truthiness: an explicit `= []` means
            # "replace prime-rl's default filter list with no filters" and
            # must reach the API as an empty list, not be omitted.
            pre_batch_filters=[f.model_dump(exclude_none=True) for f in cfg.pre_batch_filters]
            if cfg.pre_batch_filters is not None
            else None,
            post_batch_filters=[f.model_dump(exclude_none=True) for f in cfg.post_batch_filters]
            if cfg.post_batch_filters is not None
            else None,
            learning_rate=cfg.learning_rate,
            lora_alpha=cfg.lora_alpha,
            max_inflight_rollouts=cfg.max_inflight_rollouts,
            oversampling_factor=cfg.oversampling_factor,
            checkpoints_config=cfg.checkpoints.to_api_dict(),
            adapters_config=cfg.adapters.to_api_dict(),
            checkpoint_id=cfg.checkpoint_id,
            cluster_name=cfg.cluster_name,
            infrastructure_config=cfg.infrastructure.to_api_dict(),
            tailscale_config=cfg.tailscale.to_api_dict(),
            enable_thinking=cfg.sampling.enable_thinking,
            reasoning_effort=cfg.sampling.reasoning_effort,
            run_config=cfg.run_config if cfg.run_config else None,
            loss=cfg.loss,
            teacher=cfg.teacher.to_api_dict() if cfg.teacher else None,
        )

        if output == "json":
            output_data_as_json({"run": run.model_dump()}, console)
            return

        if run.status == "QUEUED":
            queue_msg = "✓ Run created and queued"
            if run.runs_ahead is not None:
                queue_msg += f" (~{run.runs_ahead} runs ahead)"
            console.print(f"[yellow]{queue_msg}[/yellow]")
            if run.queue_reason:
                console.print(f"[yellow]Reason:[/yellow] {run.queue_reason}")
            console.print("[dim]The run will start automatically when capacity is available.[/dim]")
        else:
            console.print("[green]✓ Run created successfully![/green]")

        if run.notice:
            console.print(f"[cyan]{run.notice}[/cyan]")

        dashboard_url = f"{app_config.frontend_url}/dashboard/training/{run.id}"
        console.print("\n[cyan]Monitor run at:[/cyan]")
        console.print(f"  [link={dashboard_url}]{dashboard_url}[/link]")

        if run.status != "QUEUED":
            console.print("\n[dim]View logs with:[/dim]")
            console.print(f"  prime train logs {run.id} -f")

    except ValidationError as e:
        console.print("[red]Configuration Error:[/red]")
        for err in e.errors:
            loc = err.get("loc", [])
            path = ".".join(str(x) for x in loc if x != "body")
            msg = err.get("msg", "")
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            if path:
                console.print(f"  [yellow]{path}[/yellow]: {msg}")
            else:
                console.print(f"  {msg}")
        raise typer.Exit(1)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("models", rich_help_panel="Commands", epilog=RL_MODELS_JSON_HELP)
def list_models(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List available models for Hosted Training."""
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)
        config = Config()

        models = rl_client.list_models(team_id=config.team_id)

        if output == "json":
            output_data_as_json({"models": [m.model_dump() for m in models]}, console)
            return

        if not models:
            console.print("[yellow]No models available for Hosted Training.[/yellow]")
            console.print(
                "[dim]This could mean no healthy Hosted Training clusters are running.[/dim]"
            )
            return

        table = Table(
            title="Hosted Training - Models",
        )
        table.add_column("Model", style="cyan")
        table.add_column("Status")
        table.add_column("Input", style="green", justify="right")
        table.add_column("Output", style="green", justify="right")
        table.add_column("Train", style="green", justify="right")

        promo_labels: List[str] = []
        for model in sorted(models, key=lambda model: _model_name_sort_key(model.name)):
            if model.at_capacity:
                status = "[red]At Capacity[/red]"
            else:
                status = "[green]Available[/green]"
            list_train, eff_train = model.resolve_prices("training")
            list_input, eff_input = model.resolve_prices("inference_input")
            list_output, eff_output = model.resolve_prices("inference_output")
            table.add_row(
                model.name,
                status,
                format_promo_price(list_input, eff_input) or "-",
                format_promo_price(list_output, eff_output) or "-",
                format_promo_price(list_train, eff_train) or "-",
            )
            if model.promo_label and model.promo_label not in promo_labels:
                promo_labels.append(model.promo_label)

        caption = (
            "[dim]Prices are per 1M tokens. All models support context windows of 64K tokens.[/dim]"
        )
        caption_lines = [caption]
        if promo_labels:
            joined = ", ".join(rich_escape(label) for label in promo_labels)
            caption_lines.append(f"[bold yellow]{joined}[/bold yellow]")
        table.caption = "\n".join(caption_lines)
        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _prompt_required_text(label: str, help_text: str, empty_error: str) -> str:
    console.print(f"\n[bold]{label}[/bold] [dim](required)[/dim]")
    console.print(f"[dim]{help_text}[/dim]")
    while True:
        value = typer.prompt("", prompt_suffix="> ").strip()
        if value:
            return value
        console.print(f"[red]{empty_error}[/red]")


def _prompt_optional_text(label: str, help_text: str) -> str | None:
    console.print(f"\n[bold]{label}[/bold] [dim](optional)[/dim]")
    console.print(f"[dim]{help_text}[/dim]")
    value = typer.prompt("", default="", show_default=False, prompt_suffix="> ").strip()
    return value or None


def _format_model_request_feedback(models: str, context: str | None) -> str:
    message = f"Hosted Training model request\n\nModels:\n{models}"
    if context:
        message += f"\n\nContext:\n{context}"
    return message


@app.command("request", rich_help_panel="Commands")
def request_models() -> None:
    """Request models for Hosted Training."""
    console.print("[bold]Hosted Training Model Request[/bold]")
    console.print("[dim]Tell us which models you want available for training.[/dim]")

    try:
        models = _prompt_required_text(
            "Model(s)",
            "Use provider/model names if you know them; comma-separated is fine.",
            "At least one model is required.",
        )
        context = _prompt_optional_text(
            "Use case or context",
            "Share what you want to train or why this model matters.",
        )
    except Abort:
        console.print("\n[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)

    try:
        submit_feedback(
            message=_format_model_request_feedback(models, context),
            category="feature",
        )
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print("[green]Request submitted. Thanks![/green]")


def _unwrap_single_schema_variant(prop: Dict[str, Any]) -> Dict[str, Any]:
    """Return the concrete schema when a field is an optional/single wrapped variant."""
    if "anyOf" in prop:
        non_null = [variant for variant in prop["anyOf"] if variant.get("type") != "null"]
        if len(non_null) == 1:
            return _unwrap_single_schema_variant(non_null[0])

    if "allOf" in prop and len(prop["allOf"]) == 1:
        return _unwrap_single_schema_variant(prop["allOf"][0])

    return prop


def _resolve_schema_ref(prop: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a local JSON schema ref into its concrete schema when possible."""
    if "$ref" not in prop:
        return prop

    ref_name = prop["$ref"].rsplit("/", 1)[-1]
    return defs.get(ref_name, prop)


def _flatten_config_schema(
    schema: Dict[str, Any],
    defs: Dict[str, Any],
    prefix: str = "",
) -> list[tuple[str, str, str]]:
    """Walk a JSON schema and return (dotted_path, type, default) for each leaf field."""
    leaf_rows: list[tuple[str, str, str]] = []
    nested_rows: list[tuple[str, str, str]] = []
    props = schema.get("properties", {})
    for name, prop in props.items():
        if name in ("model_config", "env_file"):
            continue
        path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        resolved = _resolve_schema_ref(_unwrap_single_schema_variant(prop), defs)

        if resolved.get("type") == "object" and "properties" in resolved:
            nested_rows.extend(_flatten_config_schema(resolved, defs, path))
            continue

        if resolved.get("type") == "array":
            items = _resolve_schema_ref(
                _unwrap_single_schema_variant(resolved.get("items", {})),
                defs,
            )
            if items.get("type") == "object" and "properties" in items:
                nested_rows.extend(_flatten_config_schema(items, defs, f"{path}[]"))
                continue

        type_str = _schema_type_str(prop, defs)
        default = str(prop.get("default", "")) if "default" in prop else ""
        leaf_rows.append((path, type_str, default))

    return leaf_rows + nested_rows


def _schema_type_str(prop: Dict[str, Any], defs: Dict[str, Any]) -> str:
    """Produce a human-readable type string from a JSON schema property."""
    unwrapped = _unwrap_single_schema_variant(prop)
    if unwrapped is not prop:
        return _schema_type_str(unwrapped, defs)

    if "$ref" in prop:
        return prop["$ref"].rsplit("/", 1)[-1]

    if "anyOf" in prop:
        parts = [
            _schema_type_str(variant, defs)
            for variant in prop["anyOf"]
            if variant.get("type") != "null"
        ]
        return " | ".join(parts) if parts else "any"

    if "allOf" in prop:
        parts = [_schema_type_str(variant, defs) for variant in prop["allOf"]]
        return " & ".join(part for part in parts if part) or "any"

    if prop.get("type") == "array":
        inner = _schema_type_str(prop.get("items", {}), defs)
        return f"list[{inner}]"

    resolved = _resolve_schema_ref(prop, defs)
    if resolved is not prop:
        return _schema_type_str(resolved, defs)

    return prop.get("type", "any")


RL_CONFIGS_JSON_HELP = json_output_help(
    ".configs[] = {section, name, type, default}",
)


@app.command("configs", rich_help_panel="Commands", epilog=RL_CONFIGS_JSON_HELP)
def list_configs(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List available configuration options for Hosted Training."""
    validate_output_format(output, console)

    schema = RLConfig.model_json_schema()
    defs = schema.get("$defs", {})
    rows = _flatten_config_schema(schema, defs)
    if output == "json":
        output_data_as_json(
            {
                "configs": [
                    {
                        "section": r[0].rsplit(".", 1)[0] if "." in r[0] else "",
                        "name": r[0].rsplit(".", 1)[-1],
                        "type": r[1],
                        "default": r[2],
                    }
                    for r in rows
                ]
            },
            console,
        )
        return

    table = Table(title="Hosted Training - Config Options")
    table.add_column("Section", style="magenta")
    table.add_column("Config", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Default", style="yellow")

    prev_section = None
    for path, type_str, default in rows:
        section = path.rsplit(".", 1)[0] if "." in path else ""
        name = path.rsplit(".", 1)[-1]
        if prev_section is not None and section != prev_section:
            table.add_section()
        display_section = section if section != prev_section else ""
        prev_section = section
        table.add_row(display_section, name, type_str, default)

    console.print(table)
    console.print(
        "\n[dim]Use these in your TOML config file. "
        "See 'prime train init' to generate a template.[/dim]"
    )


def _list_runs_impl(
    team: Optional[str], num: int, page: int, output: str, mine: bool = False
) -> None:
    """Implementation for listing Hosted Training runs."""
    validate_output_format(output, console)

    if num < 1 or page < 1:
        console.print("[red]Error:[/red] --num and --page must be at least 1")
        raise typer.Exit(1)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)
        config = Config()

        team_id = team or config.team_id

        all_runs = rl_client.list_runs(team_id=team_id)

        if mine:
            current_user_id = config.user_id
            if not current_user_id:
                console.print(
                    "[red]Error:[/red] Cannot filter by user - no user_id configured. "
                    "Run [bold]prime whoami[/bold] to refresh your config."
                )
                raise typer.Exit(1)
            all_runs = [r for r in all_runs if r.user_id == current_user_id]

        total_count = len(all_runs)

        # Sort by created_at descending and paginate
        all_runs.sort(key=lambda r: r.created_at, reverse=True)
        start = (page - 1) * num
        runs = all_runs[start : start + num]

        if output == "json":
            output_data_as_json(
                {
                    "runs": [r.model_dump() for r in runs],
                    "total": total_count,
                    "page": page,
                    "per_page": num,
                },
                console,
            )
            return

        if not runs:
            if page > 1:
                console.print("[yellow]No more results.[/yellow]")
            else:
                console.print("[yellow]No Hosted Training runs found.[/yellow]")
            return

        table = Table(title="Hosted Training Runs")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold")
        table.add_column("Model", style="magenta")
        table.add_column("Environments", style="green")
        table.add_column("Steps", justify="right")
        table.add_column("Created", style="dim")

        for run in runs:
            formatted = _format_run_for_display(run)
            status_color = _get_status_color(run.status)
            table.add_row(
                formatted["id"],
                f"[{status_color}]{formatted['status']}[/{status_color}]",
                formatted["model"][:30],
                formatted["environments"],
                formatted["steps"],
                formatted["created_at"],
            )

        console.print(table)

        if total_count > page * num:
            console.print(
                f"\n[yellow]Showing page {page} of results. "
                f"Use --page {page + 1} to see more.[/yellow]"
            )
        else:
            console.print(f"\n[dim]Total: {total_count} run(s)[/dim]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list", rich_help_panel="Commands", epilog=RL_LIST_JSON_HELP)
def list_runs(
    team: Optional[str] = typer.Option(None, "--team", "-t", help="Filter by team ID"),
    mine: bool = typer.Option(
        False, "--mine", help="Filter to only your own runs (useful for admin accounts)"
    ),
    num: int = typer.Option(20, "--num", "-n", help="Items per page"),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your runs (alias: ls)."""
    _list_runs_impl(team, num, page, output, mine=mine)


@app.command("get", rich_help_panel="Commands", epilog=RL_RUN_JSON_HELP)
def get_run(
    run_id: str = typer.Argument(..., help="Run ID to get details for"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Get details of a specific run.

    Example:

        prime train get <run_id>

        prime train get <run_id> -o json
    """
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        run = rl_client.get_run(run_id)

        if output == "json":
            output_data_as_json({"run": run.model_dump()}, console)
            return

        # Display run details
        formatted = _format_run_for_display(run)
        status_color = _get_status_color(run.status)

        console.print(f"[bold]Run {run_id}[/bold]\n")
        status_text = run.status
        if run.status == "QUEUED" and run.runs_ahead is not None:
            status_text += f" (~{run.runs_ahead} runs ahead)"
        console.print(f"  Status: [{status_color}]{status_text}[/{status_color}]")
        console.print(f"  Model: [magenta]{formatted['model']}[/magenta]")
        console.print(f"  Environments: [green]{formatted['environments']}[/green]")
        console.print(f"  Max Steps: {formatted['steps']}")
        batch_size = "-" if run.batch_size is None else str(run.batch_size)
        console.print(f"  Batch Size: {batch_size}")
        console.print(f"  Rollouts per Example: {formatted['rollouts']}")
        if run.max_tokens:
            console.print(f"  Max Tokens: {run.max_tokens}")
        if run.wandb_project:
            console.print(f"  W&B: {run.wandb_entity or ''}/{run.wandb_project}")
        if run.team_id:
            console.print(f"  Team: {run.team_id}")
        console.print(f"  Created: [dim]{formatted['created_at']}[/dim]")
        if run.started_at:
            console.print(f"  Started: [dim]{run.started_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
        if run.completed_at:
            console.print(f"  Completed: [dim]{run.completed_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
        if run.error_message:
            console.print(f"  Error: [red]{run.error_message}[/red]")

        if run.failure_analysis:
            _render_failure_analysis(run.failure_analysis)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("stop", rich_help_panel="Commands")
def stop_run(
    run_id: str = typer.Argument(..., help="Run ID to stop"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Stop a run."""
    try:
        if not force:
            confirm = typer.confirm(f"Are you sure you want to stop run {run_id}?")
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)

        api_client = APIClient()
        rl_client = RLClient(api_client)

        run = rl_client.stop_run(run_id)

        console.print(f"[green]✓ Run {run_id} stopped successfully[/green]")
        console.print(f"Status: {run.status}")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("delete", rich_help_panel="Commands")
def delete_run(
    run_id: str = typer.Argument(..., help="Run ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a run."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to permanently delete run {run_id}?")
        if not confirm:
            console.print("Cancelled.")
            raise typer.Exit(0)

    api_client = APIClient()

    # Always go through the LoRA-shared endpoint. The backend's
    # `delete_single_run` kind-routes DEDICATED_FULL_FT rows server-side
    # to `training_service.delete_dedicated_run` (helm uninstall + ns
    # delete + token disable + soft-delete), so a single ownership-gated
    # call handles both kinds correctly. Avoids:
    #   * the admin-gate on the dedicated `/training/runs` endpoint
    #     (non-admin owners couldn't delete their own runs via the
    #     hosted-first probe),
    #   * an Optional-kind-on-the-wire race where a missing field would
    #     mis-route to the LoRA cleanup path on the CLI side and leak
    #     the helm release.
    rl_client = RLClient(api_client)
    try:
        rl_client.delete_run(run_id)
        console.print(f"[green]✓ Run {run_id} deleted successfully[/green]")
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("restart", rich_help_panel="Commands")
def restart_run(
    run_id: str = typer.Argument(..., help="Run ID to restart"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Restart a running run from its latest checkpoint.

    Only RUNNING runs can be restarted (checkpoints still on PVC).
    For STOPPED/FAILED/COMPLETED runs, checkpoints have been cleaned up.

    Example:

        prime train restart <run_id>
    """
    try:
        if not force:
            confirm = typer.confirm(
                f"Are you sure you want to restart run {run_id} from its latest checkpoint?"
            )
            if not confirm:
                console.print("Cancelled.")
                raise typer.Exit(0)

        api_client = APIClient()
        rl_client = RLClient(api_client)

        run = rl_client.restart_run(run_id)

        console.print(f"[green]✓ Run {run_id} restarting from checkpoint[/green]")
        console.print(f"Status: {run.status}")

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def _print_new_lines(last_lines: list[str], current_lines: list[str]) -> None:
    """Print only lines in current_lines that aren't in the tail of last_lines.

    Uses suffix-prefix overlap detection to avoid reprinting lines between polls.
    """
    if current_lines == last_lines:
        return
    if not last_lines:
        for line in current_lines:
            console.print(line)
        return
    overlap = 0
    max_overlap = min(len(last_lines), len(current_lines))
    for i in range(1, max_overlap + 1):
        if last_lines[-i:] == current_lines[:i]:
            overlap = i
    for line in current_lines[overlap:]:
        console.print(line)


def _is_log_startup_error(error: APIError) -> bool:
    message = str(error).lower()
    startup_markers = (
        "queued",
        "pending",
        "starting",
        "initializing",
        "not started",
        "not available yet",
        "logs are not available",
        "logs not available",
    )
    return "404" in message and any(marker in message for marker in startup_markers)


def _fetch_logs_with_startup_poll(fetch_fn: Any, tail: int, label: str) -> str:
    for poll in range(HOSTED_TRAINING_LOG_STARTUP_POLLS + 1):
        try:
            return fetch_fn(tail)
        except APIError as e:
            if not _is_log_startup_error(e):
                raise
            if poll == 0:
                console.print(
                    f"[yellow]Hosted Training run is starting; waiting for {label} logs...[/yellow]"
                )
            if poll == HOSTED_TRAINING_LOG_STARTUP_POLLS:
                console.print(
                    "[yellow]Hosted Training run has not started yet. "
                    "Logs will be available once it is running.[/yellow]"
                )
                raise typer.Exit(0)
            time.sleep(HOSTED_TRAINING_LOG_STARTUP_POLL_SECONDS)

    raise AssertionError("unreachable")


def _stream_logs(
    fetch_fn: Any,
    tail: int,
    raw: bool,
    follow: bool,
    label: str,
) -> None:
    """Common print loop for orchestrator and env-server log fetches."""

    def render(raw_logs: str) -> list[str]:
        if raw:
            return raw_logs.splitlines()
        return clean_logs(raw_logs)

    if follow:
        console.print(f"[dim]Watching {label} logs... (Ctrl+C to stop)[/dim]\n")
        last_lines: list[str] = []
        consecutive_errors = 0

        while True:
            try:
                raw_logs = fetch_fn(tail)
                consecutive_errors = 0
            except APIError as e:
                if _is_log_startup_error(e):
                    console.print(
                        "[yellow]Hosted Training run is starting; waiting for logs...[/yellow]"
                    )
                    time.sleep(HOSTED_TRAINING_LOG_STARTUP_POLL_SECONDS)
                    continue
                consecutive_errors += 1
                if "429" in str(e):
                    if consecutive_errors >= 3:
                        console.print("[yellow]Rate limited. Waiting 30s...[/yellow]")
                        time.sleep(30)
                    else:
                        time.sleep(10)
                    continue
                raise

            current_lines = render(raw_logs)
            _print_new_lines(last_lines, current_lines)
            last_lines = current_lines
            time.sleep(HOSTED_TRAINING_LOG_FOLLOW_POLL_SECONDS)
    else:
        raw_logs = _fetch_logs_with_startup_poll(fetch_fn, tail, label)
        rendered = render(raw_logs)
        if rendered:
            for line in rendered:
                console.print(line)
        else:
            console.print("[yellow]No logs available yet.[/yellow]")


def _handle_logs_api_error(e: APIError) -> None:
    if _is_log_startup_error(e):
        msg = "Hosted Training run has not started yet. Logs will be available once running."
        console.print(f"[yellow]{msg}[/yellow]")
        raise typer.Exit(0)
    console.print(f"[red]Error:[/red] {e}")
    raise typer.Exit(1)


_SINCE_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86_400}


def _parse_since(since: Optional[str]) -> Optional[int]:
    """Convert a ``--since`` flag to a seconds value the backend accepts.

    Accepts forms like ``15m``, ``1h``, ``6h``, ``24h``, ``900``. Returns
    None when the flag isn't set (backend default applies)."""
    if since is None:
        return None
    raw = since.strip().lower()
    if not raw:
        return None
    if raw[-1] in _SINCE_UNIT_SECONDS and raw[:-1].isdigit():
        seconds = int(raw[:-1]) * _SINCE_UNIT_SECONDS[raw[-1]]
    elif raw.isdigit():
        seconds = int(raw)
    else:
        raise typer.BadParameter(
            f"Invalid --since value '{since}'. Use e.g. '15m', '1h', '6h', '24h', or seconds.",
            param_hint="--since",
        )
    if seconds < 60 or seconds > 86_400:
        raise typer.BadParameter(
            "--since must be between 60s and 24h (86400s).",
            param_hint="--since",
        )
    return seconds


def _parse_env_qualifier_with_index(env: str) -> tuple[str, int, bool]:
    """Parse an env qualifier and report whether a numeric suffix was present."""
    name, sep, idx_str = env.rpartition("/")
    if sep and name and idx_str.isdigit():
        return name, int(idx_str), True
    return env, 0, False


@app.command("logs", rich_help_panel="Monitoring")
def get_logs(
    run_id: str = typer.Argument(..., help="Run ID to get logs for"),
    component: Optional[str] = typer.Option(
        None,
        "--component",
        "-c",
        help=(
            "Pod to read logs from: 'orchestrator' (default), 'trainer', "
            "'inference', or 'env-server'. trainer/inference apply only "
            "to dedicated full-FT runs. Inferred from --env when omitted."
        ),
    ),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        help=(
            "Env-server name. Implies --component=env-server. "
            "Use 'name/N' to disambiguate when multiple env-servers share a name. "
            "List with 'prime train components <run_id>'."
        ),
    ),
    tail: int = typer.Option(1000, "--tail", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Show raw logs without formatting"),
    search: Optional[str] = typer.Option(
        None,
        "--search",
        help=(
            "Filter to lines containing this text. "
            "Combine with --regex to use a regular expression instead of a substring."
        ),
    ),
    regex: bool = typer.Option(
        False,
        "--regex",
        help="Treat --search as a regex (RE2 syntax).",
    ),
    level: Optional[str] = typer.Option(
        None,
        "--level",
        help="Filter to one log level: ERROR | WARNING | SUCCESS | INFO | DEBUG.",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help=(
            "Time window for filtered queries. Accepts e.g. '15m', '1h', '6h', "
            "'24h', or a raw integer seconds value. Default 15m. "
            "Applies when --search or --level is set."
        ),
    ),
) -> None:
    """Get logs for a run.

    Defaults to the orchestrator pod. Use ``--component`` to pick one of
    ``trainer`` / ``inference`` / ``env-server`` (dedicated full-FT only).
    Pass ``--env <name>`` to read an env-server pod by name (shorthand for
    ``--component=env-server``).

    List available pods first with ``prime train components <run_id>``.

    Per-rank narrowing on multi-replica trainer/inference is not yet
    surfaced here — `--local-ranks-filter=0` in the chart's torchrun
    invocation already dedupes the in-pod rank fan-out, and per-pod
    inspection on multi-node runs requires kubectl + the PVC log files.

    Examples:

        prime train logs <run_id>
        prime train logs <run_id> -f
        prime train logs <run_id> --search Backpressure
        prime train logs <run_id> --level ERROR --since 1h
        prime train logs <run_id> --search 'Step \\d+' --regex
        prime train logs <run_id> -c trainer
        prime train logs <run_id> -c inference
        prime train logs <run_id> --env reverse-text
        prime train logs <run_id> --env reverse-text/1 -f
    """
    valid_components = ("orchestrator", "trainer", "inference", "env-server")
    if component is None:
        component = "env-server" if env is not None else "orchestrator"
    elif component not in valid_components:
        raise typer.BadParameter(
            f"Invalid component '{component}'. Use one of: {', '.join(valid_components)}.",
            param_hint="--component",
        )
    if env is not None and component != "env-server":
        raise typer.BadParameter(
            f"--env applies only to env-server logs. Drop --component={component} or drop --env.",
            param_hint="--env",
        )
    if component == "env-server" and env is None:
        raise typer.BadParameter(
            "--env is required when reading env-server logs. "
            "Run 'prime train components <run_id>' to list available env-servers.",
            param_hint="--env",
        )

    normalized_level: Optional[str] = None
    if level:
        normalized_level = level.upper()
        if normalized_level not in {"ERROR", "WARNING", "SUCCESS", "INFO", "DEBUG"}:
            raise typer.BadParameter(
                f"Invalid level '{level}'. Use ERROR, WARNING, SUCCESS, INFO, or DEBUG.",
                param_hint="--level",
            )

    if regex and not search:
        raise typer.BadParameter(
            "--regex has no effect without --search. Pass --search <pattern> too.",
            param_hint="--regex",
        )

    since_seconds = _parse_since(since)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        env_name_q, env_index_q, env_has_index_q = (
            _parse_env_qualifier_with_index(env) if env is not None else (None, 0, False)
        )

        # Dedicated full-FT env-servers run as 1-replica-per-name
        # StatefulSets, so a `name/N` qualifier (typed by hand or copied
        # from a stale source) would incorrectly route through the
        # legacy `/env-server-logs` endpoint and 404 on a backend that
        # never registered an indexed pod for this run. Probe the run
        # kind once and strip the index for full-FT runs so the unified
        # `/logs` route handles them. Skipped for the no-index path
        # (already routes correctly) and tolerated on probe failure
        # (preserves prior behaviour for transient API blips).
        if component == "env-server" and env is not None and env_has_index_q:
            try:
                if rl_client.get_run(run_id).kind == "DEDICATED_FULL_FT":
                    env = env_name_q
                    env_has_index_q = False
            except APIError:
                pass

        if component == "env-server" and env is not None and env_has_index_q:
            assert env_name_q is not None
            # Legacy shared-RFT env-server (`name/index` qualifier) — go
            # through the dedicated env-server endpoint which uses the
            # cluster_id-backed pod lookup path. Dedicated full-FT
            # env-servers use the unified /logs route with
            # component=env-server + env_name (StatefulSets always run
            # one pod per env, so no index disambiguation needed).

            def fetch(t: int) -> str:
                return rl_client.get_env_server_logs(
                    run_id,
                    env_name=env_name_q,
                    env_index=env_index_q,
                    tail_lines=t,
                    search=search,
                    regex=regex,
                    level=normalized_level,
                    since_seconds=since_seconds,
                )

            label = f"env-server {env}"
        elif component == "orchestrator":

            def fetch(t: int) -> str:
                return rl_client.get_logs(
                    run_id,
                    tail_lines=t,
                    search=search,
                    regex=regex,
                    level=normalized_level,
                    since_seconds=since_seconds,
                )

            label = "orchestrator"
        else:
            # trainer / inference / dedicated env-server — unified /logs
            # route. env (no slash) names the dedicated env-server's
            # StatefulSet.
            fetch_component = component
            fetch_env = env if component == "env-server" else None

            def fetch(t: int) -> str:
                return rl_client.get_logs(
                    run_id,
                    tail_lines=t,
                    search=search,
                    regex=regex,
                    level=normalized_level,
                    since_seconds=since_seconds,
                    component=fetch_component,
                    env_name=fetch_env,
                )

            label = f"env-server {env}" if component == "env-server" else component

        _stream_logs(
            fetch_fn=fetch,
            tail=tail,
            raw=raw,
            follow=follow,
            label=label,
        )

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching logs.[/dim]")
    except APIError as e:
        _handle_logs_api_error(e)


@app.command("components", rich_help_panel="Monitoring")
def list_components(
    run_id: str = typer.Argument(..., help="Run ID to list components for"),
) -> None:
    """List pods (orchestrator + env-servers) for a run.

    Use the env name shown here with
    ``prime train logs <run_id> -c env-server --env <name>``. When multiple
    env-servers share a name, the qualified form ``name/N`` is shown — pass
    that exact string to ``--env``.

    Example:

        prime train components <run_id>
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)
        run = rl_client.get_run(run_id)
        env_servers: List[EnvServerInfo] = rl_client.list_env_servers(run_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    table = Table(title=f"Components for {run_id}")
    table.add_column("Component", style="cyan")
    table.add_column("Env", style="green")
    table.add_column("Status")

    table.add_row("orchestrator", "-", run.status)

    name_counts: Dict[str, int] = {}
    for es in env_servers:
        if es.env_name:
            name_counts[es.env_name] = name_counts.get(es.env_name, 0) + 1

    for es in env_servers:
        if es.env_name and name_counts.get(es.env_name, 0) > 1:
            env_label = f"{es.env_name}/{es.env_index}"
        else:
            env_label = es.env_name or "?"
        table.add_row("env-server", env_label, es.status)

    console.print(table)
    if env_servers:
        console.print(
            "\n[dim]View env-server logs with:[/dim] "
            "[bold]prime train logs <run_id> -c env-server --env <env>[/bold]"
        )


@app.command("init", rich_help_panel="Commands")
def init_config(
    output_path: str = typer.Argument(
        "rl.toml",
        help="Output path for the config file",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """Generate a template config file for a Hosted Training run.

    Example:

        prime train init           # Creates rl.toml

        prime train init my-config.toml
    """
    path = Path(output_path)

    if path.exists() and not force:
        console.print(f"[red]Error:[/red] {output_path} already exists. Use --force to overwrite.")
        raise typer.Exit(1)

    # Auto-detect environment
    environment: str | None = None
    metadata = find_environment_metadata()
    if metadata:
        owner = metadata.get("owner")
        name = metadata.get("name")
        if owner and name:
            environment = f"{owner}/{name}"
            console.print(f"[dim]Detected environment: {environment}[/dim]")

    path.parent.mkdir(parents=True, exist_ok=True)

    template = generate_rl_config_template(environment)
    path.write_text(template)

    console.print(f"[green]✓[/green] Created {output_path}")
    console.print(f"\n[dim]Run with:[/dim] prime train {output_path}")


@app.command("metrics", rich_help_panel="Monitoring", epilog=RL_METRICS_JSON_HELP)
def get_metrics(
    run_id: str = typer.Argument(..., help="Run ID to get metrics for"),
    min_step: Optional[int] = typer.Option(None, "--min-step", help="Minimum step (inclusive)"),
    max_step: Optional[int] = typer.Option(None, "--max-step", help="Maximum step (inclusive)"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Maximum number of records"),
) -> None:
    """Get Hosted Training metrics for a run.

    Example:

        prime train metrics <run_id>

        prime train metrics <run_id> --min-step 10 --max-step 50

        prime train metrics <run_id> | jq '.metrics[0]'
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        metrics = rl_client.get_metrics(
            run_id,
            min_step=min_step,
            max_step=max_step,
            limit=limit,
        )

        output_data_as_json({"metrics": metrics}, console)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("rollouts", rich_help_panel="Monitoring", epilog=RL_ROLLOUTS_JSON_HELP)
def get_rollouts(
    run_id: str = typer.Argument(..., help="Run ID to get rollouts for"),
    step: int = typer.Option(..., "--step", "-s", help="Step number to get rollouts for"),
    page: int = typer.Option(1, "--page", "-p", help="Page number (1-indexed)"),
    num: int = typer.Option(100, "--num", "-n", help="Items per page"),
) -> None:
    """Get rollout samples for a run.

    Example:

        prime train rollouts <run_id> --step 10

        prime train rollouts <run_id> -s 50 --num 100 | jq '.samples[0]'
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        result = rl_client.get_rollouts(
            run_id,
            step=step,
            page=page,
            limit=num,
        )

        output_data_as_json(result, console)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("progress", rich_help_panel="Monitoring", epilog=RL_PROGRESS_JSON_HELP)
def get_progress(
    run_id: str = typer.Argument(..., help="Run ID to get progress for"),
) -> None:
    """Get progress information, including which steps have samples and distributions.

    Example:

        prime train progress <run_id>

        prime train progress <run_id> | jq '.latest_step'
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        progress = rl_client.get_progress(run_id)

        output_data_as_json(progress, console)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("distributions", rich_help_panel="Monitoring", epilog=RL_DISTRIBUTIONS_JSON_HELP)
def get_distributions(
    run_id: str = typer.Argument(..., help="Run ID to get distributions for"),
    distribution_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Distribution type (defaults to all)"
    ),
    step: Optional[int] = typer.Option(
        None, "--step", "-s", help="Step number (defaults to latest)"
    ),
) -> None:
    """Get reward/advantage distribution histogram for a run.

    Example:

        prime train distributions <run_id>

        prime train distributions <run_id> --type rewards --step 50
    """
    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        result = rl_client.get_distributions(
            run_id,
            distribution_type=distribution_type,
            step=step,
        )

        output_data_as_json(result, console)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("checkpoints", rich_help_panel="Monitoring", epilog=RL_CHECKPOINTS_JSON_HELP)
def list_checkpoints(
    run_id: str = typer.Argument(..., help="Run ID to list checkpoints for"),
    status_filter: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter by status (READY, PENDING, UPLOADING, FAILED)"
    ),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List checkpoints for a run.

    Example:

        prime train checkpoints <run_id>

        prime train checkpoints <run_id> --status READY
    """
    validate_output_format(output, console)

    try:
        api_client = APIClient()
        rl_client = RLClient(api_client)

        checkpoints = rl_client.list_checkpoints(run_id, status_filter=status_filter)

        if output == "json":
            output_data_as_json({"checkpoints": [cp.model_dump() for cp in checkpoints]}, console)
            return

        if not checkpoints:
            console.print("[yellow]No checkpoints found for this run.[/yellow]")
            return

        table = Table(title=f"Checkpoints for {run_id}")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Step", justify="right", style="bold")
        table.add_column("Status", style="bold")
        table.add_column("Size", justify="right")
        table.add_column("Created", style="dim")

        status_colors = {
            "READY": "green",
            "PENDING": "yellow",
            "UPLOADING": "blue",
            "FAILED": "red",
        }

        for cp in checkpoints:
            color = status_colors.get(cp.status, "white")
            table.add_row(
                cp.id,
                str(cp.step),
                f"[{color}]{cp.status}[/{color}]",
                format_file_size(cp.size_bytes) if cp.size_bytes is not None else "-",
                cp.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# `prime train usage` — token usage and price for one run; lives next to the
# other run-scoped monitoring commands. Implemented in commands/usage.py and
# also re-exposed as the top-level `prime usage` summary command.
app.command("usage", rich_help_panel="Monitoring", epilog=RUN_USAGE_JSON_HELP)(run_usage_command)
