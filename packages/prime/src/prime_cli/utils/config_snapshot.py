"""The config files and command a hosted run was launched with.

`prime train` and `prime eval run --hosted` send these when they create a run,
so the platform's Config tab can show the user's files as written (comments
included) and the command they typed. The files are sent unchanged; the run's
owner and team can read them. The command has secret values replaced, because
users pass secrets on the command line (`-e KEY=VALUE`, `--custom-secrets`).

Platforms without snapshot support ignore the field.
"""

import json
import re
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

# Kept within the platform's limits: a snapshot over them would fail the
# whole create request, so the CLI drops the part that does not fit instead.
MAX_CONFIG_FILES = 32
MAX_CONFIG_FILES_BYTES = 256 * 1024
MAX_COMMAND_CHARS = 16 * 1024

REDACTED = "***"

OPT_OUT_FLAG = "--no-config-upload"
NOTICE = (
    "This command, with secret values hidden, and your config file as written "
    "will be visible to you and your team on the run's Config tab. "
    f"Pass {OPT_OUT_FLAG} to skip."
)

# Flags whose value is KEY=VALUE, where VALUE may be a secret.
_KEY_VALUE_FLAGS = frozenset({"-e", "--env-var"})
_HEADER_FLAGS = frozenset({"--header"})
# A flag or JSON key is secret when one of its words is one of these...
_SECRET_WORDS = frozenset(
    {
        "apikey",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "key",
        "password",
        "passwd",
        "secret",
        "secrets",
        "token",
    }
)
# ...unless it names where the secret lives rather than holding it
# (`--api-key-var OPENAI_API_KEY`).
_NAME_SUFFIXES = frozenset({"env", "file", "id", "name", "path", "type", "url", "var"})
# Well-known credential shapes, replaced wherever they appear.
_CREDENTIAL = re.compile(
    r"\b("
    r"sk-[A-Za-z0-9_\-]{16,}"
    r"|hf_[A-Za-z0-9]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[abprs]-[A-Za-z0-9\-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
    r")"
)


def _is_secret_name(name: str) -> bool:
    words = [word for word in re.split(r"[-_\s]+", name.strip("-").lower()) if word]
    if not words or words[-1] in _NAME_SUFFIXES:
        return False
    return any(word in _SECRET_WORDS for word in words)


def _scrub_json(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        scrubbed: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _is_secret_name(key) and item not in (None, ""):
                scrubbed[key] = REDACTED
                changed = True
            else:
                scrubbed[key], item_changed = _scrub_json(item)
                changed = changed or item_changed
        return scrubbed, changed
    if isinstance(value, list):
        items = [_scrub_json(item) for item in value]
        return [item for item, _ in items], any(changed for _, changed in items)
    return value, False


def _scrub_value(value: str) -> str:
    """A plain argument: secret JSON keys and credential-shaped text go."""
    stripped = value.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            parsed = json.loads(value)
        except ValueError:
            pass
        else:
            scrubbed, changed = _scrub_json(parsed)
            if changed:
                value = json.dumps(scrubbed, separators=(",", ":"))
    return _CREDENTIAL.sub(REDACTED, value)


def _scrub_key_value(value: str) -> str:
    # `KEY` alone (read from the local environment) and env-file paths are
    # names, not secrets.
    key, sep, _ = value.partition("=")
    return f"{key}={REDACTED}" if sep else value


def _scrub_header(value: str) -> str:
    name, sep, _ = value.partition(":")
    return f"{name}: {REDACTED}" if sep else REDACTED


def _flag_scrubber(flag: str):
    if flag in _KEY_VALUE_FLAGS:
        return _scrub_key_value
    if flag in _HEADER_FLAGS:
        return _scrub_header
    if flag.startswith("--") and _is_secret_name(flag):
        return lambda _value: REDACTED
    return None


def scrub_command_args(args: Sequence[str]) -> list[str]:
    """Replace secret values in command-line arguments, keeping their shape."""
    scrubbed: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg == "--":
            scrubbed.append(arg)
            scrubbed.extend(_scrub_value(rest) for rest in args[index:])
            break
        if arg.startswith("--") and "=" in arg:
            flag, _, value = arg.partition("=")
            scrub = _flag_scrubber(flag)
            scrubbed.append(f"{flag}={(scrub or _scrub_value)(value)}")
            continue
        if arg.startswith("-e") and len(arg) > 2 and not arg.startswith("--"):
            scrubbed.append("-e" + _scrub_key_value(arg[2:]))
            continue
        scrub = _flag_scrubber(arg) if arg.startswith("-") else None
        if scrub is None:
            scrubbed.append(_scrub_value(arg))
            continue
        scrubbed.append(arg)
        if index < len(args):
            scrubbed.append(scrub(args[index]))
            index += 1
    return scrubbed


def launch_command(argv: Optional[Sequence[str]] = None) -> Optional[str]:
    """The command the user typed, as `prime ...`, with secrets replaced.
    None when it is too long to send."""
    args = list(sys.argv if argv is None else argv)[1:]
    command = shlex.join(["prime", *scrub_command_args(args)])
    return command if len(command) <= MAX_COMMAND_CHARS else None


def build_config_snapshot(
    config_paths: Sequence[str], argv: Optional[Sequence[str]] = None
) -> tuple[dict[str, Any], Optional[str]]:
    """The snapshot to send with a create request, and a warning to show when
    part of it had to be left out."""
    warning: Optional[str] = None
    files: list[dict[str, str]] = []
    total = 0
    for path in config_paths[:MAX_CONFIG_FILES]:
        try:
            content = Path(path).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            files = []
            warning = f"Could not read {path} as UTF-8 text; the Config tab won't show it."
            break
        total += len(content.encode("utf-8"))
        files.append({"path": path, "content": content})
    if len(config_paths) > MAX_CONFIG_FILES:
        files = []
        warning = f"More than {MAX_CONFIG_FILES} config files; the Config tab won't show them."
    elif total > MAX_CONFIG_FILES_BYTES:
        files = []
        warning = (
            f"Config files are over {MAX_CONFIG_FILES_BYTES // 1024} KiB; "
            "the Config tab won't show them."
        )
    if any("\x00" in file["content"] for file in files):
        files = []
        warning = "A config file contains NUL bytes; the Config tab won't show it."

    snapshot: dict[str, Any] = {"files": files}
    command = launch_command(argv)
    if command is not None:
        snapshot["command"] = command
    return snapshot, warning
