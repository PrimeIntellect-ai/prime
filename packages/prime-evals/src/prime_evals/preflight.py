"""Credential scanning and reduction for data leaving the local machine."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REDACTED = "[REDACTED]"

_PLACEHOLDER = re.compile(
    r"^(?:\[?redacted\]?|masked|dummy|example|test|none|null|empty|changeme|"
    r"replace[_ -]?me|x{4,}|\*{4,}|<[^>]+>|\$\{?[A-Z][A-Z0-9_]*\}?)$",
    re.IGNORECASE,
)
_SECRET_ENV = re.compile(
    r"API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION|COOKIE|"
    r"PRIVATE_?KEY|CONNECTION_STRING|DATABASE_URL|REDIS_URL",
    re.IGNORECASE,
)
_REFERENCE_SUFFIXES = ("_env", "_env_var", "_file", "_name", "_path", "_var", "_variable")
_SENSITIVE_FIELDS = {
    "access_key",
    "access_key_id",
    "api_key",
    "access_token",
    "auth_token",
    "authorization",
    "client_secret",
    "connection_string",
    "cookie",
    "credential",
    "database_url",
    "password",
    "passwd",
    "private_key",
    "proxy_authorization",
    "redis_url",
    "refresh_token",
    "sas_token",
    "secret",
    "secret_access_key",
    "secret_key",
    "session_token",
    "signature",
    "token",
}
_SCHEMA_VALUES = {"const", "default", "example", "examples"}
_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
_SAFE_PATH_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
_AUTH_VALUE = re.compile(r"^(?:bearer|basic)\s+(.+)$", re.IGNORECASE)

# Every shape names only the credential. The same match drives reporting and redaction.
_PATTERNS = (
    (
        "private_key",
        re.compile(
            r"(?P<secret>-----BEGIN (?P<label>(?:(?:RSA|EC|OPENSSH|DSA|ENCRYPTED) )?"
            r"PRIVATE KEY)-----.*?-----END (?P=label)-----)",
            re.DOTALL,
        ),
    ),
    (
        "provider_credential",
        re.compile(
            r"(?<![A-Za-z0-9])(?P<secret>"
            r"sk-(?:ant-|or-v1-)?[A-Za-z0-9_-]{20,}|"
            r"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|"
            r"AIza[0-9A-Za-z_-]{30,}|"
            r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
            r"hf_[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}|"
            r"glpat-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
            r"x(?:wfp|app)-[A-Za-z0-9-]{10,}|(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}|"
            r"npm_[A-Za-z0-9]{20,}|pypi-[A-Za-z0-9_-]{30,}|"
            r"SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,})"
        ),
    ),
    (
        "webhook_credential",
        re.compile(
            r"(?P<secret>https://(?:hooks\.slack\.com/services|"
            r"discord(?:app)?\.com/api/webhooks)/[^\s\"']+)",
            re.IGNORECASE,
        ),
    ),
    (
        "cookie_header",
        re.compile(r"^\s*cookie\s*:\s*(?P<secret>[^\r\n]{8,})", re.IGNORECASE | re.MULTILINE),
    ),
    (
        "credential_assignment",
        re.compile(
            r"(?:(?<![A-Za-z0-9_])[\"']?(?:authorization|proxy-authorization|"
            r"x-api-key|api[_ -]?key|"
            r"access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|client[_ -]?secret|"
            r"secret|password|passwd|cookie|private[_ -]?key|signature)\b[\"']?\s*[:=]\s*|"
            r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_?KEY|ACCESS_?TOKEN|REFRESH_?TOKEN|"
            r"AUTH_?TOKEN|SESSION_?TOKEN|TOKEN|CLIENT_?SECRET|SECRET(?:_ACCESS_?KEY)?|"
            r"PASSWORD|PASSWD|CREDENTIAL|PRIVATE_?KEY)\s*=\s*|"
            r"--(?:api-key|access-token|auth-token|client-secret|password|private-key|"
            r"secret|token)(?:=|\s+))[\"']?(?:(?:bearer|basic)\s+)?[\"']?"
            r"(?P<secret>[^\s,;\"']{16,})",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_url",
        re.compile(
            r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^:/@\s]+:)?"
            r"(?P<secret>[^/@\s]{8,})@|"
            r"[?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|sig|"
            r"signature|credential|authorization|auth|password)="
            r"(?P<secret_query>[^&#\s\"']{8,})",
            re.IGNORECASE,
        ),
    ),
)

Finding = tuple[str, str]


@dataclass(frozen=True)
class ScanReport:
    findings: tuple[Finding, ...]

    def __bool__(self) -> bool:
        return bool(self.findings)

    @property
    def locations(self) -> int:
        return len({path for path, _ in self.findings})

    @property
    def categories(self) -> dict[str, int]:
        return dict(sorted(Counter(category for _, category in self.findings).items()))

    def __str__(self) -> str:
        categories = ", ".join(f"{name}: {count}" for name, count in self.categories.items())
        return f"{self.locations} credential-bearing location(s) ({categories})"


@dataclass(frozen=True)
class PreparedUpload:
    data: Any
    report: ScanReport


@dataclass(frozen=True)
class PreparedJSONLUpload:
    path: Path
    context: dict[str, str] | None
    report: ScanReport


class UploadScanError(ValueError):
    """The upload could not be safely reduced before network access."""


def _normalize(name: str) -> str:
    return (
        re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
                "_",
                name,
            ),
        )
        .strip("_")
        .lower()
    )


def _sensitive(name: str) -> bool:
    name = _normalize(name)
    names = (name, name.removesuffix("s"))
    return not any(candidate.endswith(_REFERENCE_SUFFIXES) for candidate in names) and any(
        candidate == field or candidate.endswith(f"_{field}")
        for candidate in names
        for field in _SENSITIVE_FIELDS
    )


def _is_secret(value: Any) -> bool:
    return isinstance(value, str) and len(value) >= 8 and not _PLACEHOLDER.fullmatch(value.strip())


def secret_values(*values: str | None, secrets_file: str | Path | None = None) -> tuple[str, ...]:
    """Combine environment, configured, and optional file secrets."""
    candidates = [
        value
        for name, value in os.environ.items()
        if _SECRET_ENV.search(name) and _is_secret(value)
    ]
    candidates.extend(values)
    if secrets_file:
        for line_number, line in enumerate(Path(secrets_file).read_text().splitlines(), 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if not _is_secret(value):
                raise ValueError(
                    f"secret file line {line_number} must contain at least 8 "
                    "non-placeholder characters"
                )
            candidates.append(value)
    return tuple(dict.fromkeys(value for value in candidates if _is_secret(value)))


def _remember(value: Any, secrets: dict[str, str], category: str) -> None:
    if isinstance(value, str) and bool(value.strip()) and not _PLACEHOLDER.fullmatch(value.strip()):
        secrets.setdefault(value, category)
        if match := _AUTH_VALUE.fullmatch(value.strip()):
            _remember(match.group(1), secrets, category)
    elif isinstance(value, Mapping):
        for child in value.values():
            _remember(child, secrets, category)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _remember(child, secrets, category)


def _discover(value: Any, secrets: dict[str, str]) -> None:
    def visit(child: Any, schema_secret: bool = False, properties: bool = False) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                name = str(key)
                normalized = _normalize(name)
                if properties:
                    schema = isinstance(nested, Mapping) and (
                        nested.get("type") in _SCHEMA_TYPES
                        or any(field in nested for field in ("$ref", "allOf", "anyOf", "oneOf"))
                    )
                    if schema:
                        visit(nested, _sensitive(name))
                    else:
                        if _sensitive(name):
                            _remember(nested, secrets, "structured_secret")
                        visit(nested)
                    continue
                if _sensitive(name):
                    _remember(nested, secrets, "structured_secret")
                if schema_secret and normalized in _SCHEMA_VALUES:
                    _remember(nested, secrets, "structured_secret")
                visit(nested, schema_secret, normalized == "properties")
        elif isinstance(child, (list, tuple)):
            for nested in child:
                visit(nested, schema_secret)

    visit(value)


def _secrets(value: Any, known_secrets: Iterable[str]) -> dict[str, str]:
    secrets: dict[str, str] = {}
    for secret in known_secrets:
        _remember(secret, secrets, "known_secret")
    _discover(value, secrets)
    return secrets


def _exact_pattern(secrets: Mapping[str, str]) -> re.Pattern[str] | None:
    if not secrets:
        return None
    alternatives = "|".join(re.escape(secret) for secret in sorted(secrets, key=len, reverse=True))
    return re.compile(f"(?P<secret>{alternatives})")


def _path(parts: tuple[str | int, ...]) -> str:
    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        elif part == "[key]" or not _SAFE_PATH_KEY.fullmatch(part):
            result += ".[key]"
        else:
            result += f".{part}"
    return result


def _redact_text(text: str, secrets: Mapping[str, str]) -> tuple[str, set[str]]:
    categories = set()
    if exact := _exact_pattern(secrets):

        def replace_exact(match: re.Match[str]) -> str:
            categories.add(secrets[match.group("secret")])
            return REDACTED

        text = exact.sub(replace_exact, text)
    for category, pattern in _PATTERNS:

        def replace_shape(match: re.Match[str], category: str = category) -> str:
            group = "secret" if match.groupdict().get("secret") is not None else "secret_query"
            secret = match.group(group)
            if not _is_secret(secret):
                return match.group(0)
            categories.add(category)
            start, end = match.span(group)
            offset = match.start()
            return f"{match.group(0)[: start - offset]}{REDACTED}{match.group(0)[end - offset :]}"

        text = pattern.sub(replace_shape, text)
    return text, categories


def _reduce(
    value: Any,
    secrets: Mapping[str, str],
    findings: set[Finding],
    path: tuple[str | int, ...] = (),
) -> Any:
    if isinstance(value, Mapping):
        reduced = {}
        for key, child in value.items():
            safe_key, categories = (
                _redact_text(key, secrets) if isinstance(key, str) else (key, set())
            )
            findings.update((_path((*path, "[key]")), category) for category in categories)
            if safe_key in reduced:
                raise UploadScanError("credential reduction would create duplicate object keys")
            reduced[safe_key] = _reduce(
                child, secrets, findings, (*path, "[key]" if categories else str(key))
            )
        return reduced
    if isinstance(value, list):
        return [
            _reduce(child, secrets, findings, (*path, index)) for index, child in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _reduce(child, secrets, findings, (*path, index)) for index, child in enumerate(value)
        )
    if not isinstance(value, str):
        return value
    reduced, categories = _redact_text(value, secrets)
    findings.update((_path(path), category) for category in categories)
    return reduced


def _prepare(value: Any, secrets: Mapping[str, str]) -> PreparedUpload:
    findings: set[Finding] = set()
    data = _reduce(value, secrets, findings)
    residual: set[Finding] = set()
    _reduce(data, secrets, residual)
    if residual:
        paths = ", ".join(path for path, _ in sorted(residual)[:10])
        raise UploadScanError(f"reduced upload still contains credentials at {paths}")
    return PreparedUpload(data, ScanReport(tuple(sorted(findings))))


def scan_upload(value: Any, known_secrets: Iterable[str] = ()) -> ScanReport:
    """Classify credential locations without returning their values."""
    return _prepare(value, _secrets(value, known_secrets)).report


def prepare_upload(value: Any, known_secrets: Iterable[str] = ()) -> PreparedUpload:
    """Reduce a copy, then rescan it before any caller starts an upload."""
    return _prepare(value, _secrets(value, known_secrets))


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, child in pairs:
        if key in value:
            raise _DuplicateKeyError
        value[key] = child
    return value


def _load_line(line: str, number: int) -> Any:
    try:
        return json.loads(line, object_pairs_hook=_unique_object)
    except _DuplicateKeyError as error:
        raise UploadScanError(f"duplicate object key on JSONL line {number}") from error
    except json.JSONDecodeError as error:
        raise UploadScanError(
            f"invalid JSON on JSONL line {number} at column {error.colno}"
        ) from error


def _prefix(report: ScanReport, prefix: str) -> set[Finding]:
    return {(f"{prefix}{path[1:]}", category) for path, category in report.findings}


def prepare_jsonl_upload(
    source: str | Path,
    destination: str | Path,
    *,
    context: Mapping[str, str] | None = None,
    known_secrets: Iterable[str] = (),
) -> PreparedJSONLUpload:
    """Return a safe snapshot or redacted JSONL path without changing the source."""
    source, snapshot = Path(source), Path(destination)
    redacted = snapshot.with_name(f"redacted-{snapshot.name}")
    if source.resolve() in {snapshot.resolve(), redacted.resolve()}:
        raise UploadScanError("upload outputs must not alias the source JSONL")
    secrets: dict[str, str] = {}
    for secret in known_secrets:
        _remember(secret, secrets, "known_secret")
    _discover(context, secrets)

    try:
        with source.open("rb") as input_file, snapshot.open("wb") as output_file:
            for number, raw in enumerate(input_file, 1):
                output_file.write(raw)
                if not raw.isspace():
                    _discover(_load_line(raw.decode(), number), secrets)
    except UnicodeDecodeError as error:
        raise UploadScanError("trace JSONL must be UTF-8") from error

    context_prepared = _prepare(context, secrets) if context is not None else None
    findings = _prefix(context_prepared.report, "$.context") if context_prepared else set()
    file_findings: set[Finding] = set()

    try:
        with snapshot.open("rb") as input_file, redacted.open("wb") as output_file:
            for number, raw in enumerate(input_file, 1):
                if raw.isspace():
                    output_file.write(raw)
                    continue
                value = _prepare(_load_line(raw.decode(), number), secrets)
                file_findings.update(_prefix(value.report, f"$.lines[{number}]"))
                output_file.write(
                    json.dumps(value.data, ensure_ascii=False, separators=(",", ":")).encode()
                    + b"\n"
                    if value.report
                    else raw
                )
    except UnicodeDecodeError as error:
        raise UploadScanError("trace JSONL must be UTF-8") from error

    if file_findings:
        snapshot.unlink()
        upload_path = redacted
    else:
        redacted.unlink()
        upload_path = snapshot
    return PreparedJSONLUpload(
        upload_path,
        context_prepared.data if context_prepared else None,
        ScanReport(tuple(sorted(findings | file_findings))),
    )
