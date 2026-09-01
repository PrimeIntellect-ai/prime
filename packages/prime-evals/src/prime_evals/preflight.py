"""Credential scanning and reduction for data leaving the local machine."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

REDACTED = "[REDACTED]"

PLACEHOLDER = re.compile(
    r"^(?:\[?redacted(?:[_ -]?\d+)?\]?|masked|dummy|example|test|none|null|empty|changeme|"
    r"replace[_ -]?me|x{4,}|\*{4,}|<[^>]+>|\$\{?[A-Z][A-Z0-9_]*\}?)$",
    re.IGNORECASE,
)
SECRET_ENV = re.compile(
    r"(?:^|_)AUTH(?:$|_)|API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|"
    r"AUTHORIZATION|COOKIE|"
    r"PRIVATE_?KEY|CONNECTION_STRING|DATABASE_URL|REDIS_URL",
    re.IGNORECASE,
)
REFERENCE_SUFFIXES = ("_env", "_env_var", "_file", "_name", "_path", "_var", "_variable")
TELEMETRY_SUFFIXES = ("_count", "_counts", "_usage")
DESCRIPTOR_SUFFIXES = ("_method", "_scheme", "_type")
SENSITIVE_FIELDS = {
    "access_key",
    "access_key_id",
    "account_key",
    "api_key",
    "api_token",
    "access_token",
    "auth_token",
    "auth",
    "authentication",
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
SCHEMA_VALUES = {"const", "default", "enum", "example", "examples", "value", "values"}
SCHEMA_MARKERS = {
    "$defs",
    "$ref",
    "$schema",
    "allOf",
    "anyOf",
    "oneOf",
}
SCHEMA_CONTAINERS = {"json_schema", "schema"}
OPENAPI_MARKERS = {"openapi", "swagger"}
NAMED_DEFINITIONS = {
    "dependent_schemas",
    "defs",
    "definitions",
    "pattern_properties",
    "properties",
}
OPENAPI_COMPONENT_MAPS = {
    "callbacks",
    "examples",
    "headers",
    "links",
    "parameters",
    "path_items",
    "request_bodies",
    "responses",
    "schemas",
    "security_definitions",
    "security_schemes",
}
OPENAPI_ROOT_DEFINITIONS = {
    "definitions",
    "parameters",
    "responses",
    "security_definitions",
}
OPENAPI_SCHEMA_CONTAINERS = {"components", *OPENAPI_ROOT_DEFINITIONS}
HEADER_CONTAINER = re.compile(r"(?:^|_)headers?$")
SAFE_PATH_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
AUTH_VALUE = re.compile(r"^(?:bearer|basic|negotiate|token)\s+(.+)$", re.IGNORECASE)

# Every shape names only the credential. The same match drives reporting and redaction.
PATTERNS = (
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
        "authorization_header",
        re.compile(
            r"(?<![A-Za-z0-9_])\\?[\"']?(?:authorization|proxy-authorization)"
            r"\\?[\"']?\s*[:=]\s*(?P<quote>\\?[\"'])"
            r"(?P<secret>[^\r\n]*?)(?P=quote)",
            re.IGNORECASE,
        ),
    ),
    (
        "authorization_header",
        re.compile(
            r"(?<![A-Za-z0-9_])\\?[\"']?(?:authorization|proxy-authorization)"
            r"\\?[\"']?\s*[:=]\s*\\?[\"']?(?:(?:bearer|basic|token|negotiate)\s+)?"
            r"(?P<secret>[^\s,;\\\"']{8,})",
            re.IGNORECASE,
        ),
    ),
    (
        "authorization_header",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:authorization|proxy-authorization)\s*[:=]"
            r"(?!\s*(?:bearer|basic|negotiate|token)\s+)\s*"
            r"(?P<secret>[^\r\n]{8,})",
            re.IGNORECASE,
        ),
    ),
    (
        "cookie_header",
        re.compile(
            r"(?<![A-Za-z0-9_-])\\?[\"']?(?:set-)?cookie\\?[\"']?\s*:\s*"
            r"\\?[\"']?(?P<secret>[^\r\n\\\"']{8,})",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_assignment",
        re.compile(
            r"(?:(?<![A-Za-z0-9_])\\?[\"']?(?i:(?:[A-Za-z][A-Za-z0-9]*[_-]+)*"
            r"(?:x-api-key|api[_ -]?(?:key|token)|account[_ -]?key|"
            r"access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|session[_ -]?token|"
            r"client[_ -]?secret|secret[_ -]?access[_ -]?key|secret[_ -]?key|password|passwd|"
            r"cookie|credentials?|private[_ -]?key|sas[_ -]?token|signature))\b"
            r"\\?[\"']?\s*[:=]\s*|"
            r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_?KEY|API_?TOKEN|ACCESS_?TOKEN|"
            r"REFRESH_?TOKEN|AUTH_?TOKEN|SESSION_?TOKEN|TOKEN|CLIENT_?SECRET|"
            r"SECRET(?:_ACCESS_?KEY|_?KEY)?|PASSWORD|PASSWD|COOKIE|CREDENTIAL|"
            r"PRIVATE_?KEY|SAS_?TOKEN)\s*=\s*|"
            r"--(?:api-key|api-token|access-token|auth-token|client-secret|cookie|"
            r"password|private-key|sas-token|secret|token)(?:=|\s+))"
            r"(?:\\?\"(?P<secret_short_double>(?:\\.|[^\"\\\r\n]){8,})\\?\"|"
            r"\\?'(?P<secret_short_single>(?:\\.|[^'\\\r\n]){8,})\\?'|"
            r"(?:(?i:bearer|basic|token)\s+)?(?P<secret>[^\s,;\"']{8,}))"
        ),
    ),
    (
        "credential_assignment",
        re.compile(
            r"(?:(?<![A-Za-z0-9_])\\?[\"']?(?:[A-Za-z][A-Za-z0-9]*[_-]+)*"
            r"(?:x-api-key|api[_ -]?(?:key|token)|account[_ -]?key|"
            r"access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|client[_ -]?secret|"
            r"access[_ -]?key(?:[_ -]?id)?|"
            r"secret(?:[_ -]?(?:access[_ -]?)?key)?|"
            r"password|passwd|cookie|credentials?|private[_ -]?key|sas[_ -]?token|signature)\b"
            r"\\?[\"']?\s*[:=]\s*|"
            r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_?KEY|ACCESS_?TOKEN|REFRESH_?TOKEN|"
            r"AUTH_?TOKEN|SESSION_?TOKEN|TOKEN|CLIENT_?SECRET|SECRET(?:_ACCESS_?KEY)?|"
            r"PASSWORD|PASSWD|COOKIE|CREDENTIAL|PRIVATE_?KEY|SAS_?TOKEN|SECRET_?KEY)"
            r"\s*=\s*|"
            r"--(?:api-key|access-token|auth-token|client-secret|cookie|password|"
            r"private-key|sas-token|secret|token)(?:=|\s+))"
            r"(?:\\?\"(?P<secret_double>(?:\\.|[^\"\\\r\n]){16,})\\?\"|"
            r"\\?'(?P<secret_single>(?:\\.|[^'\\\r\n]){16,})\\?'|"
            r"(?:(?:bearer|basic|token)\s+)?(?P<secret>[^\s,;\"']{16,}))",
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
SecretFingerprint = tuple[int, str]


class ScanReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    findings: tuple[Finding, ...]

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    @property
    def locations(self) -> int:
        return len({finding[0] for finding in self.findings})

    @property
    def categories(self) -> dict[str, int]:
        return dict(sorted(Counter(finding[1] for finding in self.findings).items()))

    @property
    def description(self) -> str:
        categories = ", ".join(f"{name}: {count}" for name, count in self.categories.items())
        return f"{self.locations} credential-bearing location(s) ({categories})"


class PreparedUpload(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: Any
    report: ScanReport


class PreparedJSONLUpload(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    context: dict[str, str] | None
    report: ScanReport


class UploadScanError(ValueError):
    """The upload could not be safely reduced before network access."""


def normalize(name: str) -> str:
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


def is_sensitive(name: str) -> bool:
    name = normalize(name)
    singular = name.removesuffix("s")
    plural_secret = singular != name and any(
        singular == field or singular.endswith(f"_{field}")
        for field in SENSITIVE_FIELDS - {"token"}
    )
    names = (name, singular) if plural_secret else (name,)
    return not any(
        candidate.endswith(REFERENCE_SUFFIXES + TELEMETRY_SUFFIXES + DESCRIPTOR_SUFFIXES)
        for candidate in names
    ) and any(
        f"_{field}_" in f"_{candidate}_" or candidate.replace("_", "") == field.replace("_", "")
        for candidate in names
        for field in SENSITIVE_FIELDS
    )


def is_secret(value: Any) -> bool:
    return isinstance(value, str) and len(value) >= 8 and not PLACEHOLDER.fullmatch(value.strip())


def fingerprint_secret(secret: str) -> SecretFingerprint:
    """Return non-plaintext material for finding an echoed secret after resume."""
    if not is_secret(secret):
        raise ValueError("fingerprinted secrets must contain at least 8 non-placeholder characters")
    return len(secret), sha256(secret.encode(errors="surrogatepass")).hexdigest()


class SecretDiscovery:
    def __init__(
        self,
        known_secrets: Iterable[str] = (),
        secret_sources: Iterable[Mapping[str, str]] = (),
    ) -> None:
        self.secrets: dict[str, str] = {}
        for secret in known_secrets:
            self.remember(secret, "known_secret")
        for source in secret_sources:
            for name, secret in source.items():
                if is_sensitive(name):
                    self.remember(secret, "known_secret")

    def remember(self, value: Any, category: str, mapping_keys: bool = False) -> None:
        if is_secret(value):
            self.secrets.setdefault(value, category)
            match = AUTH_VALUE.fullmatch(value.strip())
            if match and is_secret(token := match.group(1)):
                self.secrets.setdefault(token, category)
        elif isinstance(value, Mapping):
            for key, child in value.items():
                if mapping_keys:
                    self.remember(key, category)
                self.remember(child, category)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self.remember(child, category)

    def discover(
        self,
        value: Any,
        schema_secret: bool = False,
        schema_context: bool = False,
        definitions: bool = False,
        headers: bool = False,
        components: bool = False,
        openapi: bool = False,
    ) -> None:
        if isinstance(value, Mapping):
            named_header = headers and is_sensitive(str(value.get("name") or value.get("key", "")))
            schema_type = value.get("type")
            openapi_document = openapi or any(field in value for field in OPENAPI_MARKERS)
            object_schema = (
                schema_context
                or any(field in value for field in SCHEMA_MARKERS)
                or "properties" in value
                and (
                    schema_type == "object"
                    or isinstance(schema_type, (list, tuple))
                    and "object" in schema_type
                )
                or schema_type == "array"
                and "items" in value
            )
            for key, child in value.items():
                name = str(key)
                normalized = normalize(name)
                if definitions:
                    if isinstance(child, (Mapping, list, tuple, bool)):
                        self.discover(child, is_sensitive(name), True, openapi=openapi_document)
                    else:
                        if is_sensitive(name):
                            self.remember(
                                child,
                                "structured_secret",
                                normalized.endswith(("_keys", "_secrets", "_tokens")),
                            )
                        self.discover(child, openapi=openapi_document)
                    continue
                if is_sensitive(name):
                    self.remember(
                        child,
                        "structured_secret",
                        normalized.endswith(("_keys", "_secrets", "_tokens")),
                    )
                if named_header and normalized in {"value", "values"}:
                    self.remember(child, "structured_secret")
                if schema_secret and normalized in SCHEMA_VALUES:
                    self.remember(child, "structured_secret")
                self.discover(
                    child,
                    schema_secret,
                    object_schema
                    or normalized in SCHEMA_CONTAINERS
                    or openapi_document
                    and normalized in OPENAPI_SCHEMA_CONTAINERS,
                    object_schema
                    and normalized in NAMED_DEFINITIONS
                    or components
                    and normalized in OPENAPI_COMPONENT_MAPS
                    or openapi_document
                    and normalized in OPENAPI_ROOT_DEFINITIONS
                    or openapi_document
                    and normalized == "security",
                    headers or HEADER_CONTAINER.search(normalized) is not None,
                    openapi_document and normalized == "components",
                    openapi_document,
                )
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                if (
                    headers
                    and index % 2 == 1
                    and isinstance(value[index - 1], str)
                    and is_sensitive(value[index - 1])
                ):
                    self.remember(child, "structured_secret")
                self.discover(
                    child,
                    schema_secret,
                    schema_context,
                    definitions,
                    headers,
                    components,
                    openapi,
                )
        elif isinstance(value, str):
            text = value.strip()
            if text.startswith(("{", "[")):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, (Mapping, list)):
                        self.discover(
                            parsed,
                            schema_secret,
                            schema_context,
                            definitions,
                            headers,
                            components,
                            openapi,
                        )
            for category, pattern in PATTERNS:
                if category == "credential_url":
                    continue
                for match in pattern.finditer(value):
                    secret = next(
                        matched
                        for name, matched in match.groupdict().items()
                        if name.startswith("secret") and matched is not None
                    )
                    if is_secret(secret):
                        self.secrets.setdefault(secret, category)


def secret_values(
    *values: str | None,
    secrets_file: str | Path | None = None,
    secret_sources: Iterable[Mapping[str, str]] = (),
) -> tuple[str, ...]:
    """Combine environment, configured, source-mapping, and optional file secrets."""
    candidates = [
        value
        for name, value in os.environ.items()
        if SECRET_ENV.search(name)
        and not normalize(name).endswith(REFERENCE_SUFFIXES + DESCRIPTOR_SUFFIXES)
        and is_secret(value)
    ]
    candidates.extend(values)
    if secrets_file:
        for line_number, line in enumerate(Path(secrets_file).read_text().splitlines(), 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if not is_secret(value):
                raise ValueError(
                    f"secret file line {line_number} must contain at least 8 "
                    "non-placeholder characters"
                )
            candidates.append(value)
    return tuple(SecretDiscovery(candidates, secret_sources).secrets)


def exact_pattern(secrets: Mapping[str, str]) -> re.Pattern[str] | None:
    if not secrets:
        return None
    alternatives = "|".join(re.escape(secret) for secret in sorted(secrets, key=len, reverse=True))
    return re.compile(f"(?P<secret>{alternatives})")


def finding_path(parts: tuple[str | int, ...]) -> str:
    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        elif part == "[key]" or not SAFE_PATH_KEY.fullmatch(part):
            result += ".[key]"
        else:
            result += f".{part}"
    return result


class CredentialReducer:
    def __init__(
        self,
        secrets: Mapping[str, str],
        exact: re.Pattern[str] | None = None,
        secret_fingerprints: Iterable[SecretFingerprint] = (),
    ) -> None:
        self.secrets = secrets
        self.exact = exact or exact_pattern(secrets)
        self.fingerprints: dict[int, set[str]] = {}
        for length, digest in secret_fingerprints:
            self.fingerprints.setdefault(length, set()).add(digest)
        self.categories: set[str] = set()
        self.pattern_category = ""

    def replace_exact(self, match: re.Match[str]) -> str:
        self.categories.add(self.secrets[match.group("secret")])
        return REDACTED

    def replace_shape(self, match: re.Match[str]) -> str:
        group = next(
            name
            for name, value in match.groupdict().items()
            if name.startswith("secret") and value is not None
        )
        secret = match.group(group)
        if not is_secret(secret):
            return match.group(0)
        self.categories.add(self.pattern_category)
        start, end = match.span(group)
        offset = match.start()
        return f"{match.group(0)[: start - offset]}{REDACTED}{match.group(0)[end - offset :]}"

    def find_fingerprinted(self, text: str) -> set[str]:
        fingerprinted = set()
        for length, digests in self.fingerprints.items():
            for start in range(len(text) - length + 1):
                candidate = text[start : start + length]
                if sha256(candidate.encode(errors="surrogatepass")).hexdigest() in digests:
                    fingerprinted.add(candidate)
        return fingerprinted

    def redact_text(
        self,
        text: str,
        schema_secret: bool = False,
        schema_context: bool = False,
        definitions: bool = False,
        headers: bool = False,
        components: bool = False,
        openapi: bool = False,
    ) -> tuple[str, set[str]]:
        self.categories = set()
        stripped = text.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, (Mapping, list)):
                    parsed_findings: set[Finding] = set()
                    parsed = self.reduce(
                        parsed,
                        parsed_findings,
                        schema_secret=schema_secret,
                        schema_context=schema_context,
                        definitions=definitions,
                        headers=headers,
                        components=components,
                        openapi=openapi,
                    )
                    if parsed_findings:
                        start = len(text) - len(text.lstrip())
                        end = len(text.rstrip())
                        serialized = json.dumps(parsed, separators=(",", ":"))
                        text = f"{text[:start]}{serialized}{text[end:]}"
                        self.categories.update(finding[1] for finding in parsed_findings)
        if self.exact:
            text = self.exact.sub(self.replace_exact, text)
        fingerprinted = self.find_fingerprinted(text)
        if fingerprinted:
            self.categories.add("known_secret")
        for secret in sorted(fingerprinted, key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        for category, pattern in PATTERNS:
            self.pattern_category = category
            text = pattern.sub(self.replace_shape, text)
        return text, self.categories

    def reduce(
        self,
        value: Any,
        findings: set[Finding],
        path: tuple[str | int, ...] = (),
        structured_secret: bool = False,
        schema_secret: bool = False,
        schema_context: bool = False,
        definitions: bool = False,
        headers: bool = False,
        components: bool = False,
        openapi: bool = False,
    ) -> Any:
        if isinstance(value, Mapping):
            reduced = {}
            named_header = headers and is_sensitive(str(value.get("name") or value.get("key", "")))
            schema_type = value.get("type")
            openapi_document = openapi or any(field in value for field in OPENAPI_MARKERS)
            object_schema = (
                schema_context
                or any(field in value for field in SCHEMA_MARKERS)
                or "properties" in value
                and (
                    schema_type == "object"
                    or isinstance(schema_type, (list, tuple))
                    and "object" in schema_type
                )
                or schema_type == "array"
                and "items" in value
            )
            for key, child in value.items():
                safe_key, categories = (
                    (REDACTED, {"structured_secret"})
                    if structured_secret and is_secret(key)
                    else self.redact_text(key)
                    if isinstance(key, str)
                    else (key, set())
                )
                findings.update(
                    (finding_path((*path, "[key]")), category) for category in categories
                )
                if structured_secret and safe_key == REDACTED:
                    suffix = 2
                    while safe_key in reduced:
                        safe_key = f"[REDACTED {suffix}]"
                        suffix += 1
                if safe_key in reduced:
                    raise UploadScanError("credential reduction would create duplicate object keys")
                normalized = normalize(str(key))
                definition_schema = definitions and isinstance(child, (Mapping, list, tuple, bool))
                telemetry = bool(path) and normalize(str(path[-1])) in {"metrics", "rewards"}
                child_secret = structured_secret or (
                    not definition_schema
                    and not (telemetry and isinstance(child, (int, float)))
                    and (
                        is_sensitive(str(key))
                        or named_header
                        and normalized in {"value", "values"}
                        or schema_secret
                        and normalized in SCHEMA_VALUES
                    )
                )
                reduced[safe_key] = self.reduce(
                    child,
                    findings,
                    (*path, "[key]" if categories else str(key)),
                    child_secret,
                    is_sensitive(str(key)) if definition_schema else schema_secret,
                    definition_schema
                    or object_schema
                    or normalized in SCHEMA_CONTAINERS
                    or openapi_document
                    and normalized in OPENAPI_SCHEMA_CONTAINERS,
                    not structured_secret
                    and (
                        object_schema
                        and normalized in NAMED_DEFINITIONS
                        or components
                        and normalized in OPENAPI_COMPONENT_MAPS
                        or openapi_document
                        and normalized in OPENAPI_ROOT_DEFINITIONS
                        or openapi_document
                        and normalized == "security"
                    ),
                    headers or HEADER_CONTAINER.search(normalized) is not None,
                    openapi_document and normalized == "components",
                    openapi_document,
                )
            return reduced
        if isinstance(value, (list, tuple)):
            reduced = [
                self.reduce(
                    child,
                    findings,
                    (*path, index),
                    structured_secret
                    or headers
                    and index % 2 == 1
                    and isinstance(value[index - 1], str)
                    and is_sensitive(value[index - 1]),
                    schema_secret,
                    schema_context,
                    definitions,
                    headers,
                    components,
                    openapi,
                )
                for index, child in enumerate(value)
            ]
            return tuple(reduced) if isinstance(value, tuple) else reduced
        if (
            structured_secret
            and value is not None
            and not isinstance(value, bool)
            and (
                not isinstance(value, str)
                or value.strip()
                and not PLACEHOLDER.fullmatch(value.strip())
            )
        ):
            findings.add((finding_path(path), "structured_secret"))
            return REDACTED
        if not isinstance(value, str):
            return value
        reduced, categories = self.redact_text(
            value,
            schema_secret,
            schema_context,
            definitions,
            headers,
            components,
            openapi,
        )
        findings.update((finding_path(path), category) for category in categories)
        return reduced

    def prepare(self, value: Any) -> PreparedUpload:
        findings: set[Finding] = set()
        data = self.reduce(value, findings)
        residual: set[Finding] = set()
        self.reduce(data, residual)
        if residual:
            paths = ", ".join(path for path, category in sorted(residual)[:10])
            raise UploadScanError(f"reduced upload still contains credentials at {paths}")
        return PreparedUpload(data=data, report=ScanReport(findings=tuple(sorted(findings))))


def scan_upload(
    value: Any,
    known_secrets: Iterable[str] = (),
    secret_sources: Iterable[Mapping[str, str]] = (),
    secret_fingerprints: Iterable[SecretFingerprint] = (),
) -> ScanReport:
    """Classify credential locations without returning their values."""
    discovery = SecretDiscovery(known_secrets, secret_sources)
    discovery.discover(value)
    return (
        CredentialReducer(discovery.secrets, secret_fingerprints=secret_fingerprints)
        .prepare(value)
        .report
    )


def prepare_upload(
    value: Any,
    known_secrets: Iterable[str] = (),
    secret_sources: Iterable[Mapping[str, str]] = (),
    secret_fingerprints: Iterable[SecretFingerprint] = (),
) -> PreparedUpload:
    """Reduce a copy, then rescan it before any caller starts an upload."""
    discovery = SecretDiscovery(known_secrets, secret_sources)
    discovery.discover(value)
    return CredentialReducer(discovery.secrets, secret_fingerprints=secret_fingerprints).prepare(
        value
    )


class DuplicateKeyError(ValueError):
    pass


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, child in pairs:
        if key in value:
            raise DuplicateKeyError
        value[key] = child
    return value


def load_line(line: str, number: int) -> Any:
    try:
        return json.loads(line, object_pairs_hook=unique_object)
    except DuplicateKeyError as error:
        raise UploadScanError(f"duplicate object key on JSONL line {number}") from error
    except json.JSONDecodeError as error:
        raise UploadScanError(
            f"invalid JSON on JSONL line {number} at column {error.colno}"
        ) from error


def prefix_findings(report: ScanReport, prefix: str) -> set[Finding]:
    return {(f"{prefix}{path[1:]}", category) for path, category in report.findings}


def prepare_jsonl_upload(
    source: str | Path,
    destination: str | Path,
    *,
    context: Mapping[str, str] | None = None,
    known_secrets: Iterable[str] = (),
    secret_sources: Iterable[Mapping[str, str]] = (),
    secret_fingerprints: Iterable[SecretFingerprint] = (),
) -> PreparedJSONLUpload:
    """Return a safe snapshot or redacted JSONL path without changing the source."""
    source, snapshot = Path(source), Path(destination)
    redacted = snapshot.with_name(f"redacted-{uuid4().hex}-{snapshot.name}")
    outputs = (snapshot, redacted)
    if (
        source.resolve() in {output.resolve() for output in outputs}
        or source.exists()
        and any(output.exists() and source.samefile(output) for output in outputs)
    ):
        raise UploadScanError("upload outputs must not alias the source JSONL")
    discovery = SecretDiscovery(known_secrets, secret_sources)
    discovery.discover(context)

    snapshot_started = False
    validated = False
    try:
        with source.open("rb") as input_file:
            descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            snapshot_started = True
            with os.fdopen(descriptor, "wb") as output_file:
                os.chmod(snapshot, 0o600)
                for number, raw in enumerate(input_file, 1):
                    output_file.write(raw)
                    if not raw.isspace():
                        discovery.discover(load_line(raw.decode(), number))
        validated = True
    except UnicodeDecodeError as error:
        raise UploadScanError("trace JSONL must be UTF-8") from error
    finally:
        if snapshot_started and not validated:
            snapshot.unlink(missing_ok=True)

    try:
        reducer = CredentialReducer(discovery.secrets, secret_fingerprints=secret_fingerprints)
        context_prepared = reducer.prepare(context) if context is not None else None
        findings = (
            prefix_findings(context_prepared.report, "$.context") if context_prepared else set()
        )
        file_findings: set[Finding] = set()
        output_file = None
        with snapshot.open("rb") as input_file, ExitStack() as stack:
            offset = 0
            number = 0
            while raw := input_file.readline():
                number += 1
                if raw.isspace():
                    if output_file:
                        output_file.write(raw)
                    offset += len(raw)
                    continue
                value = reducer.prepare(load_line(raw.decode(), number))
                file_findings.update(prefix_findings(value.report, f"$.lines[{number}]"))
                if value.report.has_findings and output_file is None:
                    descriptor = os.open(redacted, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    output_file = stack.enter_context(os.fdopen(descriptor, "wb"))
                    os.chmod(redacted, 0o600)
                    with snapshot.open("rb") as prefix:
                        output_file.write(prefix.read(offset))
                if output_file:
                    output_file.write(
                        json.dumps(value.data, separators=(",", ":")).encode() + b"\n"
                        if value.report.has_findings
                        else raw
                    )
                offset += len(raw)
        if output_file:
            snapshot.unlink()
            upload_path = redacted
        else:
            upload_path = snapshot
        return PreparedJSONLUpload(
            path=upload_path,
            context=context_prepared.data if context_prepared else None,
            report=ScanReport(findings=tuple(sorted(findings | file_findings))),
        )
    except BaseException:
        for output in outputs:
            output.unlink(missing_ok=True)
        raise
