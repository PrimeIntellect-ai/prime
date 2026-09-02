"""Keep known secret values out of uploads.

Redaction is exact-match only: a known value is replaced with `[REDACTED]` wherever it
appears inside a JSON string, and nothing is guessed from the shape of the text, so
ordinary content is never rewritten. Values come from the process environment
(credential-like variable names, and the password inside connection URLs), the Prime
API key, and `--secret` arguments. Local files are never modified.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional
from urllib.parse import unquote, unquote_plus, urlsplit

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 8
SECRET_NAME = re.compile(
    r"(?:(?<![A-Za-z0-9])(?:API_?KEYS?|KEYS?|CREDENTIALS?|COOKIES?|AUTHORIZATION|AUTH)"
    r"|TOKENS?|SECRETS?|PASSW(?:OR)?DS?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
"""Variable names whose values are credentials: a credential word as a whole segment
(`HF_TOKEN`, `X-Api-Key`), so `KEYCLOAK_REALM`, `OAUTH_CLIENT_ID`, or
`TOKENIZERS_PARALLELISM` is not one. No other word ends in TOKEN, SECRET, or PASSWORD,
so those may also end a segment (`PGPASSWORD`, `ACCESSTOKEN`)."""
JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def url_credentials(value: str) -> Iterator[str]:
    """The credentials inside a URL, as written and percent-decoded the way a client uses
    them: the password — or the bare user token — of `scheme://user:password@host`, and
    credential-named query values (`?token=…`). A username next to a password is a name,
    not a secret (`postgres`), so a token placed there beside a dummy password (GitHub's
    legacy `token:x-oauth-basic`) is not recognised. Prose is never a URL here: the
    value must start with `scheme://host`."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return
    if not (parts.scheme and parts.netloc):
        return
    if "@" in parts.netloc:
        # With a password slot, even an empty one, the username is a name, not a token.
        userinfo = parts.username if parts.password is None else parts.password
        if userinfo:
            yield from {userinfo, unquote(userinfo)}
    for pair in parts.query.split("&"):
        name, _, raw = pair.partition("=")
        if raw and SECRET_NAME.search(unquote(name)):
            yield from {raw, unquote(raw), unquote_plus(raw)}


def overlaps_marker(secret: str) -> bool:
    """Whether replacing with the marker could leave or form `secret`: it sits inside
    `[REDACTED]`, or begins with the marker's tail (`]bar`) or ends with its head
    (`foo[`). Such a value is a placeholder or a pathological input, never a credential."""
    return secret in REDACTED or any(
        secret.startswith(REDACTED[-n:]) or secret.endswith(REDACTED[:n])
        for n in range(1, len(REDACTED) + 1)
    )


def env_credentials(mapping: Mapping[str, object]) -> Iterator[str]:
    """The credentials in an environment-like mapping: every value under a credential-like
    name, and the URL credentials in any value whatever its name (`DATABASE_URL`,
    `HTTP_PROXY`)."""
    for name, value in mapping.items():
        if not isinstance(value, str):
            continue
        if SECRET_NAME.search(name):
            yield value
        yield from url_credentials(value)


def known_secrets(*values: Optional[str], secret_args: Iterable[str] = ()) -> set[str]:
    """The process environment's credentials (`env_credentials`), the given values, and
    `--secret` arguments (a literal, or the path of a file with one secret per line).
    Environment values shorter than `MIN_SECRET_LENGTH` are dropped — redacting them
    would rewrite ordinary text; explicit values are taken as given."""
    discovered = {
        credential
        for credential in env_credentials(os.environ)
        if len(credential) >= MIN_SECRET_LENGTH
    }
    discovered.update(value for value in values if value)
    explicit: set[str] = set()
    for arg in secret_args:
        # Taken as given: only a file entry's line terminator goes.
        explicit.update(Path(arg).read_text().splitlines() if os.path.isfile(arg) else [arg])
    explicit.discard("")
    # No fixed marker can hide a value that overlaps it. A discovered one is a sanitized
    # placeholder (`API_TOKEN=REDACTED`) and is skipped; a requested one is refused.
    if colliding := sorted(secret for secret in explicit if overlaps_marker(secret)):
        raise ValueError(f"cannot redact {colliding}: overlaps the {REDACTED} marker")
    return {secret for secret in discovered if not overlaps_marker(secret)} | explicit


class Redactor:
    """Replaces every occurrence of the secrets inside JSON strings, counting hits."""

    def __init__(self, secrets: Iterable[str]) -> None:
        secrets = set(secrets)
        alternatives = "|".join(
            re.escape(secret) for secret in sorted(secrets, key=len, reverse=True)
        )
        self.pattern = re.compile(alternatives) if secrets else None
        self.count = 0

    def json(self, text: str) -> str:
        """Redact one JSON document (or JSONL line) given as text; structure and
        non-string values stay, and so do the bytes of every string without a hit. Each
        string is decoded before matching and searched again for quoted JSON inside it
        (a tool result), so every escape spelling at every nesting depth is matched."""
        pattern = self.pattern
        if pattern is None:
            return text

        def string(match: re.Match[str]) -> str:
            token = match.group(0)
            try:
                value = json.loads(token)
            except ValueError:  # quotes in prose, not a JSON string
                return token
            redacted, hits = pattern.subn(REDACTED, self.json(value))
            if redacted == value:
                return token
            self.count += hits
            return json.dumps(redacted, ensure_ascii=token.isascii())

        return JSON_STRING.sub(string, text)

    def value(self, value: Any) -> Any:
        """Redact a JSON-compatible value (parsed `metadata.json`, results rows)."""
        return json.loads(self.json(json.dumps(value)))
