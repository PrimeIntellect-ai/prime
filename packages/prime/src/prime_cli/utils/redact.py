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
from urllib.parse import unquote, urlsplit

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 8
SECRET_NAME = re.compile(
    r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE|AUTHORIZATION|(?:^|[_-])AUTH(?:[_-]|$)",
    re.IGNORECASE,
)
"""Variable names whose values are credentials."""
JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def env_credentials(mapping: Mapping[str, object]) -> Iterator[str]:
    """The credentials in an environment-like mapping: every value under a credential-like
    name, and the password — or the bare user token — inside a `scheme://user:password@host`
    value whatever its name (`DATABASE_URL`, `HTTP_PROXY`), as written and percent-decoded
    the way a client uses it. A username next to a password
    is a name, not a secret (`postgres`), so a token placed there beside a dummy password
    (GitHub's legacy `token:x-oauth-basic`) is not recognised."""
    for name, value in mapping.items():
        if not isinstance(value, str):
            continue
        if SECRET_NAME.search(name):
            yield value
        try:
            parts = urlsplit(value)
        except ValueError:
            continue
        if "@" not in parts.netloc:
            continue
        # With a password slot, even an empty one, the username is a name, not a token.
        userinfo = parts.username if parts.password is None else parts.password
        if userinfo:
            yield from {userinfo, unquote(userinfo)}


def known_secrets(*values: Optional[str], secret_args: Iterable[str] = ()) -> set[str]:
    """The process environment's credentials (`env_credentials`), the given values, and
    `--secret` arguments (a literal, or the path of a file with one secret per line).
    Environment values shorter than `MIN_SECRET_LENGTH` are dropped — redacting them
    would rewrite ordinary text; explicit values are taken as given."""
    secrets = {
        credential
        for credential in env_credentials(os.environ)
        if len(credential) >= MIN_SECRET_LENGTH
    }
    secrets.update(value for value in values if value)
    for arg in secret_args:
        # `os.path.isfile` is False for anything that cannot be a path, a long literal included.
        lines = Path(arg).read_text().splitlines() if os.path.isfile(arg) else [arg]
        secrets.update(line.strip() for line in lines)
    return {secret for secret in secrets if secret}


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
