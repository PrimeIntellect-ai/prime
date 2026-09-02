"""Keep known secret values out of uploads.

Redaction is exact-match only: a known value is replaced with `[REDACTED]` wherever it
appears inside a JSON string, and nothing is guessed from the shape of the text, so
ordinary content is never rewritten. Values come from the process environment
(variables whose names look like credentials), the Prime API key, and `--secret`
arguments. Local files are never modified.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 8
SECRET_NAME = re.compile(
    r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE|AUTHORIZATION|(?:^|[_-])AUTH(?:[_-]|$)",
    re.IGNORECASE,
)
"""Variable names whose values are credentials."""
JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def known_secrets(*values: Optional[str], secret_args: Iterable[str] = ()) -> set[str]:
    """Credential-named environment values, the given values, and `--secret` arguments
    (a literal, or the path of a file with one secret per line). Environment values
    shorter than `MIN_SECRET_LENGTH` are dropped — redacting them would rewrite ordinary
    text; explicit values are taken as given."""
    secrets = {
        value
        for name, value in os.environ.items()
        if SECRET_NAME.search(name) and len(value) >= MIN_SECRET_LENGTH
    }
    secrets.update(value for value in values if value)
    for arg in secret_args:
        path = Path(arg)
        lines = path.read_text().splitlines() if path.is_file() else [arg]
        secrets.update(line.strip() for line in lines)
    return {secret for secret in secrets if secret}


class Redactor:
    """Replaces every occurrence of the secrets inside JSON strings, counting hits."""

    def __init__(self, secrets: Iterable[str]) -> None:
        # The string itself is decoded before matching. A JSON document quoted inside it
        # (a tool result) is escaped once per nesting level, by an encoder that may also
        # escape `/`: match those spellings too.
        forms = set(secrets)
        forms |= {form.replace("/", r"\/") for form in forms}
        for _ in range(2):
            forms |= {
                json.dumps(form, ensure_ascii=escape)[1:-1]
                for form in list(forms)
                for escape in (True, False)
            }
        alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
        self.pattern = re.compile(alternatives) if forms else None
        self.count = 0

    def json(self, text: str) -> str:
        """Redact one JSON document (or JSONL line) given as text; structure and
        non-string values stay, and so do the bytes of every string without a hit."""
        pattern = self.pattern
        if pattern is None:
            return text

        def string(match: re.Match[str]) -> str:
            token = match.group(0)
            redacted, hits = pattern.subn(REDACTED, json.loads(token))
            if not hits:
                return token
            self.count += hits
            return json.dumps(redacted, ensure_ascii=token.isascii())

        return JSON_STRING.sub(string, text)

    def value(self, value: Any) -> Any:
        """Redact a JSON-compatible value (parsed `metadata.json`, results rows)."""
        return json.loads(self.json(json.dumps(value)))
