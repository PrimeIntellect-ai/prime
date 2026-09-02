"""Exact-match redaction: a known value disappears from JSON strings in every spelling
(plain, escaped, escaped again inside quoted JSON documents at any depth, with the `\\/`
and uppercase-hex escapes other encoders emit) and nothing else changes."""

import json

from prime_cli.utils.redact import Redactor, known_secrets


def test_redactor_replaces_every_spelling_inside_strings_only():
    redactor = Redactor({'pa"ss\\word-0001', "12345678", "ünïcode-secret", "hooks/abc/def"})
    doc = {
        "plain": 'pa"ss\\word-0001 and ünïcode-secret',
        "nested": json.dumps({"k": 'pa"ss\\word-0001', "u": "ünïcode-secret"}),
        "slashes": '{"url": "hooks\\/abc\\/def"}',  # a JS/PHP-style encoder escapes `/`
        "upper": '{"u": "\\u00FCn\\u00EFcode-secret"}',  # uppercase hex escapes
        "deep": json.dumps({"log": json.dumps({"k": 'pa"ss\\word-0001', "n": 1})}),
        "number": 12345678,  # a JSON number is not a string: untouched
        "text": "12345678",
        "keep": "ordinary text",
    }
    for ensure_ascii in (True, False):
        redactor.count = 0
        out = json.loads(redactor.json(json.dumps(doc, ensure_ascii=ensure_ascii)))
        assert out == {
            "plain": "[REDACTED] and [REDACTED]",
            "nested": json.dumps({"k": "[REDACTED]", "u": "[REDACTED]"}),
            "slashes": '{"url": "[REDACTED]"}',
            "upper": '{"u": "[REDACTED]"}',
            "deep": json.dumps({"log": json.dumps({"k": "[REDACTED]", "n": 1})}),
            "number": 12345678,
            "text": "[REDACTED]",
            "keep": "ordinary text",
        }
        assert redactor.count == 8


def test_redactor_without_secrets_leaves_text_untouched():
    line = '{"a": "b"}\n'
    redactor = Redactor(set())
    assert redactor.json(line) is line
    assert redactor.value({"a": "b"}) == {"a": "b"}


def test_known_secrets_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_API_KEY", "env-key-value-0001")
    monkeypatch.setenv("X-Auth", "hyphenated-auth-0001")
    monkeypatch.setenv("MY_SHORT_KEY", "short")  # too short to redact safely
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Some Author Name")  # AUTHOR is not AUTH
    monkeypatch.setenv("DATABASE_URL", "postgres://app:db-pass-000001@db/x")  # URL password
    monkeypatch.setenv("GIT_REMOTE", "https://ghp_token_000000001@github.com/o/r")  # bare token
    monkeypatch.setenv("PG_URL", "postgres://app:p%40ss-000001@db")  # percent-encoded password
    secrets_file = tmp_path / "secrets"
    secrets_file.write_text("from-file-0001\n\n  spaced  \n")

    found = known_secrets(
        "api-key-0001", None, secret_args=["literal", str(secrets_file), "j" * 400]
    )

    assert {
        "env-key-value-0001",
        "hyphenated-auth-0001",
        "api-key-0001",
        "literal",
        "from-file-0001",
        "spaced",
        "db-pass-000001",
        "ghp_token_000000001",
        "p%40ss-000001",
        "p@ss-000001",
        "j" * 400,  # longer than a filesystem name: a literal, not a file to probe
    } <= found
    assert not {"short", "Some Author Name", "", "postgres://app:db-pass-000001@db/x"} & found
