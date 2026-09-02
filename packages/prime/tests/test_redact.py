"""Exact-match redaction: a known value disappears from JSON strings in every spelling
(plain, escaped, escaped again inside a quoted JSON document) and nothing else changes."""

import json

from prime_cli.utils.redact import Redactor, known_secrets


def test_redactor_replaces_every_spelling_inside_strings_only():
    redactor = Redactor({'pa"ss\\word-0001', "12345678", "ünïcode-secret"})
    doc = {
        "plain": 'pa"ss\\word-0001 and ünïcode-secret',
        "nested": json.dumps({"k": 'pa"ss\\word-0001', "u": "ünïcode-secret"}),
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
            "number": 12345678,
            "text": "[REDACTED]",
            "keep": "ordinary text",
        }
        assert redactor.count == 5


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
    secrets_file = tmp_path / "secrets"
    secrets_file.write_text("from-file-0001\n\n  spaced  \n")

    found = known_secrets("api-key-0001", None, secret_args=["literal", str(secrets_file)])

    assert {
        "env-key-value-0001",
        "hyphenated-auth-0001",
        "api-key-0001",
        "literal",
        "from-file-0001",
        "spaced",
    } <= found
    assert not {"short", "Some Author Name", ""} & found
