"""Exact-match redaction: a known value disappears from JSON strings in every spelling
(plain, escaped, escaped again inside quoted JSON documents at any depth, with the `\\/`
and uppercase-hex escapes other encoders emit) and nothing else changes."""

import json

import pytest
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


@pytest.mark.parametrize("value", ["REDACTED", "[REDACTED", "ED]"])
def test_known_secrets_refuses_values_inside_the_marker(value):
    with pytest.raises(ValueError, match="REDACTED"):
        known_secrets(secret_args=[value])


def test_known_secrets_keeps_values_that_contain_or_border_the_marker(monkeypatch):
    """Skipping these would upload a real credential for certain; a replacement can only
    form one of them around another registered secret."""
    monkeypatch.setenv("SERVICE_PASSWORD", "live[REDACTED]pass-123")
    found = known_secrets(secret_args=["]bar", "foo["])
    assert {"live[REDACTED]pass-123", "]bar", "foo["} <= found


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
    monkeypatch.setenv("KEYCLOAK_REALM", "production-realm")  # KEYCLOAK is not KEY
    monkeypatch.setenv("COOKIECUTTER_CONFIG", "/home/user/config")  # nor COOKIE
    monkeypatch.setenv("OPENAI_APIKEY", "apikey-nosep-0001")  # but APIKEY is a key
    monkeypatch.setenv("PGPASSWORD", "pg-pass-000001")  # and PGPASSWORD a password
    monkeypatch.setenv("PGPASSFILE", "/home/user/.pgpass")  # a path, not a password
    monkeypatch.setenv("LEGACY_API_KEY", "[REDACTED]")  # a sanitized placeholder: skipped
    monkeypatch.setenv("BROWSER_URL", "wss://b.example/devtools?token=query-token-0001&v=2")
    monkeypatch.setenv("NOTE", "see ?token=prose-token-0001")  # prose, not a URL
    monkeypatch.setenv("ODD_URL", "https://h.example/?to%6ben=encoded-name-0001")  # encoded name
    monkeypatch.setenv("FORM_URL", "https://h.example/?token=plus+sep+0001")  # form encoding
    monkeypatch.setenv("SAS_URL", "https://a.blob.core.windows.net/c?sv=2020&sig=sas-sig-000001")
    monkeypatch.setenv("GITHUB_PAT", "ghp_pat_000000000001")  # PAT is a credential word
    monkeypatch.setenv("PYTHONPATH", "/opt/keep/this/path")  # PATH is not
    monkeypatch.setenv("DATABASE_URL", "postgres://app:db-pass-000001@db/x")  # URL password
    monkeypatch.setenv("GIT_REMOTE", "https://ghp_token_000000001@github.com/o/r")  # bare token
    monkeypatch.setenv("PG_URL", "postgres://app:p%40ss-000001@db")  # percent-encoded password
    monkeypatch.setenv("RO_URL", "postgres://readonly_user:@db")  # empty password: a name
    secrets_file = tmp_path / "secrets"
    secrets_file.write_text("from-file-0001\r\n\n  spaced  \n")

    found = known_secrets(
        "api-key-0001", None, secret_args=["literal", "lit ", str(secrets_file), "j" * 400]
    )

    assert {
        "env-key-value-0001",
        "hyphenated-auth-0001",
        "api-key-0001",
        "literal",
        "lit ",  # explicit values keep their whitespace
        "from-file-0001",  # a CRLF file entry loses only its terminator
        "  spaced  ",
        "db-pass-000001",
        "ghp_token_000000001",
        "p%40ss-000001",
        "p@ss-000001",
        "apikey-nosep-0001",
        "pg-pass-000001",
        "query-token-0001",
        "encoded-name-0001",
        "plus+sep+0001",
        "plus sep 0001",  # as a form-decoding client uses it
        "sas-sig-000001",  # a signed URL's signature is its bearer credential
        "ghp_pat_000000000001",
        "j" * 400,  # longer than a filesystem name: a literal, not a file to probe
    } <= found
    assert (
        not {
            "short",
            "Some Author Name",
            "",
            "postgres://app:db-pass-000001@db/x",
            "readonly_user",
            "production-realm",
            "/home/user/config",
            "/home/user/.pgpass",
            "[REDACTED]",
            "prose-token-0001",
            "/opt/keep/this/path",
        }
        & found
    )
