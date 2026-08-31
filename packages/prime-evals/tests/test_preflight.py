import json

import pytest

from prime_evals.preflight import (
    REDACTED,
    UploadScanError,
    prepare_jsonl_upload,
    prepare_upload,
    scan_upload,
    secret_values,
)


def test_preflight_reduces_credentials_without_reducing_trace_review_data():
    provider_key = "sk-test-0123456789abcdefghijklmnopqrstuv"
    opaque_key = "opaque-judge-key-0123456789"
    upload_key = "opaque-prime-key-0123456789"
    payload = {
        "metadata": {
            "judgeApiKey": opaque_key,
            "api_key_var": "JUDGE_API_KEY",
        },
        "results": [
            {
                "prompt": f"The leaked provider key is {provider_key}",
                "completion": f"Repeated judge key: {opaque_key}; upload key: {upload_key}",
                "answer": "reference answer",
                "rubric": "award one point for the reference answer",
                "traceback": "/Users/alice/project failed on 10.0.0.4",
                "tool_defs": [
                    {
                        "name": "login",
                        "parameters": {
                            "type": "object",
                            "properties": {"password": {"type": "string"}},
                        },
                    }
                ],
            }
        ],
    }

    prepared = prepare_upload(payload, [upload_key])
    serialized = json.dumps(prepared.data)

    assert provider_key not in serialized
    assert opaque_key not in serialized
    assert upload_key not in serialized
    assert payload["metadata"]["judgeApiKey"] == opaque_key
    assert prepared.data["metadata"] == {
        "judgeApiKey": REDACTED,
        "api_key_var": "JUDGE_API_KEY",
    }
    result = prepared.data["results"][0]
    assert result["answer"] == "reference answer"
    assert result["rubric"].startswith("award one point")
    assert result["traceback"] == "/Users/alice/project failed on 10.0.0.4"
    assert result["tool_defs"][0]["parameters"]["properties"] == {"password": {"type": "string"}}
    assert prepared.report.locations == 3
    assert prepared.report.categories == {
        "known_secret": 1,
        "provider_credential": 1,
        "structured_secret": 2,
    }
    assert scan_upload(prepared.data).findings == ()


def test_preflight_catches_assignments_flags_urls_webhooks_and_private_keys():
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n0123456789abcdefghijklmnopqrstuv\n-----END PRIVATE KEY-----"
    )
    webhook = "https://hooks.slack.com/" + "services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
    payload = {
        "text": (
            "TOKEN=opaque-token-0123456789 "
            "--api-key cli-token-0123456789 "
            "redis://user:password@example.com/0?access_token=query-token-0123456789\n"
            f"{webhook}\n"
            f"{private_key}"
        )
    }

    prepared = prepare_upload(payload)

    assert "opaque-token" not in prepared.data["text"]
    assert "cli-token" not in prepared.data["text"]
    assert "user:password" not in prepared.data["text"]
    assert "query-token" not in prepared.data["text"]
    assert "hooks.slack.com" not in prepared.data["text"]
    assert "BEGIN PRIVATE KEY" not in prepared.data["text"]
    assert prepared.report.categories == {
        "credential_assignment": 1,
        "credential_url": 1,
        "private_key": 1,
        "webhook_credential": 1,
    }


@pytest.mark.parametrize("label", ["DSA PRIVATE KEY", "ENCRYPTED PRIVATE KEY"])
def test_preflight_catches_private_key_pem_variants(label):
    private_key = f"-----BEGIN {label}-----\nopaque-key-material\n-----END {label}-----"

    assert prepare_upload({"completion": private_key}).data["completion"] == REDACTED


def test_structured_authorization_redacts_the_repeated_bearer_token():
    token = "opaque-bearer-token-0123456789"
    prepared = prepare_upload(
        {
            "headers": {"Authorization": f"Bearer {token}"},
            "completion": f"the model repeated {token}",
        }
    )

    assert prepared.data["headers"]["Authorization"] == REDACTED
    assert token not in prepared.data["completion"]


def test_nested_and_properties_credentials_do_not_bypass_discovery():
    secret = "opaque-nested-secret-0123456789"
    token = "opaque-generic-token-0123456789"
    payload = {
        "APIKey": secret,
        "secret": {"value": secret},
        "token": token,
        "properties": {"password": secret},
        "schema": {"properties": {"password": {"type": "string", "description": "keep this"}}},
    }

    prepared = prepare_upload(payload)

    assert prepared.data["APIKey"] == REDACTED
    assert prepared.data["secret"]["value"] == REDACTED
    assert prepared.data["token"] == REDACTED
    assert prepared.data["properties"]["password"] == REDACTED
    assert prepared.data["schema"] == payload["schema"]


def test_jsonl_preflight_uses_secrets_discovered_in_later_lines(tmp_path):
    secret = "opaque-later-line-secret-0123456789"
    source = tmp_path / "traces.jsonl"
    destination = tmp_path / "safe.jsonl"
    original = (
        json.dumps({"prompt": f"the model repeated {secret}", "answer": "keep me"})
        + "\n"
        + json.dumps({"config": {"apiKey": secret}, "rubric": "keep this too"})
        + "\n"
    )
    source.write_text(original)

    prepared = prepare_jsonl_upload(
        source,
        destination,
        context={"password": secret, "source": "local-eval"},
    )

    assert source.read_text() == original
    assert prepared.path == tmp_path / "redacted-safe.jsonl"
    assert prepared.context == {"password": REDACTED, "source": "local-eval"}
    assert secret not in prepared.path.read_text()
    records = [json.loads(line) for line in prepared.path.read_text().splitlines()]
    assert records[0]["answer"] == "keep me"
    assert records[1]["rubric"] == "keep this too"
    assert prepared.report.locations == 3


def test_jsonl_preflight_uploads_an_exact_snapshot_when_clean(tmp_path):
    source = tmp_path / "traces.jsonl"
    destination = tmp_path / "safe.jsonl"
    source.write_bytes(b'{ "answer": "reference" }\n\n')

    prepared = prepare_jsonl_upload(source, destination)

    assert prepared.path == destination
    assert destination.read_bytes() == source.read_bytes()
    assert prepared.report.findings == ()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"id":}\n', "invalid JSON"),
        ('{"password":"first-secret", "password":"second-secret"}\n', "duplicate object key"),
    ],
)
def test_jsonl_preflight_fails_closed_on_ambiguous_input(tmp_path, content, message):
    source = tmp_path / "traces.jsonl"
    source.write_text(content)

    with pytest.raises(UploadScanError, match=message):
        prepare_jsonl_upload(source, tmp_path / "safe.jsonl")


def test_secret_values_file(tmp_path):
    path = tmp_path / "secrets.txt"
    path.write_text("# exact values\nopaque-secret-0123456789\n\nsecond-secret-0123456789\n")

    values = secret_values(secrets_file=path)
    assert "opaque-secret-0123456789" in values
    assert "second-secret-0123456789" in values

    path.write_text("short\n")
    with pytest.raises(ValueError, match="line 1"):
        secret_values(secrets_file=path)
