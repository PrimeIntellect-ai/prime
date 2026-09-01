import json

import pytest

from prime_evals.preflight import (
    REDACTED,
    UploadScanError,
    fingerprint_secret,
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


@pytest.mark.parametrize(
    "assignment",
    [
        "password=abcdefgh1234",
        "TOKEN=abcdefgh1234",
        "SECRET_KEY=abcdefgh1234",
        "api_token=abcdefghijklmnop",
        "--token 123456789abc",
        'password="abcd efgh"',
        'password="abcd,efgh"',
        r"password=\"abcdefgh1234\"",
        r"password=\'abcdefgh1234\'",
        "sas_token=opaque-sas-token-0123456789",
        "cookie=abcdefgh",
    ],
)
def test_preflight_catches_short_explicit_assignments(assignment):
    assert REDACTED in prepare_upload({"completion": assignment}).data["completion"]


def test_preflight_catches_azure_storage_account_keys():
    secret = "QWxhZGRpbjpvcGVuIHNlc2FtZQ=="
    connection = (
        "DefaultEndpointsProtocol=https;AccountName=example;"
        f"AccountKey={secret};EndpointSuffix=core.windows.net"
    )

    prepared = prepare_upload({"completion": connection})

    assert secret not in prepared.data["completion"]


@pytest.mark.parametrize(
    "token",
    [
        "xwfp-0123456789-abcdefghijklmnop",
        "xapp-0123456789-abcdefghijklmnop",
        "rk_" + "live_" + "0" * 24,
        "sk_" + "test_" + "0" * 24,
    ],
)
def test_preflight_catches_additional_provider_credentials(token):
    assert prepare_upload({"completion": token}).data["completion"] == REDACTED


def test_preflight_redacts_every_value_in_a_raw_cookie_header():
    session = "opaque-cookie-session-0123456789"
    refresh = "opaque-cookie-refresh-0123456789"
    header = json.dumps({"Cookie": f"session={session}; refresh={refresh}"})
    prepared = prepare_upload(
        {
            "completion": f"request: {header}",
            "response": f"Set-Cookie: session={session}",
        }
    )

    assert session not in prepared.data["completion"]
    assert refresh not in prepared.data["completion"]
    assert session not in prepared.data["response"]


def test_preflight_catches_quoted_json_assignments_and_short_structured_secrets():
    token = "opaque-json-token-0123456789"
    payload = {
        "completion": json.dumps({"Authorization": f"Bearer {token}"}),
        "error": "Authorization: Basic dXNlcjpwYXNz",
        "answer": "Use abc123 and token=version-123 for the example.",
        "password": "s3cr3t",
        "api_key": "abc123",
    }

    prepared = prepare_upload(payload)

    assert token not in prepared.data["completion"]
    assert "dXNlcjpwYXNz" not in prepared.data["error"]
    assert prepared.data["answer"] == payload["answer"]
    assert prepared.data["password"] == REDACTED
    assert prepared.data["api_key"] == REDACTED

    digest = 'Authorization: Digest username="alice", nonce="opaque-nonce-0123456789"'
    negotiate = "Authorization: Negotiate opaque-spnego-token-0123456789"
    prepared = prepare_upload({"digest": digest, "negotiate": negotiate})
    assert "opaque-nonce" not in prepared.data["digest"]
    assert "opaque-spnego" not in prepared.data["negotiate"]


def test_preflight_catches_escaped_authorization_headers():
    token = "opaque-token-0123456789"
    completion = rf"{{\"Authorization\":\"Bearer {token}\"}}"

    prepared = prepare_upload({"completion": completion})

    assert token not in prepared.data["completion"]


def test_preflight_distinguishes_oauth_metadata_from_credential_value_fields():
    payload = {
        "oauth": "enabled",
        "hasOauth": True,
        "auth_header": "Bearer opaque-auth-token-0123456789",
        "authorization_header": "Bearer opaque-authorization-token-0123456789",
        "secret_value": "opaque-secret-0123456789",
        "api_key_value": "opaque-api-key-0123456789",
        "token_value": "opaque-token-0123456789",
    }

    prepared = prepare_upload(payload)

    assert prepared.data == {
        "oauth": "enabled",
        "hasOauth": True,
        "auth_header": REDACTED,
        "authorization_header": REDACTED,
        "secret_value": REDACTED,
        "api_key_value": REDACTED,
        "token_value": REDACTED,
    }


@pytest.mark.parametrize("field", ["apikey", "accesstoken", "clientsecret", "secretkey"])
def test_preflight_catches_concatenated_credential_fields(field):
    assert prepare_upload({field: "opaque-value-0123456789"}).data[field] == REDACTED


def test_preflight_catches_quoted_assignments_and_sensitive_mapping_keys():
    secrets = ["opaque-map-key-0123456789", "second-map-key-0123456789"]
    aws_secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    session_token = "opaque-aws-session-token-0123456789"
    prepared = prepare_upload(
        {
            "completion": (
                'password="correct horse battery staple" '
                f'{{"aws_secret_access_key":"{aws_secret}",'
                f'"aws_session_token":"{session_token}"}}'
            ),
            "api_keys": {secret: {"owner": "alice"} for secret in secrets},
            "prompt": f"the model repeated {secrets[0]}",
            "team_id": "team-0123456789",
        }
    )

    assert "correct horse battery staple" not in prepared.data["completion"]
    assert aws_secret not in prepared.data["completion"]
    assert session_token not in prepared.data["completion"]
    assert list(prepared.data["api_keys"]) == [REDACTED, "[REDACTED 2]"]
    assert prepared.data["prompt"] == f"the model repeated {REDACTED}"
    assert prepared.data["team_id"] == "team-0123456789"
    assert scan_upload(prepared.data).findings == ()


def test_pattern_discovered_secrets_are_redacted_everywhere():
    secret = "opaque-password-value-0123456789"
    prepared = prepare_upload({"prompt": f"model repeated {secret}", "log": f"password={secret}"})

    assert secret not in json.dumps(prepared.data)
    assert prepared.data["prompt"] == f"model repeated {REDACTED}"


def test_url_userinfo_is_redacted_without_becoming_a_global_secret():
    prepared = prepare_upload(
        {
            "log": "redis://user:password@example.com/0",
            "rubric": "mention the password field",
        }
    )

    assert "user:password" not in prepared.data["log"]
    assert prepared.data["rubric"] == "mention the password field"


def test_preflight_redacts_numeric_structured_secrets():
    prepared = prepare_upload(
        {"password": 12345678, "token": 987654321, "secret": None, "auth": False}
    )

    assert prepared.data == {
        "password": REDACTED,
        "token": REDACTED,
        "secret": None,
        "auth": False,
    }

    usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    assert prepare_upload({"usage": usage}).data["usage"] == usage
    token_usage = {"prompt_tokens": 123, "completion_tokens": 42}
    assert prepare_upload({"token_usage": token_usage}).data["token_usage"] == token_usage


@pytest.mark.parametrize("label", ["DSA PRIVATE KEY", "ENCRYPTED PRIVATE KEY"])
def test_preflight_catches_private_key_pem_variants(label):
    private_key = f"-----BEGIN {label}-----\nopaque-key-material\n-----END {label}-----"

    assert prepare_upload({"completion": private_key}).data["completion"] == REDACTED


@pytest.mark.parametrize("scheme", ["Bearer", "Token"])
def test_structured_authorization_redacts_the_repeated_token(scheme):
    token = "opaque-bearer-token-0123456789"
    prepared = prepare_upload(
        {
            "headers": {"Authorization": f"{scheme} {token}"},
            "completion": f"the model repeated {token}",
        }
    )

    assert prepared.data["headers"]["Authorization"] == REDACTED
    assert token not in prepared.data["completion"]


@pytest.mark.parametrize(
    "headers",
    [
        [["authorization", "Bearer opaque-header-token-0123456789"]],
        [
            {
                "key": "Authorization",
                "value": "Bearer opaque-header-token-0123456789",
            }
        ],
    ],
)
def test_preflight_recognizes_suffixed_header_containers(headers):
    token = "opaque-header-token-0123456789"
    prepared = prepare_upload(
        {
            "request_headers": headers,
            "completion": f"the model repeated {token}",
        }
    )

    assert token not in json.dumps(prepared.data)


def test_preflight_preserves_openapi_security_scheme_definitions():
    security_schemes = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    prepared = prepare_upload({"components": {"securitySchemes": security_schemes}})

    assert prepared.data["components"]["securitySchemes"] == security_schemes


def test_preflight_uses_named_secret_sources_and_fingerprints():
    runtime_secret = "runtime-secret-0123456789"
    capability = "rollout-capability-0123456789"
    prepared = prepare_upload(
        {"completion": f"{runtime_secret} {capability} reviewable-setting"},
        secret_sources=[{"RUNTIME_SECRET": runtime_secret, "X-Custom": "reviewable-setting"}],
        secret_fingerprints=[fingerprint_secret(capability)],
    )

    assert prepared.data["completion"] == f"{REDACTED} {REDACTED} reviewable-setting"
    assert prepared.report.categories == {"known_secret": 1}
    with pytest.raises(ValueError, match="fingerprinted secrets"):
        fingerprint_secret("")


def test_preflight_fingerprint_scan_handles_unpaired_surrogates():
    capability = "opaque-capability-0123456789"
    payload = {"completion": f"prefix \ud800 {capability} suffix"}

    prepared = prepare_upload(
        payload,
        secret_fingerprints=[fingerprint_secret(capability)],
    )

    assert capability not in prepared.data["completion"]
    assert "\ud800" in prepared.data["completion"]


def test_nested_and_properties_credentials_do_not_bypass_discovery():
    secret = "opaque-nested-secret-0123456789"
    token = "opaque-generic-token-0123456789"
    access_key = "opaque-aws-secret-access-key-0123456789"
    plural_key = "opaque-plural-key-0123456789"
    property_key = "opaque-property-key-0123456789"
    payload = {
        "APIKey": secret,
        "apiKeys": [plural_key],
        "awsSecretAccessKey": access_key,
        "credential": {"value": "pin"},
        "authentication": "opaque-authentication-0123456789",
        "headers": [
            ["authorization", "Bearer opaque-pair-token-0123456789"],
            {"name": "Authorization", "value": "Bearer opaque-named-token-0123456789"},
        ],
        "rubric": "keep the pin label",
        "secret": {"value": secret},
        "token": token,
        "properties": {"password": {"type": "string", "value": property_key}},
        "schema": {
            "properties": {
                "password": {"type": ["string", "null"], "description": "keep this"},
                "apiKey": {"description": "typeless schema"},
            }
        },
    }

    prepared = prepare_upload(payload)

    assert prepared.data["APIKey"] == REDACTED
    assert prepared.data["apiKeys"] == [REDACTED]
    assert prepared.data["awsSecretAccessKey"] == REDACTED
    assert prepared.data["credential"]["value"] == REDACTED
    assert prepared.data["authentication"] == REDACTED
    assert prepared.data["headers"] == [
        ["authorization", REDACTED],
        {"name": "Authorization", "value": REDACTED},
    ]
    assert prepared.data["rubric"] == payload["rubric"]
    assert prepared.data["secret"]["value"] == REDACTED
    assert prepared.data["token"] == REDACTED
    assert prepared.data["properties"]["password"]["value"] == REDACTED
    assert prepared.data["schema"] == payload["schema"]

    event = {
        "type": "event",
        "properties": {"password": {"value": "opaque-password-0123456789"}},
    }
    assert prepare_upload(event).data["properties"]["password"]["value"] == REDACTED
    event = {
        "items": [],
        "properties": {"password": {"value": "opaque-password-0123456789"}},
    }
    assert prepare_upload(event).data["properties"]["password"]["value"] == REDACTED


def test_sensitive_schema_enums_are_redacted():
    secret = "opaque-schema-secret-0123456789"
    schema = {
        "type": "object",
        "properties": {"api_key": {"type": "string", "enum": [secret]}},
    }

    prepared = prepare_upload({"schema": schema})

    assert prepared.data["schema"]["properties"]["api_key"]["enum"] == [REDACTED]


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
    assert prepared.path.name.startswith("redacted-")
    assert prepared.path.name.endswith("-safe.jsonl")
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


def test_jsonl_preflight_rejects_output_paths_that_alias_the_source(tmp_path):
    source = tmp_path / "safe.jsonl"
    source.write_text('{"answer":"keep"}\n')

    with pytest.raises(UploadScanError, match="must not alias"):
        prepare_jsonl_upload(source, tmp_path / "safe.jsonl")

    assert source.read_text() == '{"answer":"keep"}\n'


def test_jsonl_preflight_preserves_unrelated_redacted_siblings(tmp_path):
    source = tmp_path / "traces.jsonl"
    destination = tmp_path / "safe.jsonl"
    sibling = tmp_path / "redacted-safe.jsonl"
    source.write_text('{"answer":"keep"}\n')
    sibling.write_text("unrelated\n")

    prepare_jsonl_upload(source, destination)

    assert sibling.read_text() == "unrelated\n"


def test_jsonl_preflight_rejects_a_hard_link_to_the_source(tmp_path):
    source = tmp_path / "traces.jsonl"
    destination = tmp_path / "safe.jsonl"
    source.write_text('{"answer":"keep"}\n')
    destination.hardlink_to(source)

    with pytest.raises(UploadScanError, match="must not alias"):
        prepare_jsonl_upload(source, destination)

    assert source.read_text() == '{"answer":"keep"}\n'


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


def test_secret_values_includes_auth_environment_variables(monkeypatch):
    secret = "opaque-auth-secret-0123456789"
    monkeypatch.setenv("AUTH", secret)
    monkeypatch.setenv("API_KEY_VAR", "REFERENCE_API_KEY")

    assert secret in secret_values()
    assert "REFERENCE_API_KEY" not in secret_values()


def test_secret_values_extracts_named_source_credentials():
    token = "opaque-source-token-0123456789"

    values = secret_values(
        secret_sources=[
            {
                "Authorization": f"Bearer {token}",
                "X-Custom": "reviewable-setting",
            }
        ]
    )

    assert f"Bearer {token}" in values
    assert token in values
    assert "reviewable-setting" not in values
