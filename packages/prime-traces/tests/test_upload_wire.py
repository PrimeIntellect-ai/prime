import gzip
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Dict, Tuple

import httpx
import pytest

from prime_traces import (
    LineFormat,
    LineFormatConflictError,
    RetryableAPIError,
    TransportError,
    ValidationRejectedError,
)

RAW = b'{"id":"a"}\n{"id":"b"}\n'
COMMITTED = {"upload_id": "x" * 64, "status": "committed"}


def parse_multipart(content: bytes, content_type: str) -> Dict[str, Tuple[Dict[str, str], bytes]]:
    """Parse a multipart body into {part_name: (headers, exact body bytes)}.

    Body bytes are preserved exactly (no stripping), so binary payloads such
    as gzip members survive.
    """
    boundary = content_type.split("boundary=")[1].encode()
    sections = (b"\r\n" + content).split(b"\r\n--" + boundary)
    parts: Dict[str, Tuple[Dict[str, str], bytes]] = {}
    for section in sections[1:]:
        if section.startswith(b"--"):
            break
        section = section[2:]  # the CRLF that terminated the boundary line
        header_blob, _, body = section.partition(b"\r\n\r\n")
        headers: Dict[str, str] = {}
        for header_line in header_blob.split(b"\r\n"):
            key, _, value = header_line.partition(b":")
            headers[key.strip().lower().decode()] = value.strip().decode()
        match = re.search(r'name="([^"]+)"', headers.get("content-disposition", ""))
        assert match, f"part without a name: {headers}"
        parts[match.group(1)] = (headers, body)
    return parts


def upload(client, **kwargs):
    return client.upload_lines([b'{"id":"a"}\n', b'{"id":"b"}\n'], **kwargs)


class TestRequestShape:
    def test_bare_trace_upload_without_compression(self, make_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            captured["content"] = request.content
            return httpx.Response(201, json=COMMITTED)

        client = make_client(handler)
        [receipt] = upload(client, context={"source": "hosted_eval"}, compress=False)

        request = captured["request"]
        assert request.url.path == "/api/v1/traces"
        assert request.headers["Authorization"] == "Bearer test-key"

        digest = hashlib.sha256(RAW).hexdigest()
        assert request.headers["Idempotency-Key"] == f"sha256:{digest}"
        # Bare-trace is the default: the header must be absent, not "trace".
        assert "X-Prime-Line-Format" not in request.headers

        parts = parse_multipart(captured["content"], request.headers["content-type"])
        assert set(parts) == {"metadata", "traces"}

        metadata_headers, metadata_body = parts["metadata"]
        assert metadata_headers["content-type"] == "application/json"
        assert json.loads(metadata_body) == {
            "schema_version": 1,
            "context": {"source": "hosted_eval"},
        }

        traces_headers, traces_body = parts["traces"]
        assert traces_headers["content-type"] == "application/x-ndjson"
        assert "content-encoding" not in traces_headers
        assert traces_body == RAW

        assert receipt.status == "committed"

    def test_episode_upload_with_gzip(self, make_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            captured["content"] = request.content
            return httpx.Response(201, json=COMMITTED)

        client = make_client(handler)
        upload(client, line_format=LineFormat.EPISODE, compress=True)

        request = captured["request"]
        assert request.headers["X-Prime-Line-Format"] == "episode"

        # The digest is over the *uncompressed* bytes even when the part is
        # gzip-compressed for transport.
        digest = hashlib.sha256(RAW).hexdigest()
        assert request.headers["Idempotency-Key"] == f"sha256:{digest}"

        traces_headers, traces_body = parse_multipart(
            captured["content"], request.headers["content-type"]
        )["traces"]
        assert traces_headers["content-encoding"] == "gzip"
        assert gzip.decompress(traces_body) == RAW

    def test_metadata_omits_context_when_absent(self, make_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["content"] = request.content
            captured["content_type"] = request.headers["content-type"]
            return httpx.Response(201, json=COMMITTED)

        upload(make_client(handler), compress=False)
        _, metadata_body = parse_multipart(captured["content"], captured["content_type"])[
            "metadata"
        ]
        assert json.loads(metadata_body) == {"schema_version": 1}


class TestRetrySemantics:
    def test_retries_on_503_honoring_retry_after(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                return httpx.Response(503, headers={"Retry-After": "1.5"})
            return httpx.Response(201, json=COMMITTED)

        [receipt] = upload(make_client(handler))
        assert receipt.status == "committed"
        # Same bytes, same key, both attempts.
        assert len(set(attempts)) == 1
        assert no_sleep == [1.5]

    def test_retry_after_http_date_is_honored(self, make_client, no_sleep):
        """RFC 9110 also allows Retry-After as an HTTP-date — gateways emit
        that form — so it converts to the remaining delay instead of falling
        back to a shorter local backoff that retries before the server asked."""
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                target = datetime.now(timezone.utc) + timedelta(seconds=30)
                return httpx.Response(
                    429, headers={"Retry-After": format_datetime(target, usegmt=True)}
                )
            return httpx.Response(201, json=COMMITTED)

        [receipt] = upload(make_client(handler))
        assert receipt.status == "committed"
        [delay] = no_sleep
        # Remaining time from now: allow for the seconds format_datetime drops
        # and the wall clock advancing while the test runs.
        assert 25 <= delay <= 30

    def test_retry_after_past_http_date_retries_immediately(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                target = datetime.now(timezone.utc) - timedelta(hours=1)
                return httpx.Response(
                    503, headers={"Retry-After": format_datetime(target, usegmt=True)}
                )
            return httpx.Response(201, json=COMMITTED)

        [receipt] = upload(make_client(handler))
        assert receipt.status == "committed"
        assert no_sleep == [0.0]

    def test_retry_after_is_capped(self, make_client, no_sleep):
        """Retry-After is server-controlled input: honored, but bounded so a
        misconfigured gateway cannot park the uploader for a day."""
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                return httpx.Response(503, headers={"Retry-After": "86400"})
            return httpx.Response(201, json=COMMITTED)

        [receipt] = upload(make_client(handler))
        assert receipt.status == "committed"
        assert no_sleep == [60.0]

    def test_unparseable_retry_after_falls_back_to_backoff(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                return httpx.Response(503, headers={"Retry-After": "soon-ish"})
            return httpx.Response(201, json=COMMITTED)

        [receipt] = upload(make_client(handler))
        assert receipt.status == "committed"
        # First-attempt jittered backoff: base 1.0 x (0.5..1.5).
        [delay] = no_sleep
        assert 0.5 <= delay <= 1.5

    @pytest.mark.parametrize("parse_error", [TypeError, OverflowError])
    def test_retry_after_parser_failure_falls_back_to_backoff(
        self, make_client, no_sleep, monkeypatch, parse_error
    ):
        attempts = []

        def fail_to_parse(_value):
            raise parse_error("malformed date")

        monkeypatch.setattr(
            "prime_traces.core.client.parsedate_to_datetime",
            fail_to_parse,
        )

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                return httpx.Response(503, headers={"Retry-After": "not-a-date"})
            return httpx.Response(201, json=COMMITTED)

        [receipt] = upload(make_client(handler))
        assert receipt.status == "committed"
        [delay] = no_sleep
        assert 0.5 <= delay <= 1.5

    @pytest.mark.parametrize("status", [502, 504])
    def test_retries_gateway_responses(self, make_client, no_sleep, status):
        """502/504 come from a gateway, not the service — no error envelope,
        and the first attempt may have been processed. Content addressing makes
        the retry safe: the same key replays the committed receipt."""
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                return httpx.Response(status, text="upstream connect error")
            return httpx.Response(201, json=COMMITTED)

        [receipt] = upload(make_client(handler))
        assert receipt.status == "committed"
        assert len(set(attempts)) == 1
        assert len(no_sleep) == 1

    def test_retries_transport_failures_with_same_key(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.headers["Idempotency-Key"])
            if len(attempts) == 1:
                raise httpx.ConnectError("connection refused", request=request)
            if len(attempts) == 2:
                # Ambiguous: the server may have processed the request. Safe
                # only because the same key replays the committed receipt.
                raise httpx.ReadError("connection reset mid-response", request=request)
            return httpx.Response(201, json=COMMITTED)

        [receipt] = upload(make_client(handler))
        assert receipt.status == "committed"
        assert len(set(attempts)) == 1
        assert len(no_sleep) == 2

    def test_transport_failure_exhausts_attempts(self, make_client, no_sleep):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(TransportError):
            upload(make_client(handler), max_attempts=2)
        assert len(no_sleep) == 1

    def test_gives_up_after_max_attempts(self, make_client, no_sleep):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.1"},
                json={"error": {"code": "rate_limited", "message": "slow down"}},
            )

        with pytest.raises(RetryableAPIError) as exc_info:
            upload(make_client(handler), max_attempts=3)
        assert exc_info.value.status_code == 429
        # The service code survives so callers can tell rate_limited from
        # writer_pool_saturated / storage_unavailable / auth_unavailable.
        assert exc_info.value.code == "rate_limited"
        assert len(no_sleep) == 2  # sleeps between attempts, not after the last

    def test_non_positive_max_attempts_rejected(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json=COMMITTED)

        with pytest.raises(ValueError, match="max_attempts"):
            upload(make_client(handler), max_attempts=0)

    def test_durable_rejection_stops_the_upload(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"code": "invalid_trace", "message": "line 7: not a trace"}},
            )

        with pytest.raises(ValidationRejectedError) as exc_info:
            upload(make_client(handler))
        assert exc_info.value.code == "invalid_trace"

    def test_line_format_conflict_is_typed(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={"error": {"code": "line_format_conflict", "message": "conflict"}},
            )

        with pytest.raises(LineFormatConflictError):
            upload(make_client(handler))
