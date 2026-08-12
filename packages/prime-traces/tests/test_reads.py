import httpx
import pytest

from prime_traces import (
    ErrorCode,
    ForbiddenError,
    NotFoundError,
    RetryableAPIError,
    TracesAPIClient,
    TransportError,
    UnauthorizedError,
)

# The pinned summary shape — mirrors the service's `TraceSummary`
# (prime-traces/src/traces/models.py in the platform repo), including the
# null-for-unrecorded convention.
SUMMARY = {
    "trace_id": "8d3f1a2b",
    "upload_id": "5ee85e41",
    "episode_id": None,
    "created_at": "2026-07-20T18:02:11.482Z",
    "ingested_at": "2026-07-20T18:06:02.117Z",
    "run_id": "run_9f3k2m",
    "environment_id": "terminal-bench-2",
    "model": {"provider": "prime", "id": "deepseek-v4-flash"},
    "task_id": "tb2-0187",
    "agent_name": "solver",
    "score": {"reward": 0.85, "outcome": "done"},
    "execution": {"has_error": False, "is_truncated": False},
    "duration_ms": 215537,
    "total_tokens": 84213,
    "size_bytes": 417284,
    "context": {"source": "hosted_eval"},
}

RESERVED_TRACE_ID = "trace/with?reserved#chars%and space"
ENCODED_TRACE_PATH = b"/api/v1/traces/trace%2Fwith%3Freserved%23chars%25and%20space"


class TestList:
    def test_filters_encode_as_documented_query_params(self, make_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json={"items": [SUMMARY], "next_cursor": None})

        client = make_client(handler)
        page = client.list(
            run_id="run_9f3k2m",
            reward_min=0.5,
            has_error=False,
            context={"source": "hosted_eval"},
            limit=50,
        )

        assert captured["params"] == {
            "run_id": "run_9f3k2m",
            "reward_min": "0.5",
            "has_error": "false",
            "context.source": "hosted_eval",
            "limit": "50",
        }
        [summary] = page.items
        assert summary.trace_id == "8d3f1a2b"
        assert summary.score.reward == 0.85
        assert summary.model.id == "deepseek-v4-flash"
        assert summary.agent_name == "solver"
        assert summary.total_tokens == 84213
        assert summary.episode_id is None

    def test_unscored_reward_stays_none_not_zero(self, make_client):
        unscored = {**SUMMARY, "score": {"reward": None, "outcome": None}}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"items": [unscored], "next_cursor": None})

        [summary] = make_client(handler).list().items
        assert summary.score.reward is None

    def test_iter_follows_cursor(self, make_client):
        pages = {
            None: {"items": [{**SUMMARY, "trace_id": "t1"}], "next_cursor": "c1"},
            "c1": {"items": [{**SUMMARY, "trace_id": "t2"}], "next_cursor": None},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            cursor = request.url.params.get("cursor")
            return httpx.Response(200, json=pages[cursor])

        ids = [s.trace_id for s in make_client(handler).iter(run_id="run_9f3k2m")]
        assert ids == ["t1", "t2"]

    def test_tolerates_additive_summary_fields(self, make_client):
        # Fields the SDK has no typed slot for yet — the documented additive
        # growth path — must not break parsing.
        grown = {**SUMMARY, "num_turns": 4, "usage": {"prompt": 61000, "completion": 23213}}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"items": [grown], "next_cursor": None})

        [summary] = make_client(handler).list().items
        assert summary.trace_id == "8d3f1a2b"


class TestPointReads:
    def test_get_summary(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/traces/8d3f1a2b"
            return httpx.Response(200, json=SUMMARY)

        assert make_client(handler).get("8d3f1a2b").task_id == "tb2-0187"

    def test_get_encodes_trace_id_as_one_path_segment(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.raw_path == ENCODED_TRACE_PATH
            return httpx.Response(200, json=SUMMARY)

        make_client(handler).get(RESERVED_TRACE_ID)

    def test_get_raw_streams_document(self, make_client):
        raw = b'{"version":4,"id":"8d3f1a2b","nodes":[]}'

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["raw"] == "true"
            return httpx.Response(200, content=raw)

        assert make_client(handler).get_raw("8d3f1a2b") == raw

    def test_get_raw_encodes_trace_id_before_adding_query(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.raw_path == ENCODED_TRACE_PATH + b"?raw=true"
            return httpx.Response(200, content=b"{}")

        make_client(handler).get_raw(RESERVED_TRACE_ID)

    def test_download_raw_writes_file(self, make_client, tmp_path):
        raw = b'{"version":4,"id":"8d3f1a2b"}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=raw)

        dest = tmp_path / "trace.json"
        written = make_client(handler).download_raw("8d3f1a2b", dest)
        assert written == len(raw)
        assert dest.read_bytes() == raw

    def test_download_raw_encodes_trace_id_as_one_path_segment(self, make_client, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.raw_path == ENCODED_TRACE_PATH + b"?raw=true"
            return httpx.Response(200, content=b"{}")

        make_client(handler).download_raw(RESERVED_TRACE_ID, tmp_path / "trace.json")

    def test_download_raw_failure_preserves_existing_file(self, make_client, tmp_path):
        dest = tmp_path / "trace.json"
        dest.write_bytes(b"previous download")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, json={"error": {"code": "trace_not_found", "message": "no such trace"}}
            )

        with pytest.raises(NotFoundError):
            make_client(handler).download_raw("missing", dest)
        assert dest.read_bytes() == b"previous download"
        assert not (tmp_path / "trace.json.partial").exists()

    def test_not_found_is_typed(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": {"code": None, "message": "no such trace"}})

        with pytest.raises(NotFoundError):
            make_client(handler).get("missing")

    def test_unauthorized_is_typed(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "bad token"})

        with pytest.raises(UnauthorizedError):
            make_client(handler).get("8d3f1a2b")

    def test_forbidden_is_typed_with_server_message(self, make_client):
        """403 is an expected path, not an edge case: hosted-eval worker
        tokens are write-only, so any read they attempt lands here. The
        server's message names the missing scope and must survive."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={
                    "error": {
                        "code": "forbidden",
                        "message": "Token is missing 'traces:read' scope",
                    }
                },
            )

        with pytest.raises(ForbiddenError) as exc_info:
            make_client(handler).get("8d3f1a2b")
        assert exc_info.value.code == "forbidden"
        assert "traces:read" in str(exc_info.value)

    def test_service_not_enabled_is_a_nameable_403(self, make_client):
        """The owner allowlist gates every public route while the beta runs, so
        this is the 403 a new account is most likely to see — and the code has
        to be nameable, because it is the one a caller cannot fix by minting a
        better token."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={
                    "error": {
                        "code": "service_not_enabled",
                        "message": "Prime Traces is not enabled for this account",
                    }
                },
            )

        with pytest.raises(ForbiddenError) as exc_info:
            make_client(handler).get("8d3f1a2b")
        assert ErrorCode(exc_info.value.code) is ErrorCode.SERVICE_NOT_ENABLED


class TestReadRetries:
    """Idempotent requests retry transient failures with a bounded budget,
    mirroring the sibling SDKs (prime-sandboxes retries idempotent methods on
    502/503/504 and transport errors)."""

    _UNAVAILABLE = {"error": {"code": "storage_unavailable", "message": "storage is down"}}

    def test_get_retries_transient_503_honoring_retry_after(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                return httpx.Response(503, headers={"Retry-After": "2"}, json=self._UNAVAILABLE)
            return httpx.Response(200, json=SUMMARY)

        summary = make_client(handler).get("8d3f1a2b")
        assert summary.trace_id == "8d3f1a2b"
        assert len(attempts) == 2
        assert no_sleep == [2.0]

    def test_get_gives_up_after_bounded_attempts(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            return httpx.Response(503, json=self._UNAVAILABLE)

        with pytest.raises(RetryableAPIError) as exc_info:
            make_client(handler).get("8d3f1a2b")
        assert exc_info.value.code == "storage_unavailable"
        assert len(attempts) == 3
        assert len(no_sleep) == 2  # sleeps between attempts, not after the last

    def test_get_does_not_retry_404(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            return httpx.Response(
                404, json={"error": {"code": "trace_not_found", "message": "no such trace"}}
            )

        with pytest.raises(NotFoundError):
            make_client(handler).get("missing")
        assert len(attempts) == 1
        assert no_sleep == []

    def test_get_retries_transport_failures(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json=SUMMARY)

        assert make_client(handler).get("8d3f1a2b").trace_id == "8d3f1a2b"
        assert len(attempts) == 2
        assert len(no_sleep) == 1

    def test_delete_retries_transient_503(self, make_client, no_sleep):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.method)
            if len(attempts) == 1:
                return httpx.Response(503, json=self._UNAVAILABLE)
            return httpx.Response(202)

        make_client(handler).delete("8d3f1a2b")
        assert attempts == ["DELETE", "DELETE"]
        assert len(no_sleep) == 1

    def test_stream_retries_before_first_byte(self, make_client, no_sleep, tmp_path):
        raw = b'{"version":4,"id":"8d3f1a2b"}'
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                return httpx.Response(503, json=self._UNAVAILABLE)
            return httpx.Response(200, content=raw)

        dest = tmp_path / "trace.json"
        written = make_client(handler).download_raw("8d3f1a2b", dest)
        assert written == len(raw)
        assert dest.read_bytes() == raw
        assert len(attempts) == 2

    def test_stream_failure_mid_body_is_not_retried(self, make_client, no_sleep, tmp_path):
        """Once body bytes have flowed, a transparent retry would restart the
        document under the consumer and duplicate its prefix — so it raises,
        and the caller re-runs against a cleaned-up ``.partial``."""
        attempts = []

        def broken_body():
            yield b'{"version":4,'
            raise httpx.ReadError("connection reset mid-body")

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            return httpx.Response(200, content=broken_body())

        dest = tmp_path / "trace.json"
        with pytest.raises(TransportError):
            make_client(handler).download_raw("8d3f1a2b", dest)
        assert len(attempts) == 1
        assert no_sleep == []
        assert not dest.exists()
        assert not (tmp_path / "trace.json.partial").exists()


class TestDelete:
    def test_delete_trace_with_created_at_hint(self, make_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["params"] = dict(request.url.params)
            return httpx.Response(202)

        make_client(handler).delete("8d3f1a2b", created_at="2026-07-20T18:02:11.482Z")
        assert captured["method"] == "DELETE"
        assert captured["path"] == "/api/v1/traces/8d3f1a2b"
        assert captured["params"] == {"created_at": "2026-07-20T18:02:11.482Z"}

    def test_delete_encodes_trace_id_as_one_path_segment(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.raw_path == ENCODED_TRACE_PATH
            return httpx.Response(202)

        make_client(handler).delete(RESERVED_TRACE_ID)

    def test_delete_run_sends_run_id_and_expects_no_body(self, make_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["params"] = dict(request.url.params)
            # The service answers 202 with an empty body: the run delete is one
            # synchronous mutation, so there is no job handle to return.
            return httpx.Response(202)

        assert make_client(handler).delete_run("run_9f3k2m") is None
        assert captured["method"] == "DELETE"
        assert captured["path"] == "/api/v1/traces"
        assert captured["params"] == {"run_id": "run_9f3k2m"}

    def test_absent_trace_raises_not_found(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"error": {"code": "trace_not_found", "message": "No trace 'x'"}},
            )

        with pytest.raises(NotFoundError) as caught:
            make_client(handler).delete("x")
        assert caught.value.code == "trace_not_found"

    def test_404_after_an_ambiguous_attempt_is_treated_as_deleted(self, make_client):
        """A delete that lands and loses its response must not report failure.

        The service is not idempotent — it checks existence first and 404s when
        nothing matches — so the retry of a delete that already succeeded sees
        404. Absorbed, because the attempt before it is what removed the rows.
        """
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                raise httpx.ReadError("connection reset")
            return httpx.Response(
                404,
                json={"error": {"code": "trace_not_found", "message": "No trace 'x'"}},
            )

        make_client(handler).delete("8d3f1a2b")
        assert len(attempts) == 2

    @pytest.mark.parametrize("failure", ["gateway", "transport"])
    def test_hinted_delete_does_not_absorb_404_after_ambiguous_attempt(self, make_client, failure):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(dict(request.url.params))
            if len(attempts) == 1:
                if failure == "transport":
                    raise httpx.ReadError("connection reset", request=request)
                return httpx.Response(502, text="upstream response lost")
            return httpx.Response(
                404,
                json={"error": {"code": "trace_not_found", "message": "hint did not match"}},
            )

        with pytest.raises(NotFoundError):
            make_client(handler).delete("8d3f1a2b", created_at="2026-07-20T18:02:11.482Z")
        assert attempts == [
            {"created_at": "2026-07-20T18:02:11.482Z"},
            {"created_at": "2026-07-20T18:02:11.482Z"},
        ]

    @pytest.mark.parametrize(
        "error_type",
        [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout],
    )
    def test_404_after_a_pre_delivery_failure_is_not_absorbed(self, make_client, error_type):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                raise error_type("request was not sent", request=request)
            return httpx.Response(
                404,
                json={"error": {"code": "trace_not_found", "message": "No trace 'x'"}},
            )

        with pytest.raises(NotFoundError):
            make_client(handler).delete("8d3f1a2b")
        assert len(attempts) == 2

    @pytest.mark.parametrize("status", [502, 504])
    def test_404_after_a_gateway_failure_is_treated_as_deleted(self, make_client, status):
        """A gateway may have forwarded the request upstream before failing, so
        502/504 is as ambiguous as a dropped connection."""
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                return httpx.Response(status, text="upstream connect error")
            return httpx.Response(
                404,
                json={"error": {"code": "trace_not_found", "message": "No trace 'x'"}},
            )

        make_client(handler).delete("8d3f1a2b")
        assert len(attempts) == 2

    @pytest.mark.parametrize("status", [429, 503])
    def test_404_after_the_service_declined_is_not_absorbed(self, make_client, status):
        """429/503 are the service refusing the work, so nothing can have been
        deleted behind one. A 404 after it is a real "no such trace" and must
        surface — absorbing it would report a delete that never happened as
        accepted."""
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(request.url.path)
            if len(attempts) == 1:
                return httpx.Response(
                    status,
                    json={"error": {"code": "storage_unavailable", "message": "down"}},
                )
            return httpx.Response(
                404,
                json={"error": {"code": "run_not_found", "message": "No traces in run 'nope'"}},
            )

        with pytest.raises(NotFoundError) as caught:
            make_client(handler).delete_run("nope")
        assert caught.value.code == "run_not_found"
        assert len(attempts) == 2


class TestTeamHeader:
    @staticmethod
    def _client(team_id, handler):
        return TracesAPIClient(
            api_key="test-key",
            base_url="http://testserver",
            team_id=team_id,
            transport=httpx.MockTransport(handler),
        )

    def test_sent_when_team_id_given(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["team"] = request.headers.get("X-Prime-Team-ID")
            return httpx.Response(200, json={})

        self._client("team_123", handler).get_json("/traces")
        assert captured["team"] == "team_123"

    def test_absent_when_team_id_empty(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["team"] = request.headers.get("X-Prime-Team-ID")
            return httpx.Response(200, json={})

        self._client("", handler).get_json("/traces")
        assert captured["team"] is None
