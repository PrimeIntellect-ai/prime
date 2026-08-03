import httpx
import pytest

from prime_traces import NotFoundError, TracesAPIClient, UnauthorizedError

SUMMARY = {
    "trace_id": "8d3f1a2b",
    "upload_id": "5ee85e41",
    "created_at": "2026-07-20T18:02:11.482Z",
    "ingested_at": "2026-07-20T18:06:02.117Z",
    "run_id": "run_9f3k2m",
    "environment_id": "terminal-bench-2",
    "model": {"provider": "prime", "id": "deepseek-v4-flash"},
    "task_id": "tb2-0187",
    "score": {"reward": 0.85, "outcome": "done"},
    "execution": {"has_error": False, "is_truncated": False},
    "duration_ms": 215537,
    "context": {"source": "hosted_eval"},
}


class TestList:
    def test_filters_encode_as_documented_query_params(self, make_client):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json={"traces": [SUMMARY], "next_cursor": None})

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
        [summary] = page.traces
        assert summary.trace_id == "8d3f1a2b"
        assert summary.score.reward == 0.85
        assert summary.model.id == "deepseek-v4-flash"

    def test_unscored_reward_stays_none_not_zero(self, make_client):
        unscored = {**SUMMARY, "score": {"reward": None, "outcome": None}}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"traces": [unscored], "next_cursor": None})

        [summary] = make_client(handler).list().traces
        assert summary.score.reward is None

    def test_iter_follows_cursor(self, make_client):
        pages = {
            None: {"traces": [{**SUMMARY, "trace_id": "t1"}], "next_cursor": "c1"},
            "c1": {"traces": [{**SUMMARY, "trace_id": "t2"}], "next_cursor": None},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            cursor = request.url.params.get("cursor")
            return httpx.Response(200, json=pages[cursor])

        ids = [s.trace_id for s in make_client(handler).iter(run_id="run_9f3k2m")]
        assert ids == ["t1", "t2"]

    def test_tolerates_additive_summary_fields(self, make_client):
        grown = {**SUMMARY, "total_tokens": 123, "num_turns": 4}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"traces": [grown], "next_cursor": None})

        [summary] = make_client(handler).list().traces
        assert summary.trace_id == "8d3f1a2b"


class TestPointReads:
    def test_get_summary(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/traces/8d3f1a2b"
            return httpx.Response(200, json=SUMMARY)

        assert make_client(handler).get("8d3f1a2b").task_id == "tb2-0187"

    def test_get_raw_streams_document(self, make_client):
        raw = b'{"version":4,"id":"8d3f1a2b","nodes":[]}'

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["raw"] == "true"
            return httpx.Response(200, content=raw)

        assert make_client(handler).get_raw("8d3f1a2b") == raw

    def test_download_raw_writes_file(self, make_client, tmp_path):
        raw = b'{"version":4,"id":"8d3f1a2b"}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=raw)

        dest = tmp_path / "trace.json"
        written = make_client(handler).download_raw("8d3f1a2b", dest)
        assert written == len(raw)
        assert dest.read_bytes() == raw

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

    def test_delete_run_returns_job_id(self, make_client):
        def handler(request: httpx.Request) -> httpx.Response:
            assert dict(request.url.params) == {"run_id": "run_9f3k2m"}
            return httpx.Response(202, json={"job_id": "job-1"})

        assert make_client(handler).delete_run("run_9f3k2m") == "job-1"


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
            captured["team"] = request.headers.get("X-Prime-Team-Id")
            return httpx.Response(200, json={})

        self._client("team_123", handler).get_json("/traces")
        assert captured["team"] == "team_123"

    def test_absent_when_team_id_empty(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["team"] = request.headers.get("X-Prime-Team-Id")
            return httpx.Response(200, json={})

        self._client("", handler).get_json("/traces")
        assert captured["team"] is None
