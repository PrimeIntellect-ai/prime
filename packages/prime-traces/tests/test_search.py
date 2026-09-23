import httpx
import pytest

from prime_traces import NotFoundError

MATCH = {
    "trace_id": "trace-1",
    "upload_id": "upload-1",
    "generation": 2,
    "episode_id": None,
    "node_idx": 4,
    "role": "tool",
    "field": "content",
    "representation": "text",
    "match_start": 2,
    "match_end": 7,
    "excerpt": "🙂 hello world",
    "excerpt_start": 0,
}
FIRST_PAGE = {
    "items": [],
    "next_cursor": "continue",
    "coverage": {
        "examined_traces": 256,
        "unindexed_trace_ids": ["pending"],
        "partial_index": True,
    },
}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_search_contract_and_continuation(make_client, make_async_client, asynchronous):
    calls = []

    def handler(request):
        assert request.url.raw_path.split(b"?")[0] == b"/api/v1/runs/run%20%231/search"
        params = dict(request.url.params)
        assert params["query"] == "🙂 hello %_"
        assert "run_id" not in params
        assert params["field"] == "content"
        assert params["role"] == "tool"
        assert params["run_step"] == "0"
        assert params["has_error"] == "false"
        assert params["reward_min"] == "0.0"
        calls.append(params)
        return httpx.Response(
            200,
            json=FIRST_PAGE
            if len(calls) == 1
            else {"items": [MATCH], "next_cursor": None, "coverage": None},
        )

    client = (make_async_client if asynchronous else make_client)(handler)
    kwargs = dict(run_id="run #1", role="tool", run_step=0, has_error=False, reward_min=0.0)
    page = (
        await client.search("🙂 hello %_", **kwargs)
        if asynchronous
        else client.search("🙂 hello %_", **kwargs)
    )
    assert len(calls) == 1  # One page per call; the SDK never paginates implicitly.
    assert page.items == [] and page.next_cursor == "continue"
    assert page.coverage.partial_index and page.coverage.unindexed_trace_ids == ["pending"]
    kwargs["cursor"] = page.next_cursor
    page = (
        await client.search("🙂 hello %_", **kwargs)
        if asynchronous
        else client.search("🙂 hello %_", **kwargs)
    )
    assert calls[1]["cursor"] == "continue"
    assert page.items[0].match_start == 2 and page.next_cursor is None
    assert page.coverage is None


@pytest.mark.parametrize("code", [None, "trace_not_found", "run_not_found"])
def test_search_preserves_not_found_error(make_client, code):
    def handler(request):
        return httpx.Response(
            404,
            json={"detail": "Not Found"}
            if code is None
            else {"error": {"code": code, "message": "not found"}},
        )

    with pytest.raises(NotFoundError) as exc:
        make_client(handler).search("hello", run_id="run")
    assert exc.value.code == code


def test_search_refuses_a_run_id_that_cannot_be_one_path_segment(make_client):
    def handler(request):
        pytest.fail("an unaddressable run must not make a network request")

    with pytest.raises(ValueError, match="run_id"):
        make_client(handler).search("hello", run_id="team/run")
