"""Upload limits should match the bytes HTTPX actually sends."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from prime_evals.evals import AsyncEvalsClient, EvalsClient


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "sample_count,fit_count,expected_batch_sizes",
    [(1, 1, [1]), (3, 2, [2, 1])],
    ids=["single-sample-at-limit", "two-samples-per-batch"],
)
def test_push_samples_batches_by_actual_encoded_bytes(
    monkeypatch, is_async, sample_count, fit_count, expected_batch_sizes
):
    samples = [{"answer": "漢" * 10} for _ in range(sample_count)]
    max_payload_bytes = len(
        httpx.Request("POST", "/", json={"samples": samples[:fit_count]}).content
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={})

    async def async_respond(request):
        return respond(request)

    api_client = SimpleNamespace(base_url="https://api.example", api_key="test-key")
    if is_async:
        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            "prime_evals.evals.httpx.AsyncClient",
            lambda **kwargs: real_client(transport=httpx.MockTransport(async_respond), **kwargs),
        )
        client = AsyncEvalsClient.__new__(AsyncEvalsClient)
        client.client = api_client
        result = asyncio.run(
            client.push_samples(
                "eval-1", samples, max_payload_bytes=max_payload_bytes, max_concurrent=1
            )
        )
    else:
        real_client = httpx.Client
        monkeypatch.setattr(
            "prime_evals.evals.httpx.Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
        )
        result = EvalsClient(api_client).push_samples(
            "eval-1", samples, max_payload_bytes=max_payload_bytes, max_workers=1
        )

    assert result == {"samples_pushed": sample_count, "samples_skipped": 0}
    assert [
        len(json.loads(request.content)["samples"]) for request in requests
    ] == expected_batch_sizes
    assert all(len(request.content) <= max_payload_bytes for request in requests)
