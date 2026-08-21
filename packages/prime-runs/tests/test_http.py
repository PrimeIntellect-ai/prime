"""Transport behaviour: error mapping and retry."""

import httpx
import pytest

from prime_runs._http import PlatformClient, encode_json
from prime_runs.exceptions import (
    APIError,
    ForbiddenError,
    NotFoundError,
    PaymentRequiredError,
    RetryableAPIError,
    TransportError,
    UnauthorizedError,
    is_transient,
)


def client_for(handler, *, base_url: str = "http://testserver", **kwargs) -> PlatformClient:
    return PlatformClient(
        api_key="test-key",
        base_url=base_url,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, UnauthorizedError),
        (402, PaymentRequiredError),
        (403, ForbiddenError),
        (404, NotFoundError),
        (400, APIError),
        (422, APIError),
    ],
)
def test_status_codes_map_to_types_callers_can_branch_on(status, expected):
    """The classes are ``prime_traces``' own, so a producer that already
    handles the traces client's errors handles these with the same clauses."""
    client = client_for(lambda request: httpx.Response(status, json={"detail": "nope"}))

    with pytest.raises(expected) as caught:
        client.get("/evaluations/x")

    assert caught.value.status_code == status
    if status not in (401, 402):
        assert "nope" in str(caught.value)


def test_a_forbidden_response_is_permanent_so_a_sink_retires_on_it():
    """403 is the gated-account signal. Retrying it for the rest of a run would
    log one failure per batch and never succeed."""
    client = client_for(lambda request: httpx.Response(403, json={"code": "service_not_enabled"}))

    with pytest.raises(ForbiddenError) as caught:
        client.get("/evaluations/x")

    assert not is_transient(caught.value)


def test_an_unauthorized_error_says_what_to_do_about_it():
    client = client_for(lambda request: httpx.Response(401, json={"detail": "bad token"}))

    with pytest.raises(UnauthorizedError, match="PRIME_API_KEY"):
        client.get("/evaluations/x")


def test_retryable_statuses_are_retried_then_surface(no_sleep):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, json={"detail": "overloaded"})

    with pytest.raises(RetryableAPIError) as caught:
        client_for(handler, max_attempts=3).get("/evaluations/x")

    assert len(attempts) == 3
    assert caught.value.status_code == 503
    # Two waits between three attempts, on prime_traces' jittered schedule.
    assert len(no_sleep) == 2
    assert all(0.0 < delay <= 30.0 for delay in no_sleep)


def test_a_retry_succeeds_without_bothering_the_caller(no_sleep):
    responses = [httpx.Response(429), httpx.Response(200, json={"ok": True})]

    client = client_for(lambda request: responses.pop(0))

    assert client.get("/evaluations/x") == {"ok": True}


def test_retry_after_is_honoured(no_sleep):
    """The schedule itself is ``prime_traces.core.client.retry_delay``; what is
    ours is feeding it the server's header."""
    responses = [
        httpx.Response(429, headers={"Retry-After": "7.5"}),
        httpx.Response(200, json={"ok": True}),
    ]

    client_for(lambda request: responses.pop(0)).get("/evaluations/x")

    assert no_sleep == [7.5]


def test_transport_failures_are_retried_and_typed(no_sleep):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(TransportError):
        client_for(handler, max_attempts=2).get("/evaluations/x")

    assert len(no_sleep) == 1


def test_an_omitted_request_timeout_keeps_the_clients_default():
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(17.0, connect=3.0),
    )
    client = PlatformClient(
        api_key="test-key",
        base_url="http://testserver",
        client=http_client,
    )

    assert client.get("/evaluations/x") == {"ok": True}
    assert seen == [{"connect": 3.0, "read": 17.0, "write": 17.0, "pool": 17.0}]
    http_client.close()


def test_an_empty_body_is_a_valid_response():
    client = client_for(lambda request: httpx.Response(204))

    assert client.post("/evaluations/x/finalize", json_body={"metrics": {}}) == {}


def test_a_non_json_body_names_the_request_that_produced_it():
    client = client_for(lambda request: httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(APIError, match="non-JSON"):
        client.get("/evaluations/x")


def test_encoding_refuses_values_json_cannot_carry():
    """Bare ``NaN`` is JavaScript, not JSON; it comes back as an opaque 400."""
    with pytest.raises(ValueError):
        encode_json({"reward": float("nan")})


def test_an_ambiguous_failure_does_not_replay_a_create(no_sleep):
    """If the platform created the run and the response was lost, a retry makes
    a second one and only the second is tracked — an orphaned duplicate run."""
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(502, text="bad gateway")

    with pytest.raises(RetryableAPIError):
        client_for(handler, max_attempts=5).post("/evaluations/", json_body={"name": "r"})

    assert len(attempts) == 1
    assert no_sleep == []


def test_a_refusal_is_replayed_even_for_a_create(no_sleep):
    """429 is refused before the server does any work, so there is nothing on
    the other side to duplicate."""
    responses = [httpx.Response(429), httpx.Response(201, json={"evaluation_id": "e1"})]

    client = client_for(lambda request: responses.pop(0))

    assert client.post("/evaluations/", json_body={"name": "r"})["evaluation_id"] == "e1"


def test_a_connection_that_was_never_made_is_replayed_for_a_create(no_sleep):
    """Nothing reached the server, so replaying cannot duplicate anything."""
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(201, json={"evaluation_id": "e1"})

    assert client_for(handler).post("/evaluations/", json_body={"name": "r"})


def test_a_read_timeout_does_not_replay_a_create(no_sleep):
    """The bytes went out; the platform may have processed them."""
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(TransportError):
        client_for(handler, max_attempts=5).post("/evaluations/", json_body={"name": "r"})

    assert len(calls) == 1


def test_a_post_declared_idempotent_still_retries(no_sleep):
    """Get-or-create and terminal-state writes are safe to replay."""
    responses = [httpx.Response(502), httpx.Response(200, json={"data": {"id": "env-1"}})]

    client = client_for(lambda request: responses.pop(0))

    assert client.post("/environmentshub/resolve", json_body={}, idempotent=True)


@pytest.mark.parametrize(
    "given",
    [
        "http://testserver",
        "http://testserver/",
        "http://testserver/api/v1",
        "http://testserver/api/v1/",
    ],
)
def test_a_base_url_written_with_the_api_prefix_is_not_doubled(given):
    """``Config`` strips the suffix, so an explicit ``base_url=`` that does not
    would 404 on exactly the value that works through the environment."""
    assert client_for(lambda request: httpx.Response(200, json={}), base_url=given).api_prefix == (
        "http://testserver/api/v1"
    )


def test_idempotent_methods_still_replay_ambiguous_failures(no_sleep):
    responses = [httpx.Response(504), httpx.Response(200, json={"ok": True})]

    client = client_for(lambda request: responses.pop(0))

    assert client.put("/evaluations/x", json_body={"metrics": {}}) == {"ok": True}
