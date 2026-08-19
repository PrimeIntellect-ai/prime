"""Transport behaviour: error mapping and retry."""

import httpx
import pytest

from prime_runs._http import PlatformClient, encode_json, retry_delay
from prime_runs.exceptions import (
    NotFoundError,
    PaymentRequiredError,
    RetryableAPIError,
    RunAPIError,
    TransportError,
    UnauthorizedError,
)


def client_for(handler, **kwargs) -> PlatformClient:
    return PlatformClient(
        api_key="test-key",
        base_url="http://testserver",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, UnauthorizedError),
        (402, PaymentRequiredError),
        (404, NotFoundError),
        (400, RunAPIError),
        (422, RunAPIError),
    ],
)
def test_status_codes_map_to_types_callers_can_branch_on(status, expected):
    client = client_for(lambda request: httpx.Response(status, json={"detail": "nope"}))

    with pytest.raises(expected) as caught:
        client.get("/evaluations/x")

    assert caught.value.status_code == status
    assert "nope" in str(caught.value)


def test_an_unauthorized_error_says_what_to_do_about_it():
    client = client_for(lambda request: httpx.Response(401, json={"detail": "bad token"}))

    with pytest.raises(UnauthorizedError, match="PRIME_API_KEY"):
        client.get("/evaluations/x")


def test_retryable_statuses_are_retried_then_surface(no_sleep):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, json={"code": "ingest_unavailable"})

    with pytest.raises(RetryableAPIError) as caught:
        client_for(handler, max_attempts=3).get("/evaluations/x")

    assert len(attempts) == 3
    assert caught.value.code == "ingest_unavailable"
    assert no_sleep == [1.0, 2.0]


def test_a_retry_succeeds_without_bothering_the_caller(no_sleep):
    responses = [httpx.Response(429), httpx.Response(200, json={"ok": True})]

    client = client_for(lambda request: responses.pop(0))

    assert client.get("/evaluations/x") == {"ok": True}


def test_retry_after_beats_the_exponential_schedule():
    assert retry_delay(1, 7.5) == 7.5
    assert retry_delay(1, None) == 1.0
    assert retry_delay(4, None) == 8.0
    # Never wait longer than the ceiling, whatever the server asked for.
    assert retry_delay(1, 900.0) == 16.0


def test_transport_failures_are_retried_and_typed(no_sleep):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(TransportError):
        client_for(handler, max_attempts=2).get("/evaluations/x")

    assert len(no_sleep) == 1


def test_an_empty_body_is_a_valid_response():
    client = client_for(lambda request: httpx.Response(204))

    assert client.post("/evaluations/x/finalize", json_body={"metrics": {}}) == {}


def test_a_non_json_body_names_the_request_that_produced_it():
    client = client_for(lambda request: httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(RunAPIError, match="non-JSON"):
        client.get("/evaluations/x")


def test_encoding_refuses_values_json_cannot_carry():
    """Bare ``NaN`` is JavaScript, not JSON; it comes back as an opaque 400."""
    with pytest.raises(ValueError):
        encode_json({"reward": float("nan")})
