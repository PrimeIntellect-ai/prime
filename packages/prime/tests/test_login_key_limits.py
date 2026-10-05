import base64
from typing import Any, Dict, List, Optional

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
}


class FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeBackend:
    """Stands in for the challenge endpoints, approving on the first poll."""

    def __init__(self, echo_limits: bool) -> None:
        self.echo_limits = echo_limits
        self.generate_bodies: List[Dict[str, Any]] = []
        self.polls = 0

    def post(self, url: str, json: Dict[str, Any]) -> FakeResponse:
        self.generate_bodies.append(json)
        self.public_pem = json["encryptionPublicKey"]
        payload: Dict[str, Any] = {"challenge": "code", "status_auth_token": "status"}
        if self.echo_limits:
            payload["limits"] = {"maxSandboxGpuCount": None, **json.get("limits", {})}
        return FakeResponse(payload)

    def get(self, url: str, params: Dict[str, Any], headers: Dict[str, str]) -> FakeResponse:
        self.polls += 1
        public_key = serialization.load_pem_public_key(self.public_pem.encode())
        encrypted = public_key.encrypt(  # type: ignore[union-attr]
            b"pit_minted",
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return FakeResponse({"result": base64.b64encode(encrypted).decode()})


@pytest.fixture
def backend(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, request: Any) -> FakeBackend:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    fake = FakeBackend(echo_limits=getattr(request, "param", True))
    monkeypatch.setattr("prime_cli.commands.login.httpx.post", fake.post)
    monkeypatch.setattr("prime_cli.commands.login.httpx.get", fake.get)

    def mock_get(self: Any, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if endpoint == "/user/whoami":
            return {"data": {"id": "user-id", "name": "Test User"}}
        return {"data": [], "total_count": 0}

    monkeypatch.setattr("prime_cli.commands.login.APIClient.get", mock_get)
    return fake


def test_login_sends_requested_key_limits(backend: FakeBackend) -> None:
    result = runner.invoke(
        app,
        ["login", "--headless", "--max-concurrent-sandboxes", "10", "--max-tunnel-ttl-hours", "2"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert backend.generate_bodies[0]["limits"] == {
        "maxConcurrentSandboxes": 10,
        "maxTunnelTtlHours": 2,
    }
    assert "Successfully logged in!" in result.output
    assert "maxConcurrentSandboxes=10" in result.output


def test_login_without_limit_flags_sends_no_limits(backend: FakeBackend) -> None:
    result = runner.invoke(app, ["login", "--headless"], env=TEST_ENV)

    assert result.exit_code == 0, result.output
    assert "limits" not in backend.generate_bodies[0]
    assert "API key limits" not in result.output


@pytest.mark.parametrize("backend", [False], indirect=True)
def test_login_aborts_when_server_ignores_key_limits(backend: FakeBackend) -> None:
    result = runner.invoke(
        app, ["login", "--headless", "--max-concurrent-sandboxes", "10"], env=TEST_ENV
    )

    assert result.exit_code == 1
    assert "does not support API key limits" in result.output
    # No key may be minted, let alone saved, once the limits were dropped.
    assert backend.polls == 0


def test_login_rejects_out_of_range_limit(backend: FakeBackend) -> None:
    result = runner.invoke(app, ["login", "--max-tunnel-ttl-hours", "0"], env=TEST_ENV)

    assert result.exit_code != 0
    assert backend.generate_bodies == []
