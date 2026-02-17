from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from prime_tunnel import Config, Tunnel, TunnelClient
from prime_tunnel.models import TunnelInfo


def test_tunnel_init():
    """Test Tunnel initialization."""
    tunnel = Tunnel(local_port=8080)
    assert tunnel.local_port == 8080
    assert tunnel.local_addr == "127.0.0.1"
    assert tunnel.name is None
    assert not tunnel.is_running


def test_tunnel_init_with_name():
    """Test Tunnel initialization with name."""
    tunnel = Tunnel(local_port=9000, name="my-tunnel")
    assert tunnel.local_port == 9000
    assert tunnel.name == "my-tunnel"


def test_config_default_base_url():
    """Test Config default base URL."""
    config = Config()
    assert config.DEFAULT_BASE_URL == "https://api.primeintellect.ai"


def test_config_base_url_from_env(monkeypatch):
    """Test Config base_url from environment variable."""
    monkeypatch.setenv("PRIME_BASE_URL", "https://custom.example.com")
    config = Config()
    assert config.base_url == "https://custom.example.com"


def test_config_base_url_strips_api_v1(monkeypatch):
    """Test Config strips /api/v1 from base URL."""
    monkeypatch.setenv("PRIME_BASE_URL", "https://example.com/api/v1")
    config = Config()
    assert config.base_url == "https://example.com"


def test_config_api_key_from_env(monkeypatch):
    """Test Config api_key from environment variable."""
    monkeypatch.setenv("PRIME_API_KEY", "test-key-123")
    config = Config()
    assert config.api_key == "test-key-123"


def test_config_bin_dir():
    """Test Config bin_dir property."""
    config = Config()
    assert config.bin_dir.name == "bin"
    assert ".prime" in str(config.bin_dir)


def test_tunnel_client_init():
    """Test TunnelClient initialization."""
    client = TunnelClient(api_key="test-key")
    assert client.api_key == "test-key"
    assert client.base_url == Config.DEFAULT_BASE_URL


def _make_started_tunnel() -> Tunnel:
    """Create a Tunnel that looks like it was started."""
    tunnel = Tunnel(local_port=8080)
    tunnel._started = True
    tunnel._process = MagicMock()
    tunnel._tunnel_info = TunnelInfo(
        tunnel_id="t-test123",
        hostname="t-test123.tunnel.example.com",
        url="https://t-test123.tunnel.example.com",
        frp_token="tok",
        proxy_name="test-user",
        server_host="frp.example.com",
        server_port=7000,
        expires_at=datetime.now(timezone.utc),
    )
    tunnel._config_file = MagicMock()
    tunnel._config_file.exists.return_value = True
    return tunnel


def test_sync_stop_noop_when_not_started():
    tunnel = Tunnel(local_port=8080)
    tunnel.sync_stop()  # should not raise


def test_sync_stop_terminates_process_and_deletes_registration():
    tunnel = _make_started_tunnel()
    process_mock = tunnel._process

    with patch("prime_tunnel.tunnel.httpx.delete") as mock_delete:
        tunnel.sync_stop()

    process_mock.terminate.assert_called_once()
    mock_delete.assert_called_once()
    assert "t-test123" in mock_delete.call_args[0][0]
    assert tunnel._started is False
    assert tunnel._tunnel_info is None
    assert tunnel._process is None


def test_sync_stop_kills_on_timeout():
    tunnel = _make_started_tunnel()
    process_mock = tunnel._process
    import subprocess

    process_mock.wait.side_effect = subprocess.TimeoutExpired(cmd="frpc", timeout=5)

    with patch("prime_tunnel.tunnel.httpx.delete"):
        tunnel.sync_stop()

    process_mock.terminate.assert_called_once()
    process_mock.kill.assert_called_once()
    assert tunnel._started is False


def test_sync_stop_survives_delete_failure():
    tunnel = _make_started_tunnel()

    with patch("prime_tunnel.tunnel.httpx.delete", side_effect=Exception("network")):
        tunnel.sync_stop()  # should not raise

    assert tunnel._started is False
    assert tunnel._tunnel_info is None
