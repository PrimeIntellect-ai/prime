from prime_tunnel import Config, Tunnel, TunnelClient, TunnelConfig


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


def test_tunnel_config():
    """Test TunnelConfig model."""
    config = TunnelConfig(local_port=8888, local_addr="0.0.0.0", name="test")
    assert config.local_port == 8888
    assert config.local_addr == "0.0.0.0"
    assert config.name == "test"


def test_tunnel_config_defaults():
    """Test TunnelConfig default values."""
    config = TunnelConfig()
    assert config.local_port == 8765
    assert config.local_addr == "127.0.0.1"
    assert config.name is None


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
