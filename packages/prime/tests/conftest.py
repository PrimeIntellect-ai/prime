import pytest


@pytest.fixture(autouse=True)
def _auto_detect_terminal(monkeypatch):
    # Rich treats FORCE_COLOR (any value) as a terminal; with TERM=dumb it then
    # fixes the width at 80 and ignores the COLUMNS the tests pass.
    monkeypatch.delenv("FORCE_COLOR", raising=False)


@pytest.fixture
def stub_local_schema_validation(monkeypatch):
    monkeypatch.setattr(
        "prime_cli.api.training.HostedTrainingClient.prepare_local_validation",
        lambda self, payload: {"mode": payload["mode"], "submission_supported": True},
    )
    monkeypatch.setattr(
        "prime_cli.utils.training_compiler.compile_locally",
        lambda config, preparation: {"mode": preparation["mode"], "schema_commit": "a" * 40},
    )


@pytest.fixture(autouse=True)
def _no_user_identity_agent(monkeypatch):
    """Keep the user's real ~/.ssh/config (ssh -G) out of volume ssh tests."""
    from prime_cli.commands import volumes

    monkeypatch.setattr(volumes, "_identity_agent", lambda host: None)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A fake Hub API that every APIClient talks to, for push tests."""
    import httpx
    from fake_hub import API, FakeHub
    from prime_cli.core.client import APIClient
    from prime_cli.utils import hub as hub_module

    fake = FakeHub()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.setenv("PRIME_API_BASE_URL", API)
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    original_init = APIClient.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.client = httpx.Client(
            transport=httpx.MockTransport(fake.handle), headers=self.client.headers
        )

    monkeypatch.setattr(APIClient, "__init__", init)

    def put(url, content, headers, timeout):
        fake.uploads[url.rsplit("/", 1)[-1]] = content.read()
        return httpx.Response(200, request=httpx.Request("PUT", url))

    monkeypatch.setattr(hub_module.httpx, "put", put)
    return fake
