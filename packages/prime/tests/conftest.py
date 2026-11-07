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
