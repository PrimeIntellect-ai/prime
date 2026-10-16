import pytest


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
