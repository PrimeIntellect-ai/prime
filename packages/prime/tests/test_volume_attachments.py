import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prime_cli.api.training import HostedTrainingClient, Volume, VolumeAttachment
from prime_cli.commands import volumes
from prime_cli.core import APIError
from prime_cli.main import app
from typer.testing import CliRunner

ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}


def test_create_cli_passes_the_cluster_and_shows_the_server_backend(monkeypatch):
    body = {
        "name": "models",
        "size": "1Ti",
        "status": "PENDING",
        "clusterId": "a",
        "cluster": "telus",
        "pvcName": "vol-models",
        "backend": "juicefs",
    }
    api = Mock()
    api.post.return_value = body
    monkeypatch.setattr(volumes, "_client", lambda: (HostedTrainingClient(api), "team"))
    result = CliRunner().invoke(
        app,
        ["volumes", "create", "models", "--cluster", "telus", "-o", "json"],
        env=ENV,
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == Volume.model_validate(body).model_dump(
        mode="json", by_alias=True
    )
    api.post.assert_called_once_with(
        "/training/volumes",
        json={
            "name": "models",
            "size": "1Ti",
            "teamId": "team",
            "cluster": "telus",
        },
    )


@pytest.mark.parametrize("server_backend", ["juicefs", "cluster"])
def test_create_leaves_the_default_backend_to_the_server(monkeypatch, server_backend):
    body = {
        "name": "models",
        "size": "1Ti",
        "status": "PENDING",
        "clusterId": "a",
        "cluster": "telus",
        "pvcName": "vol-models",
        "backend": server_backend,
    }
    api = Mock()
    api.post.return_value = body
    monkeypatch.setattr(volumes, "_client", lambda: (HostedTrainingClient(api), None))
    result = CliRunner().invoke(app, ["volumes", "create", "models"], env=ENV)
    assert result.exit_code == 0, result.output
    api.post.assert_called_once_with("/training/volumes", json={"name": "models", "size": "1Ti"})
    attach_hint = "prime volumes attach models --cluster <name>" in result.output
    assert attach_hint == (server_backend == "juicefs")


@pytest.mark.parametrize("as_json", [False, True])
def test_attach_cli_uses_target_cluster_and_returns_pending(monkeypatch, as_json):
    body = {"clusterId": "b", "cluster": "e2e-spk", "status": "PENDING"}
    api = Mock()
    api.post.return_value = body
    monkeypatch.setattr(volumes, "_client", lambda: (HostedTrainingClient(api), "team"))
    args = ["volumes", "attach", "models", "--cluster", "e2e-spk"]
    if as_json:
        args += ["-o", "json"]
    result = CliRunner().invoke(app, args, env=ENV)
    assert result.exit_code == 0, result.output
    api.post.assert_called_once_with(
        "/training/volumes/models/attachments", json={"cluster": "e2e-spk", "teamId": "team"}
    )
    if as_json:
        assert json.loads(result.output) == body
    else:
        assert "e2e-spk" in result.output and "PENDING" in result.output


def test_detach_cli_keeps_data_and_sends_owner_context(monkeypatch):
    api = Mock()
    monkeypatch.setattr(volumes, "_client", lambda: (HostedTrainingClient(api), "team"))
    result = CliRunner().invoke(
        app, ["volumes", "detach", "models", "--cluster", "e2e-spk"], env=ENV
    )
    assert result.exit_code == 0, result.output
    assert "data is retained" in result.output
    api.delete.assert_called_once_with(
        "/training/volumes/models/attachments/e2e-spk", params={"teamId": "team"}
    )


@pytest.mark.parametrize("operation", ["attach", "detach"])
def test_attachment_errors_are_reported_without_success(monkeypatch, operation):
    method = Mock(side_effect=APIError("volume has active runs [busy]"))
    client = SimpleNamespace(**{f"{operation}_volume": method})
    monkeypatch.setattr(volumes, "_client", lambda: (client, None))
    result = CliRunner().invoke(
        app, ["volumes", operation, "models", "--cluster", "e2e-spk"], env=ENV
    )
    assert result.exit_code == 1, result.output
    assert "volume has active runs [busy]" in result.output
    assert "data is retained" not in result.output


def test_attachment_paths_are_encoded_and_personal_scope_is_omitted():
    api = Mock()
    body = {"clusterId": "b", "status": "RUNNING"}
    api.post.return_value = body
    client = HostedTrainingClient(api)
    assert client.attach_volume("model/data", "cluster b") == VolumeAttachment.model_validate(body)
    client.detach_volume("model/data", "cluster/b")
    api.post.assert_called_once_with(
        "/training/volumes/model%2Fdata/attachments", json={"cluster": "cluster b"}
    )
    api.delete.assert_called_once_with(
        "/training/volumes/model%2Fdata/attachments/cluster%2Fb", params=None
    )


def test_list_exposes_portable_backend_and_binding_status(monkeypatch):
    body = {
        "name": "models",
        "size": "1Ti",
        "status": "RUNNING",
        "clusterId": "a",
        "cluster": "telus",
        "pvcName": "vol-models",
        "backend": "juicefs",
        "attachments": [{"clusterId": "b", "cluster": "e2e-spk", "status": "PENDING"}],
    }
    api = Mock()
    api.get.return_value = {"volumes": [body]}
    monkeypatch.setattr(volumes, "_client", lambda: (HostedTrainingClient(api), "team"))
    table = CliRunner().invoke(app, ["volumes", "list"], env=ENV)
    assert table.exit_code == 0, table.output
    assert "juicefs" in table.output and "e2e-spk (PENDING)" in table.output
    result = CliRunner().invoke(app, ["volumes", "list", "-o", "json"], env=ENV)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        Volume.model_validate(body).model_dump(mode="json", by_alias=True)
    ]
