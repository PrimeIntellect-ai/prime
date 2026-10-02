import json
from unittest.mock import Mock

import pytest
from prime_cli.api.training import HostedTrainingClient, Volume
from prime_cli.commands import volumes
from prime_cli.core import APIError
from prime_cli.main import app
from typer.testing import CliRunner

ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}


def response():
    return {
        "name": "models-b",
        "size": "2Ti",
        "status": "PENDING",
        "clusterId": "cluster-b",
        "cluster": "e2e-spk",
        "pvcName": "vol-models-b",
    }


@pytest.mark.parametrize("size,team", [(None, None), ("3Ti", "team")])
def test_copy_client_uses_dedicated_endpoint_and_optional_fields(size, team):
    api = Mock()
    api.post.return_value = response()
    result = HostedTrainingClient(api).copy_volume(
        "models-a", "models-b", "e2e-spk", size=size, team_id=team
    )
    payload = {"name": "models-b", "cluster": "e2e-spk"}
    if size is not None:
        payload["size"] = size
    if team is not None:
        payload["teamId"] = team
    api.post.assert_called_once_with("/training/volumes/models-a/copy", json=payload)
    assert result == Volume.model_validate(response())


def test_copy_client_encodes_source_path():
    api = Mock()
    api.post.return_value = response()
    HostedTrainingClient(api).copy_volume("invalid/source", "models-b", "e2e-spk")
    api.post.assert_called_once_with(
        "/training/volumes/invalid%2Fsource/copy", json={"name": "models-b", "cluster": "e2e-spk"}
    )


def test_copy_cli_reports_accepted_not_completed(monkeypatch):
    api = Mock()
    api.post.return_value = response()
    monkeypatch.setattr(volumes, "_client", lambda: (HostedTrainingClient(api), "team"))
    result = CliRunner().invoke(
        app, ["volumes", "copy", "models-a", "models-b", "--cluster", "e2e-spk"], env=ENV
    )
    assert result.exit_code == 0, result.output
    api.post.assert_called_once_with(
        "/training/volumes/models-a/copy",
        json={"name": "models-b", "cluster": "e2e-spk", "teamId": "team"},
    )
    output = " ".join(result.output.split())
    assert "PENDING" in output and "source is retained" in output
    assert "only when it is RUNNING" in output


def test_copy_cli_size_and_json(monkeypatch):
    api = Mock()
    api.post.return_value = response()
    monkeypatch.setattr(volumes, "_client", lambda: (HostedTrainingClient(api), None))
    result = CliRunner().invoke(
        app,
        [
            "volumes",
            "copy",
            "models-a",
            "models-b",
            "--cluster",
            "e2e-spk",
            "--size",
            "3Ti",
            "-o",
            "json",
        ],
        env=ENV,
    )
    assert result.exit_code == 0, result.output
    api.post.assert_called_once_with(
        "/training/volumes/models-a/copy",
        json={"name": "models-b", "cluster": "e2e-spk", "size": "3Ti"},
    )
    assert json.loads(result.output) == Volume.model_validate(response()).model_dump(by_alias=True)


def test_copy_requires_destination_cluster_before_api_call(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(volumes, "_client", factory)
    result = CliRunner().invoke(app, ["volumes", "copy", "models-a", "models-b"], env=ENV)
    assert result.exit_code == 2
    factory.assert_not_called()


def test_copy_error_does_not_fall_back_to_empty_volume_create(monkeypatch):
    api = Mock()
    api.post.side_effect = APIError("copy not available [server]")
    monkeypatch.setattr(volumes, "_client", lambda: (HostedTrainingClient(api), None))
    result = CliRunner().invoke(
        app, ["volumes", "copy", "models-a", "models-b", "--cluster", "e2e-spk"], env=ENV
    )
    assert result.exit_code == 1, result.output
    assert "copy not available [server]" in result.output
    assert "Copy requested" not in result.output
    assert api.post.call_count == 1
    api.delete.assert_not_called()
