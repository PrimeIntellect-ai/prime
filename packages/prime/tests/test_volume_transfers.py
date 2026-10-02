"""`prime volumes export/import/transfers` — the S3 transfer CLI (ENG-6450).

Hermetic: a fake HostedTrainingClient and monkeypatched AWS env; no network.
"""

import json
from types import SimpleNamespace

import pytest
from prime_cli.api.training import HostedTrainingClient, VolumeTransfer
from prime_cli.commands import volumes
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Pin the module console to a wide, deterministic size.

    The console is built once at import, and rich ignores a width set without
    a height (it falls back to the terminal / dumb-terminal 80x25), so a
    width-only setting would make these table and progress assertions depend
    on the terminal the tests happen to run under.
    """
    monkeypatch.setattr(volumes.console, "_width", 200)
    monkeypatch.setattr(volumes.console, "_height", 50)


def _run(*args):
    return CliRunner().invoke(app, ["volumes", *args], env=ENV)


def _text(result) -> str:
    """Result output with ANSI stripped, so assertions hold on a tty too."""
    return strip_ansi(result.output)


def _transfer(**over) -> VolumeTransfer:
    fields = {
        "id": "t1",
        "volumeName": "data",
        "direction": "export",
        "path": "checkpoints/step-1000",
        "url": "s3://bucket/prefix",
        "status": "running",
        "createdAt": "2026-10-01T07:15:03Z",
    }
    fields.update(over)
    return VolumeTransfer.model_validate(fields)


def _client(monkeypatch, **methods):
    """Install a fake client; `methods` are the transfer methods to expose."""
    monkeypatch.setattr(volumes, "_client", lambda: (SimpleNamespace(**methods), "team1"))


def _aws_env(monkeypatch, token=None):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shh-secret")
    if token is None:
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    else:
        monkeypatch.setenv("AWS_SESSION_TOKEN", token)


def test_client_transfer_wire_contract():
    wire = {
        "id": "t1",
        "volumeName": "data",
        "direction": "export",
        "path": "p",
        "url": "s3://b/x",
        "status": "pending",
    }
    requests = []

    class FakeAPI:
        def post(self, path, json):
            requests.append(("POST", path, json))
            return wire

        def get(self, path, params):
            requests.append(("GET", path, params))
            if path.endswith("/transfers"):
                return {"transfers": [wire]}
            return wire

        def delete(self, path, params):
            requests.append(("DELETE", path, params))

    client = HostedTrainingClient(FakeAPI())
    credentials = {"accessKeyId": "AKIA", "secretAccessKey": "s", "sessionToken": None}

    created = client.create_volume_transfer(
        "data",
        direction="export",
        path="p",
        url="s3://b/x",
        credentials=credentials,
        region="us-east-1",
        endpoint_url="https://minio.local",
        team_id="team1",
    )
    assert created.id == "t1"
    listed = client.list_volume_transfers("data", team_id="team1")
    assert [t.id for t in listed] == ["t1"]
    assert client.get_volume_transfer("data", "t1", team_id="team1").url == "s3://b/x"
    assert client.cancel_volume_transfer("data", "t1", team_id="team1") is None

    assert requests == [
        (
            "POST",
            "/training/volumes/data/transfers",
            {
                "direction": "export",
                "path": "p",
                "url": "s3://b/x",
                "credentials": credentials,
                "region": "us-east-1",
                "endpointUrl": "https://minio.local",
                "teamId": "team1",
            },
        ),
        ("GET", "/training/volumes/data/transfers", {"teamId": "team1"}),
        ("GET", "/training/volumes/data/transfers/t1", {"teamId": "team1"}),
        ("DELETE", "/training/volumes/data/transfers/t1", {"teamId": "team1"}),
    ]


def test_client_omits_unset_options():
    posted = []
    wire = {
        "id": "t1",
        "volumeName": "data",
        "direction": "import",
        "path": "",
        "url": "s3://b",
        "status": "pending",
    }
    api = SimpleNamespace(post=lambda path, json=None: posted.append(json) or wire)
    client = HostedTrainingClient(api)
    client.create_volume_transfer("data", direction="import", path="", url="s3://b", credentials={})
    assert posted[0] == {"direction": "import", "path": "", "url": "s3://b", "credentials": {}}


def test_export_payload_and_hints(monkeypatch):
    created = []
    _client(
        monkeypatch,
        create_volume_transfer=lambda *a, **kw: created.append((a, kw)) or _transfer(),
    )
    _aws_env(monkeypatch)
    result = _run(
        "export", "data", "checkpoints/step-1000", "s3://bucket/prefix", "--region", "us-east-1"
    )
    assert result.exit_code == 0, _text(result)

    assert created[0][0] == ("data",)
    assert created[0][1] == {
        "direction": "export",
        "path": "checkpoints/step-1000",
        "url": "s3://bucket/prefix",
        "credentials": {
            "accessKeyId": "AKIAEXAMPLE",
            "secretAccessKey": "shh-secret",
            "sessionToken": None,
        },
        "region": "us-east-1",
        "endpoint_url": None,
        "team_id": "team1",
    }
    assert "Started export of data:checkpoints/step-1000 to s3://bucket/prefix" in _text(result)
    assert "Transfer id: t1" in _text(result)
    assert "prime volumes transfers list data --id t1" in _text(result)
    assert "prime volumes transfers list data --id t1 --follow" in _text(result)
    # Credentials are sent once, never echoed.
    assert "shh-secret" not in _text(result)


def test_export_forwards_region_and_session_token(monkeypatch):
    created = []
    _client(
        monkeypatch,
        create_volume_transfer=lambda *a, **kw: created.append((a, kw)) or _transfer(),
    )
    _aws_env(monkeypatch, token="session-token")
    result = _run(
        "export",
        "data",
        "p",
        "s3://b",
        "--region",
        "eu-west-1",
        "--endpoint-url",
        "https://s3.example",
    )
    assert result.exit_code == 0, _text(result)
    payload = created[0][1]
    assert payload["region"] == "eu-west-1"
    assert payload["endpoint_url"] == "https://s3.example"
    assert payload["credentials"]["sessionToken"] == "session-token"


def test_import_destination_is_last(monkeypatch):
    created = []
    _client(
        monkeypatch,
        create_volume_transfer=lambda *a, **kw: created.append((a, kw))
        or _transfer(direction="import"),
    )
    _aws_env(monkeypatch)
    result = _run("import", "data", "s3://bucket/prefix", "datasets/sft-data")
    assert result.exit_code == 0, _text(result)
    assert created[0][0] == ("data",)
    assert created[0][1]["direction"] == "import"
    assert created[0][1]["path"] == "datasets/sft-data"
    assert created[0][1]["url"] == "s3://bucket/prefix"
    assert "Started import of data:datasets/sft-data from s3://bucket/prefix" in _text(result)


@pytest.mark.parametrize("missing", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
def test_missing_aws_env_exits_before_any_call(monkeypatch, missing):
    def boom():
        raise AssertionError("the API client must not be built without credentials")

    monkeypatch.setattr(volumes, "_client", boom)
    _aws_env(monkeypatch)
    monkeypatch.delenv(missing, raising=False)
    result = _run("export", "data", "p", "s3://b")
    assert result.exit_code == 1
    assert f"{missing} is not set" in _text(result)
    assert "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN" in _text(result)


def test_transfers_table(monkeypatch):
    running = _transfer(
        progress={
            "transferred": "412.345 GiB",
            "total": "1.500 TiB",
            "percentage": 27,
            "rate": "182.400 MiB/s",
            "eta": "1h48m0s",
            "errors": 0,
        }
    )
    done = _transfer(
        id="t2",
        direction="import",
        path="datasets/sft-data",
        url="s3://datasets-bucket/raw",
        status="succeeded",
        progress={"transferred": "1.500 TiB", "total": "1.500 TiB", "errors": 0},
        startedAt="2026-10-01T07:00:00Z",
        completedAt="2026-10-01T09:21:00Z",
    )
    _client(monkeypatch, list_volume_transfers=lambda name, **kw: [running, done])
    result = _run("transfers", "list", "data")
    assert result.exit_code == 0, _text(result)
    for header in ("ID", "DIRECTION", "PATH", "URL", "STATUS", "PROGRESS"):
        assert header in _text(result)
    assert "412.3 GiB / 1.5 TiB (27%), 182.4 MiB/s, ETA 1h48m" in _text(result)
    assert "1.5 TiB in 2h21m" in _text(result)
    assert "s3://datasets-bucket/raw" in _text(result)


def test_transfers_output_json_is_a_list(monkeypatch):
    _client(monkeypatch, list_volume_transfers=lambda name, **kw: [_transfer(), _transfer(id="t2")])
    result = _run("transfers", "list", "data", "--json")
    assert result.exit_code == 0, _text(result)
    data = json.loads(_text(result))
    assert [t["id"] for t in data] == ["t1", "t2"]
    assert data[0]["volumeName"] == "data"


def test_transfers_id_shows_one_object(monkeypatch):
    calls = []
    _client(
        monkeypatch,
        get_volume_transfer=lambda name, tid, **kw: calls.append((name, tid, kw))
        or _transfer(id=tid),
    )
    result = _run("transfers", "list", "data", "--id", "t7", "--json")
    assert result.exit_code == 0, _text(result)
    assert calls == [("data", "t7", {"team_id": "team1"})]
    data = json.loads(_text(result))
    assert isinstance(data, dict) and data["id"] == "t7"


def test_transfers_follow_polls_to_terminal(monkeypatch):
    states = [
        _transfer(
            progress={
                "transferred": "412.345 GiB",
                "total": "1.500 TiB",
                "percentage": 27,
                "rate": "182.400 MiB/s",
                "eta": "1h48m0s",
            }
        ),
        _transfer(
            status="succeeded",
            progress={"transferred": "1.500 TiB", "total": "1.500 TiB"},
            startedAt="2026-10-01T07:00:00Z",
            completedAt="2026-10-01T09:21:00Z",
        ),
    ]
    monkeypatch.setattr(volumes.time, "sleep", lambda s: None)
    _client(monkeypatch, get_volume_transfer=lambda name, tid, **kw: states.pop(0))
    result = _run("transfers", "list", "data", "--id", "t1", "--follow")
    assert result.exit_code == 0, _text(result)
    lines = [ln for ln in _text(result).splitlines() if ln.strip()]
    assert lines == [
        "running    412.3 GiB / 1.5 TiB (27%), 182.4 MiB/s, ETA 1h48m",
        "succeeded  1.5 TiB in 2h21m",
    ]


def test_transfers_follow_survives_a_failed_poll(monkeypatch):
    from prime_cli.core import APIError

    states = [APIError("502 Bad Gateway"), _transfer(status="cancelled")]
    monkeypatch.setattr(volumes.time, "sleep", lambda s: None)
    _client(monkeypatch, get_volume_transfer=lambda name, tid, **kw: _raise_or_pop(states))
    result = _run("transfers", "list", "data", "--id", "t1", "--follow")
    assert result.exit_code == 0, _text(result)
    assert "cancelled" in _text(result)


def _raise_or_pop(states):
    item = states.pop(0)
    if isinstance(item, Exception):
        raise item
    return item


@pytest.mark.parametrize(
    "args",
    [
        ["transfers", "list", "data", "--follow"],
        ["transfers", "list", "data", "--id", "t1", "--follow", "--json"],
    ],
)
def test_follow_requires_id_and_human_output(monkeypatch, args):
    def boom():
        raise AssertionError("no client call for a usage error")

    monkeypatch.setattr(volumes, "_client", boom)
    result = _run(*args)
    assert result.exit_code == 2


def test_transfers_cancel(monkeypatch):
    cancelled = []
    _client(
        monkeypatch,
        cancel_volume_transfer=lambda name, tid, **kw: cancelled.append((name, tid, kw)),
    )
    result = _run("transfers", "cancel", "data", "t9")
    assert result.exit_code == 0, _text(result)
    assert cancelled == [("data", "t9", {"team_id": "team1"})]
    assert "Cancelling transfer t9." in _text(result)


def test_progress_helpers():
    assert volumes._short_number("412.345 GiB") == "412.3 GiB"
    assert volumes._short_number("1.500 TiB") == "1.5 TiB"
    assert volumes._short_number("182.400 MiB/s") == "182.4 MiB/s"
    assert volumes._short_eta("1h48m0s") == "1h48m"
    assert volumes._short_eta("0h48m0s") == "0h48m"
    assert volumes._short_eta("30s") == "30s"
    assert volumes._duration("2026-10-01T07:00:00Z", "2026-10-01T09:21:00Z") == "2h21m"
    assert volumes._duration(None, "2026-10-01T09:21:00Z") is None
    running = _transfer(
        progress={
            "transferred": "412.345 GiB",
            "total": "1.500 TiB",
            "percentage": 27,
            "rate": "182.400 MiB/s",
            "eta": "1h48m0s",
        }
    )
    assert volumes._progress_cell(running) == "412.3 GiB / 1.5 TiB (27%), 182.4 MiB/s, ETA 1h48m"
    done = _transfer(
        status="succeeded",
        progress={"total": "1.500 TiB"},
        startedAt="2026-10-01T07:00:00Z",
        completedAt="2026-10-01T09:21:00Z",
    )
    assert volumes._progress_cell(done) == "1.5 TiB in 2h21m"
    assert volumes._progress_cell(_transfer(status="pending")) == "-"
    assert volumes._progress_cell(_transfer(status="failed", errorMessage="rclone exited 1")) == (
        "rclone exited 1"
    )
