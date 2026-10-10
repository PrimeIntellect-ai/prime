"""Hermetic read-only volume object commands; no sessions, writes, or network."""

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from types import SimpleNamespace

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from prime_cli.api.training import Volume, VolumeTransferRoute
from prime_cli.commands import volumes
from prime_cli.core import APIError
from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

ENV = {"PRIME_DISABLE_VERSION_CHECK": "1", "COLUMNS": "200"}
MODIFIED = datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)


def _error(code="404", operation="HeadObject"):
    return ClientError({"Error": {"Code": code, "Message": "object unavailable"}}, operation)


class FakeBody:
    def __init__(self, content, failure=None):
        self.content = content
        self.failure = failure
        self.closed = False
        self.chunk_sizes = []

    def iter_chunks(self, chunk_size):
        self.chunk_sizes.append(chunk_size)
        if self.failure:
            raise self.failure
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def close(self):
        self.closed = True


class FakeS3:
    """Faithful delimiter pages, including pages containing only common prefixes."""

    def __init__(self, objects, page_size=2):
        self.objects = objects
        self.page_size = page_size
        self.requests = []
        self.closed = False
        self.body = None
        self.failure = None

    def _request(self, operation, kwargs):
        assert kwargs["Bucket"] == "b"
        assert kwargs.get("Key", kwargs.get("Prefix", "")).startswith("vol1/")
        self.requests.append((operation, kwargs))
        if self.failure:
            raise self.failure

    def head_object(self, **kwargs):
        self._request("head", kwargs)
        key = kwargs["Key"]
        if key not in self.objects:
            raise _error()
        return {"ContentLength": len(self.objects[key]), "LastModified": MODIFIED}

    def list_objects_v2(self, **kwargs):
        self._request("list", kwargs)
        prefix = kwargs["Prefix"]
        entries = {}
        for key, value in self.objects.items():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            if kwargs.get("Delimiter") and "/" in suffix:
                directory = prefix + suffix.split("/", 1)[0] + "/"
                entries[directory] = ("directory", {"Prefix": directory})
            else:
                entries[key] = (
                    "file",
                    {"Key": key, "Size": len(value), "LastModified": MODIFIED},
                )
        ordered = [entries[key] for key in sorted(entries)]
        start = int(kwargs.get("ContinuationToken", "0"))
        page = ordered[start : start + self.page_size]
        end = start + len(page)
        result = {
            "Contents": [entry for kind, entry in page if kind == "file"],
            "CommonPrefixes": [entry for kind, entry in page if kind == "directory"],
            "IsTruncated": end < len(ordered),
        }
        if result["IsTruncated"]:
            result["NextContinuationToken"] = str(end)
        return result

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"

        def paginate(**kwargs):
            while True:
                page = self.list_objects_v2(**kwargs)
                yield page
                if not page["IsTruncated"]:
                    break
                kwargs["ContinuationToken"] = page["NextContinuationToken"]

        return SimpleNamespace(paginate=paginate)

    def get_object(self, **kwargs):
        self._request("get", kwargs)
        key = kwargs["Key"]
        if key not in self.objects:
            raise _error(operation="GetObject")
        self.body = self.body or FakeBody(self.objects[key])
        return {"Body": self.body}

    def close(self):
        self.closed = True


@pytest.fixture
def storage(monkeypatch):
    s3 = FakeS3(
        {
            "vol1/root.txt": b"abc",
            "vol1/data/a.txt": b"12345",
            "vol1/data/nested/b.bin": b"abcdefghijk",
            "vol1/empty/": b"",
            "vol1/database/c.txt": b"1234567",
            "vol1/runs/job/output.bin": b"1234567890123",
            "vol1/runs/.sessions/private/state": b"hidden" * 200,
            "vol1/[red]literal.txt": b"xx",
            "other-volume/leak": b"never visible",
        }
    )
    route = VolumeTransferRoute(
        via="r2",
        bucket="b",
        prefix="vol1/",
        endpoint="https://r2.example",
        access_key_id="key",
        secret_access_key="secret",
        session_token="token",
    )
    routing = []
    listing = []

    def get_route(*args, **kwargs):
        routing.append((args, kwargs))
        return route

    def list_volumes(**kwargs):
        listing.append(kwargs)
        return [Volume(name="data", size="5Ti", status="RUNNING", clusterId="c", pvcName="p")]

    client = SimpleNamespace(route_volume_transfer=get_route, list_volumes=list_volumes)
    monkeypatch.setattr(volumes, "_client", lambda: (client, "team1"))
    monkeypatch.setattr(volumes, "_r2_client", lambda refresh, route: s3)
    for console in (volumes.console, volumes.err_console):
        monkeypatch.setattr(console, "_width", 200)
        monkeypatch.setattr(console, "_height", 50)
    return SimpleNamespace(s3=s3, route=route, routing=routing, listing=listing, client=client)


def _run(*args):
    return CliRunner().invoke(app, ["volumes", *args], env=ENV)


def _json(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _paths(result):
    return {entry["path"]: entry for entry in _json(result)}


def test_ls_root_delimiter_and_pagination(storage):
    result = _run("ls", "data", "--output", "json")
    entries = _paths(result)
    assert set(entries) == {"[red]literal.txt", "root.txt", "data/", "database/", "empty/", "runs/"}
    assert entries["root.txt"] == {
        "path": "root.txt",
        "size": 3,
        "modified": MODIFIED.isoformat(),
        "type": "file",
    }
    assert entries["data/"] == {
        "path": "data/",
        "size": None,
        "modified": None,
        "type": "directory",
    }
    requests = [kw for op, kw in storage.s3.requests if op == "list"]
    assert len(requests) >= 3
    assert all(kw["Prefix"] == "vol1/" and kw["Delimiter"] == "/" for kw in requests)
    assert storage.routing == [(("data", "get", "", None), {"team_id": "team1"})]
    assert storage.s3.closed
    assert result.stderr
    assert "secret" not in result.output


@pytest.mark.parametrize("operand", ["data", "data/", "/data/", "./data/", "data/nested/../"])
def test_ls_directory_operand_normalization(storage, operand):
    assert set(_paths(_run("ls", "data", operand, "-o", "json"))) == {"data/a.txt", "data/nested/"}
    requests = [kw for op, kw in storage.s3.requests if op == "list"]
    assert all(kw["Prefix"] == "vol1/data/" for kw in requests)


def test_ls_file_operand_uses_head(storage):
    entries = _paths(_run("ls", "data", "root.txt", "-o", "json"))
    assert list(entries) == ["root.txt"]
    assert entries["root.txt"]["size"] == 3
    assert storage.s3.requests == [("head", {"Bucket": "b", "Key": "vol1/root.txt"})]


def test_ls_recursive_infers_directories_and_excludes_internal_metadata(storage):
    entries = _paths(_run("ls", "data", "-R", "-o", "json"))
    assert set(entries) == {
        "[red]literal.txt",
        "root.txt",
        "data/",
        "data/a.txt",
        "data/nested/",
        "data/nested/b.bin",
        "database/",
        "database/c.txt",
        "empty/",
        "runs/",
        "runs/job/",
        "runs/job/output.bin",
    }
    assert entries["empty/"]["type"] == "directory"
    assert all("Delimiter" not in kw for op, kw in storage.s3.requests if op == "list")


def test_ls_empty_directory(storage):
    assert _json(_run("ls", "data", "empty/", "-o", "json")) == []


@pytest.mark.parametrize("flags", [[], ["--plain"], ["-l"], ["--long", "--human-readable"]])
def test_ls_text_literal_names(storage, flags):
    result = _run("ls", "data", *flags)
    assert result.exit_code == 0, result.output
    assert "[red]literal.txt" in strip_ansi(result.stdout)
    assert "root.txt" in result.stdout
    assert "sync" not in result.stdout.lower()
    assert result.stderr


def test_ls_short_names_relative_to_operand(storage):
    result = _run("ls", "data", "data/")
    assert result.exit_code == 0, result.output
    assert "a.txt" in result.stdout and "nested/" in result.stdout
    assert "data/a.txt" not in result.stdout


def test_ls_short_h_flag_is_human_readable(storage):
    storage.s3.objects["vol1/root.txt"] = b"x" * 4096
    result = _run("ls", "data", "root.txt", "-l", "-h")
    assert result.exit_code == 0, result.output
    assert "Usage:" not in result.stdout
    assert "4" in result.stdout and ("KiB" in result.stdout or "Ki" in result.stdout)


def test_existing_list_command_keeps_volume_inventory(storage):
    result = _run("list", "--output", "json")
    assert result.exit_code == 0, result.output
    assert _json(result)[0]["name"] == "data"
    assert storage.listing == [{"team_id": "team1"}]
    assert storage.routing == []


@pytest.mark.parametrize(
    ("operand", "path", "byte_count", "object_count"),
    [
        ("/", "/", 41, 7),
        ("data", "data", 16, 2),
        ("data/", "data/", 16, 2),
        ("root.txt", "root.txt", 3, 1),
        ("empty/", "empty/", 0, 1),
        ("runs", "runs", 13, 1),
    ],
)
def test_du_subtree_whole_volume_usage_and_capacity(
    storage, operand, path, byte_count, object_count
):
    result = _run("du", "data", operand, "-o", "json")
    usage = _json(result)
    assert usage == {
        "path": path,
        "bytes": byte_count,
        "objects": object_count,
        "volumeBytes": 41,
        "volumeObjects": 7,
        "capacity": "5Ti",
        "capacityBytes": 5 * 1024**4,
        "usedPercent": pytest.approx(41 / (5 * 1024**4) * 100, rel=1e-12, abs=0),
    }
    requests = [kw for op, kw in storage.s3.requests if op == "list"]
    assert len(requests) >= 4
    assert all(kw["Prefix"] == "vol1/" and "Delimiter" not in kw for kw in requests)
    assert storage.listing == [{"team_id": "team1"}]
    assert storage.s3.closed
    assert result.stderr


def test_du_human_readable(storage):
    storage.s3.objects["vol1/root.txt"] = b"x" * 4096
    result = _run("du", "data", "root.txt", "-h")
    assert result.exit_code == 0, result.output
    assert "Usage:" not in result.stdout
    assert "4" in result.stdout and ("KiB" in result.stdout or "Ki" in result.stdout)


def test_cat_binary_stream_and_cleanup(storage):
    payload = bytes(range(256)) * 1025
    storage.s3.objects["vol1/binary.bin"] = payload
    result = _run("cat", "data", "/binary.bin")
    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == payload
    assert storage.s3.body.chunk_sizes == [64 * 1024]
    assert storage.s3.body.closed and storage.s3.closed
    assert result.stderr
    assert storage.routing == [(("data", "get", "", None), {"team_id": "team1"})]


def test_cat_empty_file(storage):
    storage.s3.objects["vol1/zero.txt"] = b""
    result = _run("cat", "data", "zero.txt")
    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == b""
    assert storage.s3.body.closed and storage.s3.closed


def test_cat_stream_error_closes_body_and_client(storage):
    storage.s3.body = FakeBody(b"", failure=OSError("read failed"))
    result = _run("cat", "data", "root.txt")
    assert result.exit_code == 1, result.output
    assert result.stdout_bytes == b""
    assert "read failed" in result.stderr
    assert storage.s3.body.closed and storage.s3.closed


@pytest.mark.parametrize("command", ["ls", "du", "cat"])
def test_missing_operand_errors_on_stderr(storage, command):
    result = _run(command, "data", "missing")
    assert result.exit_code == 1, result.output
    assert result.stdout_bytes == b""
    assert result.stderr
    assert storage.s3.closed


@pytest.mark.parametrize("operand", ["data/", "empty/", "/"])
def test_cat_directory_errors(storage, operand):
    result = _run("cat", "data", operand)
    assert result.exit_code == 2, result.output
    assert result.stdout_bytes == b""
    assert result.stderr


@pytest.mark.parametrize("command", ["ls", "du", "cat"])
@pytest.mark.parametrize("operand", ["../outside", "bad name", "runs/.sessions/private", "x" * 256])
def test_invalid_path_fails_before_credentials(storage, command, operand):
    result = _run(command, "data", operand)
    assert result.exit_code in (1, 2), result.output
    assert result.stdout_bytes == b""
    assert result.stderr
    assert storage.routing == []
    assert storage.s3.requests == []


def test_object_key_total_byte_limit_includes_volume_prefix(storage):
    storage.route.prefix = "v" * 500 + "/"
    operand = "/".join(["p" * 200] * 3)
    result = _run("cat", "data", operand)
    assert result.exit_code == 2, result.output
    assert result.stdout_bytes == b""
    assert storage.s3.requests == []


@pytest.mark.parametrize("command", ["ls", "du", "cat"])
def test_credential_failure_is_safe_and_uses_stderr(storage, command):
    def denied(*args, **kwargs):
        raise APIError("volume access denied")

    storage.client.route_volume_transfer = denied
    result = _run(command, "data", "root.txt")
    assert result.exit_code == 1, result.output
    assert result.stdout_bytes == b""
    assert "access denied" in result.stderr
    assert storage.s3.requests == []


@pytest.mark.parametrize("command", ["ls", "du", "cat"])
def test_object_authorization_failure_closes_client(storage, command):
    storage.s3.failure = _error("AccessDenied")
    result = _run(command, "data", "root.txt")
    assert result.exit_code == 1, result.output
    assert result.stdout_bytes == b""
    assert result.stderr
    assert storage.s3.closed


def test_du_unknown_capacity_preserves_usage(storage):
    storage.client.list_volumes = lambda **kwargs: [
        Volume(name="data", size=None, status="RUNNING", clusterId="c", pvcName="p")
    ]
    usage = _json(_run("du", "data", "-o", "json"))
    assert usage["bytes"] == 41
    assert usage["objects"] == 7
    assert usage["capacity"] is None
    assert usage["capacityBytes"] is None
    assert usage["usedPercent"] is None


@pytest.mark.parametrize("command", ["ls", "du", "cat"])
def test_session_route_is_refused_without_session_operations(storage, command):
    storage.route.via = "session"
    result = _run(command, "data", "root.txt")
    assert result.exit_code == 1, result.output
    assert result.stdout_bytes == b""
    assert result.stderr
    assert storage.s3.requests == []


def test_ls_real_botocore_paginator_keeps_prefix_only_pages(storage, monkeypatch):
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url="https://r2.example",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
    )
    monkeypatch.setattr(volumes, "_r2_client", lambda refresh, route: s3)
    with Stubber(s3) as stub:
        stub.add_response(
            "list_objects_v2",
            {
                "IsTruncated": True,
                "NextContinuationToken": "next",
                "CommonPrefixes": [
                    {"Prefix": "vol1/empty/"},
                    {"Prefix": "vol1/data/"},
                ],
            },
            {"Bucket": "b", "Prefix": "vol1/", "Delimiter": "/"},
        )
        stub.add_response(
            "list_objects_v2",
            {
                "IsTruncated": False,
                "Contents": [
                    {
                        "Key": "vol1/root.txt",
                        "Size": 3,
                        "LastModified": MODIFIED,
                    }
                ],
            },
            {"Bucket": "b", "Prefix": "vol1/", "Delimiter": "/", "ContinuationToken": "next"},
        )
        assert set(_paths(_run("ls", "data", "-o", "json"))) == {"data/", "empty/", "root.txt"}
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("command", ["ls", "du"])
@pytest.mark.parametrize("output", ["table", "json"])
def test_colon_filename_is_never_expanded_as_emoji(storage, plain, command, output):
    path = ":smile:.txt"
    storage.s3.objects["vol1/" + path] = b"content"
    args = [command, "data", path, "-o", output]
    if plain:
        args.append("--plain")
    result = _run(*args)
    assert result.exit_code == 0, result.output
    if output == "json":
        data = _json(result)
        assert (data[0] if command == "ls" else data)["path"] == path
    else:
        assert path in result.stdout
    assert "😄" not in result.stdout


def test_cat_closed_stdout_pipe_exits_cleanly():
    """Exercise a real OS pipe: CliRunner's in-memory stdout cannot raise EPIPE."""
    script = textwrap.dedent("""
        from types import SimpleNamespace
        from prime_cli.api.training import VolumeTransferRoute
        from prime_cli.commands import volumes
        from prime_cli.main import app

        route = VolumeTransferRoute(
            via="r2", bucket="b", prefix="vol1/", endpoint="https://r2.example",
            access_key_id="test-key", secret_access_key="test-secret",
        )
        client = SimpleNamespace(route_volume_transfer=lambda *args, **kwargs: route)
        body = SimpleNamespace(
            iter_chunks=lambda chunk_size: iter([b"x" * (128 * 1024)]),
            close=lambda: None,
        )
        s3 = SimpleNamespace(get_object=lambda **kwargs: {"Body": body}, close=lambda: None)
        volumes._client = lambda: (client, "team1")
        volumes._r2_client = lambda refresh, route: s3
        app(args=["volumes", "cat", "data", "file.bin"])
    """)
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        env={**os.environ, **ENV},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        process.stdout.close()
        code = process.wait(timeout=10)
        stderr = process.stderr.read().decode()
        assert code == 0, stderr
        assert "Traceback" not in stderr
        assert "BrokenPipeError" not in stderr
        assert "next sync" in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stderr.close()
