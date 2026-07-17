"""Tests for `prime images update-bulk`."""

import json

import pytest
from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
}


def _write_manifest(tmp_path, lines):
    manifest = tmp_path / "updates.jsonl"
    manifest.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return manifest


def _rename_line(name, new_name):
    return {
        "source": {"owner": {"type": "personal"}, "name": name, "tag": "v1"},
        "set": {"name": new_name},
    }


def _patch_success_client(monkeypatch, captured_requests):
    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured_requests.append({"method": method, "path": path, "json": json})
            return {
                "success": True,
                "dryRun": json.get("dryRun", False),
                "results": [
                    {"source": update["source"], "success": True} for update in json["updates"]
                ],
            }

    monkeypatch.setattr("prime_cli.commands.images_update_bulk.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))


def test_update_bulk_applies_manifest(tmp_path, monkeypatch):
    captured = []
    _patch_success_client(monkeypatch, captured)
    manifest = _write_manifest(tmp_path, [_rename_line("a", "a2"), _rename_line("b", "b2")])

    result = runner.invoke(
        app,
        ["images", "update-bulk", "--manifest", str(manifest)],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0]["method"] == "PATCH"
    assert captured[0]["path"] == "/images"
    assert captured[0]["json"]["mode"] == "explicit"
    assert len(captured[0]["json"]["updates"]) == 2
    assert "Updated:" in result.output and "Failed:" in result.output


def test_update_bulk_validates_all_lines_before_sending(tmp_path, monkeypatch):
    captured = []
    _patch_success_client(monkeypatch, captured)
    manifest = tmp_path / "updates.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(_rename_line("a", "a2")),
                "not-json",
                json.dumps({"source": {"reference": "x:v1"}, "set": {}}),
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        ["images", "update-bulk", "--manifest", str(manifest)],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "cannot start bulk update" in result.output
    assert "line 2" in result.output
    assert "line 3" in result.output
    assert captured == []  # nothing was sent


@pytest.mark.parametrize(
    ("section", "unknown_key"),
    [
        ("source", "nmae"),
        ("set", "visibilty"),
        ("source.owner", "scope"),
        ("set.owner", "scope"),
    ],
)
def test_update_bulk_rejects_nested_unknown_keys_before_sending(
    tmp_path, monkeypatch, section, unknown_key
):
    captured = []
    _patch_success_client(monkeypatch, captured)
    update = _rename_line("a", "a2")
    if section == "set.owner":
        update["set"]["owner"] = {"type": "personal"}
    target = update
    for part in section.split("."):
        target = target[part]
    target[unknown_key] = "unexpected"
    manifest = _write_manifest(tmp_path, [update])

    result = runner.invoke(
        app,
        ["images", "update-bulk", "--manifest", str(manifest)],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert f"{section}: unknown key(s): {unknown_key}" in result.output
    assert captured == []


def test_update_bulk_detects_local_duplicates(tmp_path, monkeypatch):
    captured = []
    _patch_success_client(monkeypatch, captured)
    manifest = _write_manifest(
        tmp_path,
        [
            _rename_line("a", "a2"),
            _rename_line("a", "a3"),  # duplicate source
            _rename_line("b", "a2"),  # duplicate destination
        ],
    )

    result = runner.invoke(
        app,
        ["images", "update-bulk", "--manifest", str(manifest)],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "duplicate source" in result.output
    assert "duplicate destination" in result.output
    assert captured == []


def test_update_bulk_dry_run_uses_server_dry_run(tmp_path, monkeypatch):
    captured = []
    _patch_success_client(monkeypatch, captured)
    manifest = _write_manifest(tmp_path, [_rename_line("a", "a2")])

    result = runner.invoke(
        app,
        ["images", "update-bulk", "--manifest", str(manifest), "--dry-run"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured[0]["json"]["dryRun"] is True
    assert "Would update" in result.output


def test_update_bulk_owner_move_requires_confirmation(tmp_path, monkeypatch):
    captured = []
    _patch_success_client(monkeypatch, captured)
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "source": {"owner": {"type": "personal"}, "name": "a", "tag": "v1"},
                "set": {"owner": {"type": "team", "teamId": "team_123"}},
            }
        ],
    )

    result = runner.invoke(
        app,
        ["images", "update-bulk", "--manifest", str(manifest)],
        env=TEST_ENV,
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Update cancelled" in result.output
    assert captured == []


def test_update_bulk_writes_failures_manifest(tmp_path, monkeypatch):
    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            results = []
            for update in json["updates"]:
                if update["source"]["name"] == "b":
                    results.append(
                        {
                            "source": update["source"],
                            "success": False,
                            "error": {
                                "code": "destination_exists",
                                "message": "Destination image b2:v1 already exists",
                            },
                        }
                    )
                else:
                    results.append({"source": update["source"], "success": True})
            return {"success": False, "dryRun": False, "results": results}

    monkeypatch.setattr("prime_cli.commands.images_update_bulk.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    manifest = _write_manifest(tmp_path, [_rename_line("a", "a2"), _rename_line("b", "b2")])
    failures_out = tmp_path / "failed.jsonl"

    result = runner.invoke(
        app,
        [
            "images",
            "update-bulk",
            "--manifest",
            str(manifest),
            "--failures-out",
            str(failures_out),
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "already exists" in result.output
    assert failures_out.exists()
    retry_lines = [json.loads(line) for line in failures_out.read_text().splitlines() if line]
    assert retry_lines == [_rename_line("b", "b2")]


def test_update_bulk_transport_failure_records_unsent(tmp_path, monkeypatch):
    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            from prime_sandboxes import APIError

            raise APIError("HTTP 500: boom")

    monkeypatch.setattr("prime_cli.commands.images_update_bulk.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    manifest = _write_manifest(tmp_path, [_rename_line("a", "a2")])
    failures_out = tmp_path / "failed.jsonl"

    result = runner.invoke(
        app,
        [
            "images",
            "update-bulk",
            "--manifest",
            str(manifest),
            "--failures-out",
            str(failures_out),
        ],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "boom" in result.output
    assert "Not sent:" in result.output
    retry_lines = [json.loads(line) for line in failures_out.read_text().splitlines() if line]
    assert retry_lines == [_rename_line("a", "a2")]
