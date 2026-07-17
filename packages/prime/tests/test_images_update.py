"""Tests for `prime images update` and the publish/unpublish bulk transports."""

from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
}


def _success_response(json):
    return {
        "success": True,
        "dryRun": json.get("dryRun", False),
        "results": [
            {
                "source": update["source"],
                "success": True,
                "before": {
                    "owner": {"type": "personal"},
                    "name": update["source"].get("name", "app"),
                    "tag": update["source"].get("tag", "v1"),
                    "visibility": "PRIVATE",
                },
                "after": {
                    "owner": update["set"].get("owner", {"type": "personal"}),
                    "name": update["set"].get("name") or update["source"].get("name", "app"),
                    "tag": update["set"].get("tag") or update["source"].get("tag", "v1"),
                    "visibility": update["set"].get("visibility", "PRIVATE"),
                },
            }
            for update in json["updates"]
        ],
    }


def _patch_client(monkeypatch, captured, response_builder=_success_response):
    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            return response_builder(json)

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))


def test_update_rename_personal_image(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "update", "app:v1", "--name", "renamed", "--tag", "v2"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "PATCH"
    assert captured["path"] == "/images"
    assert captured["json"]["mode"] == "explicit"
    assert captured["json"]["updates"] == [
        {
            "source": {"owner": {"type": "personal"}, "name": "app", "tag": "v1"},
            "set": {"name": "renamed", "tag": "v2"},
        }
    ]
    assert "Updated" in result.output


def test_update_uses_team_context_for_plain_reference(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "update", "app:v1", "--public"],
        env={**TEST_ENV, "PRIME_TEAM_ID": "team-ctx"},
    )

    assert result.exit_code == 0, result.output
    assert captured["json"]["updates"][0]["source"] == {
        "owner": {"type": "team", "teamId": "team-ctx"},
        "name": "app",
        "tag": "v1",
    }
    assert captured["json"]["updates"][0]["set"] == {"visibility": "PUBLIC"}


def test_update_owner_prefixed_reference_passes_reference_form(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "update", "prime/alice/app:v1", "--private"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["json"]["updates"][0]["source"] == {"reference": "prime/alice/app:v1"}


def test_update_move_to_team_requires_confirmation(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "update", "app:v1", "--to-team", "team_123"],
        env=TEST_ENV,
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Update cancelled" in result.output
    assert "json" not in captured  # nothing was sent

    result = runner.invoke(
        app,
        ["images", "update", "app:v1", "--to-team", "team_123", "--yes"],
        env=TEST_ENV,
    )
    assert result.exit_code == 0, result.output
    assert captured["json"]["updates"][0]["set"] == {
        "owner": {"type": "team", "teamId": "team_123"}
    }


def test_update_to_platform_conflicts(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "update", "app:v1", "--to-platform", "--private"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "always public" in result.output

    result = runner.invoke(
        app,
        ["images", "update", "app:v1", "--to-platform", "--to-team", "t1"],
        env=TEST_ENV,
    )
    assert result.exit_code == 1
    assert "cannot be used together" in result.output


def test_update_requires_at_least_one_change(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured)

    result = runner.invoke(app, ["images", "update", "app:v1"], env=TEST_ENV)
    assert result.exit_code == 1
    assert "at least one change" in result.output


def test_update_dry_run_previews_without_confirmation(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "update", "app:v1", "--to-team", "team_123", "--dry-run"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["json"]["dryRun"] is True
    assert "Dry run" in result.output
    assert "No changes were applied" in result.output


def test_update_failure_reports_error_and_exits_nonzero(monkeypatch):
    captured = {}

    def failing_response(json):
        return {
            "success": False,
            "dryRun": False,
            "results": [
                {
                    "source": json["updates"][0]["source"],
                    "success": False,
                    "error": {
                        "code": "destination_exists",
                        "message": "Destination image renamed:v1 already exists",
                    },
                }
            ],
        }

    _patch_client(monkeypatch, captured, failing_response)

    result = runner.invoke(
        app,
        ["images", "update", "app:v1", "--name", "renamed"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "already exists" in result.output


def test_publish_multiple_references_batches_explicit_updates(monkeypatch):
    captured_requests = []

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured_requests.append({"method": method, "path": path, "json": json})
            return {
                "success": True,
                "dryRun": False,
                "results": [
                    {"source": update["source"], "success": True} for update in json["updates"]
                ],
            }

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(
        app,
        ["images", "publish", "a:v1", "b:v2", "c:v3"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request["method"] == "PATCH"
    assert request["path"] == "/images"
    assert request["json"]["updates"] == [
        {
            "source": {"owner": {"type": "personal"}, "name": name, "tag": tag},
            "set": {"visibility": "PUBLIC"},
        }
        for name, tag in [("a", "v1"), ("b", "v2"), ("c", "v3")]
    ]
    assert "Made 3 image(s) public" in result.output
    assert "✓ a:v1" in result.output


def test_publish_multiple_references_reports_partial_failures(monkeypatch):
    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            results = []
            for update in json["updates"]:
                if update["source"]["name"] == "missing":
                    results.append(
                        {
                            "source": update["source"],
                            "success": False,
                            "error": {
                                "code": "image_not_found",
                                "message": "Image missing:latest not found",
                            },
                        }
                    )
                else:
                    results.append({"source": update["source"], "success": True})
            return {"success": False, "dryRun": False, "results": results}

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(
        app,
        ["images", "publish", "a:v1", "missing:latest"],
        env=TEST_ENV,
    )

    assert result.exit_code == 1
    assert "✓ a:v1" in result.output
    assert "✗ missing:latest: Image missing:latest not found" in result.output


def test_publish_search_uses_search_mode_with_confirmation(monkeypatch):
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            if method == "GET" and path == "/images":
                captured["preview_params"] = params
                return {"totalCount": 2, "data": []}
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            return {
                "success": True,
                "dryRun": False,
                "results": [
                    {
                        "source": {
                            "owner": {"type": "personal"},
                            "name": name,
                            "tag": "v1",
                        },
                        "success": True,
                    }
                    for name in ("exp-a", "exp-b")
                ],
            }

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(
        app,
        ["images", "publish", "--search", "exp-", "--yes"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "PATCH"
    assert captured["path"] == "/images"
    assert captured["json"]["mode"] == "search"
    assert captured["json"]["selection"] == {
        "owner": {"type": "personal"},
        "search": "exp-",
    }
    assert captured["json"]["set"] == {"visibility": "PUBLIC"}
    assert "Made 2 image(s) public" in result.output


def test_publish_search_team_context_sets_team_owner(monkeypatch):
    captured = {}

    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            if method == "GET" and path == "/images":
                return {"totalCount": 1, "data": []}
            captured["json"] = json
            return {"success": True, "dryRun": False, "results": []}

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))

    result = runner.invoke(
        app,
        ["images", "unpublish", "--search", "exp-", "--yes"],
        env={**TEST_ENV, "PRIME_TEAM_ID": "team-ctx"},
    )

    assert result.exit_code == 0, result.output
    assert captured["json"]["selection"] == {
        "owner": {"type": "team", "teamId": "team-ctx"},
        "search": "exp-",
    }
    assert captured["json"]["set"] == {"visibility": "PRIVATE"}
    assert "No images matched" in result.output
