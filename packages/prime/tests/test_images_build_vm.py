from prime_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

TEST_ENV = {
    "COLUMNS": "200",
    "LINES": "50",
    "PRIME_DISABLE_VERSION_CHECK": "1",
    "PRIME_TEAM_ID": "",
}


def _patch_api(monkeypatch, captured):
    class DummyAPIClient:
        def request(self, method, path, json=None, params=None):
            captured["method"] = method
            captured["path"] = path
            captured["json"] = json
            return {"buildId": "build-123"}

    monkeypatch.setattr("prime_cli.commands.images.APIClient", DummyAPIClient)
    monkeypatch.setattr("prime_cli.main.check_for_update", lambda: (False, None))


def test_build_vm_platform_image_sends_owner_scope(monkeypatch):
    captured = {}
    _patch_api(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "build-vm", "ubuntu:22.04", "--platform-image"],
        env=TEST_ENV,
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "method": "POST",
        "path": "/images/ubuntu/22.04/vm-build",
        "json": {"ownerScope": "platform"},
    }
    assert "(platform)" in result.output
    assert "prime images list --platform-image" in result.output


def test_build_vm_platform_image_preserves_namespaced_name(monkeypatch):
    captured = {}
    _patch_api(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "build-vm", "org/ubuntu:22.04", "--platform-image"],
        env={**TEST_ENV, "PRIME_TEAM_ID": "team-context"},
    )

    assert result.exit_code == 0, result.output
    assert captured["path"] == "/images/org/ubuntu/22.04/vm-build"
    assert captured["json"] == {"ownerScope": "platform"}


def test_build_vm_team_image_behavior_is_unchanged(monkeypatch):
    captured = {}
    _patch_api(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["images", "build-vm", "myapp:v1"],
        env={**TEST_ENV, "PRIME_TEAM_ID": "team-context"},
    )

    assert result.exit_code == 0, result.output
    assert captured["path"] == "/images/myapp/v1/vm-build"
    assert captured["json"] == {"teamId": "team-context"}
