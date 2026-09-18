from unittest.mock import Mock

import pytest
from prime_cli.commands import env
from typer.testing import CliRunner

INDEX_URL = "https://hub.primeintellect.ai/primeintellect/simple/"
WHEEL_URL = (
    "https://hub.primeintellect.ai/primeintellect/deep-swe/"
    "@f4017d3d/deep_swe-0.1.0-py3-none-any.whl"
)
URL_DEPENDENCY = "helper @ https://example.com/helper-1.0.0-py3-none-any.whl"


@pytest.fixture(autouse=True)
def workspace_python(monkeypatch):
    monkeypatch.setattr(env, "resolve_workspace_python", lambda: "/workspace/.venv/bin/python")


@pytest.mark.parametrize("installer", ["command", "automatic"])
@pytest.mark.parametrize("version", ["latest", "0.1.0", "f4017d3d"])
def test_install_uses_resolved_wheel_without_exposing_hub_dependency_index(
    monkeypatch, installer, version
):
    client = Mock()
    client.get.return_value = {
        "data": {
            "wheel_url": WHEEL_URL,
            "simple_index_url": INDEX_URL,
            "url_dependencies": [URL_DEPENDENCY],
            "visibility": "PUBLIC",
        }
    }
    monkeypatch.setattr(env, "APIClient", lambda **kwargs: client)
    monkeypatch.setattr(env, "load_verifiers_prime_plugin", lambda **kwargs: None)
    monkeypatch.setattr(env.shutil, "which", lambda tool: f"/bin/{tool}")
    execute = Mock()
    monkeypatch.setattr(env, "execute_install_command", execute)

    slug = f"primeintellect/deep-swe@{version}"
    if installer == "command":
        result = CliRunner().invoke(env.app, ["install", slug])
        assert result.exit_code == 0, result.output
    else:
        assert env._install_single_environment(slug)

    client.get.assert_called_once_with(f"/environmentshub/primeintellect/deep-swe/@{version}")
    execute.assert_called_once()
    command = execute.call_args.args[0]
    assert WHEEL_URL in command
    assert URL_DEPENDENCY in command
    assert "--extra-index-url" not in command
    assert INDEX_URL not in command
    assert "--index-strategy" not in command


@pytest.mark.parametrize("tool", ["uv", "pip"])
@pytest.mark.parametrize("no_upgrade", [False, True])
@pytest.mark.parametrize("prerelease", [False, True])
def test_wheel_install_preserves_options(tool, no_upgrade, prerelease):
    command = env._build_install_command(
        "deep-swe",
        "latest",
        INDEX_URL,
        WHEEL_URL,
        tool=tool,
        no_upgrade=no_upgrade,
        url_dependencies=[URL_DEPENDENCY],
        prerelease=prerelease,
    )

    assert command is not None
    assert WHEEL_URL in command
    assert URL_DEPENDENCY in command
    assert INDEX_URL not in command
    upgrade_flag = "-P" if tool == "uv" else "--upgrade"
    prerelease_flag = "--prerelease=allow" if tool == "uv" else "--pre"
    assert (upgrade_flag in command) is not no_upgrade
    assert (prerelease_flag in command) is prerelease
    if tool == "uv":
        assert command[:5] == ["uv", "pip", "install", "--python", "/workspace/.venv/bin/python"]
        if not no_upgrade:
            assert command[command.index("-P") + 1] == "deep_swe"
    else:
        assert command[:2] == ["pip", "install"]


@pytest.mark.parametrize("tool", ["uv", "pip"])
@pytest.mark.parametrize("version", ["latest", "0.1.0"])
def test_install_falls_back_to_index_when_no_wheel_is_available(tool, version):
    command = env._build_install_command(
        "deep-swe", version, INDEX_URL, None, tool=tool, url_dependencies=[URL_DEPENDENCY]
    )

    assert command is not None
    requirement = "deep_swe" if version == "latest" else f"deep_swe=={version}"
    assert requirement in command
    assert command[command.index("--extra-index-url") + 1] == INDEX_URL
    assert URL_DEPENDENCY in command
    if tool == "uv":
        assert command[command.index("--exclude-newer-package") + 1] == "deep_swe=false"


@pytest.mark.parametrize("tool", ["uv", "pip"])
def test_install_supports_wheel_without_index(tool):
    command = env._build_install_command("deep-swe", "latest", None, WHEEL_URL, tool=tool)

    assert command is not None
    assert WHEEL_URL in command
    assert "--extra-index-url" not in command


def test_install_requires_wheel_or_index():
    assert env._build_install_command("deep-swe", "latest", None, None) is None
