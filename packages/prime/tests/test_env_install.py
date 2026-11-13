from unittest.mock import Mock

import pytest
from prime_cli.commands import env
from prime_cli.utils.unmet_requirements import unmet_requirements
from typer.testing import CliRunner

INDEX_URL = "https://hub.primeintellect.ai/primeintellect/simple/"
SCOPED_INDEX_URL = "https://hub.primeintellect.ai/primeintellect/deep-swe/install/simple/"
WHEEL_URL = (
    "https://hub.primeintellect.ai/primeintellect/deep-swe/"
    "@f4017d3d/deep_swe-0.1.0-py3-none-any.whl"
)
URL_DEPENDENCY = "helper @ https://example.com/helper-1.0.0-py3-none-any.whl"


@pytest.fixture(autouse=True)
def workspace_python(monkeypatch):
    monkeypatch.setattr(env, "resolve_workspace_python", lambda: "/workspace/.venv/bin/python")


@pytest.mark.parametrize("installer", ["command", "automatic"])
@pytest.mark.parametrize("version", ["latest", "0.1.0"])
@pytest.mark.parametrize("scoped_index", [SCOPED_INDEX_URL, None])
def test_install_prefers_scoped_index_with_legacy_server_fallback(
    monkeypatch, installer, version, scoped_index
):
    client = Mock()
    client.get.return_value = {
        "data": {
            "wheel_url": WHEEL_URL,
            "simple_index_url": INDEX_URL,
            "install_index_url": scoped_index,
            "url_dependencies": [URL_DEPENDENCY],
            "visibility": "PUBLIC",
        }
    }
    monkeypatch.setattr(env, "APIClient", lambda **kwargs: client)
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
    assert WHEEL_URL not in command
    assert URL_DEPENDENCY in command
    assert command[command.index("--extra-index-url") + 1] == (scoped_index or INDEX_URL)
    assert ("deep_swe" if version == "latest" else f"deep_swe=={version}") in command
    assert "--index-strategy" not in command


@pytest.mark.parametrize("tool", ["uv", "pip"])
@pytest.mark.parametrize("no_upgrade", [False, True])
@pytest.mark.parametrize("prerelease", [False, True])
def test_scoped_index_install_preserves_options(tool, no_upgrade, prerelease):
    command = env._build_install_command(
        "deep-swe",
        "latest",
        SCOPED_INDEX_URL,
        WHEEL_URL,
        tool=tool,
        no_upgrade=no_upgrade,
        url_dependencies=[URL_DEPENDENCY],
        prerelease=prerelease,
    )

    assert command is not None
    assert SCOPED_INDEX_URL in command
    assert WHEEL_URL not in command
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
def test_legacy_index_without_wheel_still_works(tool, version):
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


def test_info_displays_scoped_install_index(monkeypatch):
    client = Mock()
    client.get.return_value = {
        "data": {
            "id": "env-1",
            "name": "deep-swe",
            "owner": {"name": "primeintellect"},
            "visibility": "PUBLIC",
            "wheel_url": WHEEL_URL,
            "simple_index_url": INDEX_URL,
            "install_index_url": SCOPED_INDEX_URL,
        }
    }
    monkeypatch.setattr(env, "APIClient", lambda **kwargs: client)
    result = CliRunner().invoke(env.app, ["info", "primeintellect/deep-swe"])
    assert result.exit_code == 0, result.output
    # Rich may wrap URLs, so inspect rendered output with whitespace removed.
    output = "".join(result.output.split())
    assert SCOPED_INDEX_URL in output
    assert INDEX_URL not in output


def _hub_client(monkeypatch):
    client = Mock()
    client.get.return_value = {
        "data": {"simple_index_url": INDEX_URL, "url_dependencies": [], "visibility": "PUBLIC"}
    }
    monkeypatch.setattr(env, "APIClient", lambda **kwargs: client)
    monkeypatch.setattr(env.shutil, "which", lambda tool: f"/bin/{tool}")
    return client


@pytest.mark.parametrize(
    ("slug", "installed", "installs"),
    [
        ("primeintellect/deep-swe", True, False),
        ("primeintellect/deep-swe", False, True),
        ("primeintellect/deep-swe@0.1.0", True, True),
    ],
)
def test_target_keeps_an_installed_environment_unless_pinned(
    monkeypatch, tmp_path, slug, installed, installs
):
    _hub_client(monkeypatch)
    monkeypatch.setattr(env, "_workspace_has_package", lambda package: installed)
    execute, fill = Mock(), Mock()
    monkeypatch.setattr(env, "execute_install_command", execute)
    monkeypatch.setattr(env, "_install_missing_requirements", fill)

    result = CliRunner().invoke(env.app, ["install", slug, "--target", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert execute.called is installs
    if installs:
        command = execute.call_args.args[0]
        assert command[-3:] == ["--target", str(tmp_path), "--no-deps"]
        fill.assert_called_once_with(tmp_path)


def test_a_failed_install_exits_non_zero(monkeypatch):
    _hub_client(monkeypatch)
    monkeypatch.setattr(env, "execute_install_command", Mock(side_effect=Exception("boom")))

    result = CliRunner().invoke(env.app, ["install", "primeintellect/deep-swe"])

    assert result.exit_code == 1, result.output


def _distribution(target, name, *requires):
    dist = target / f"{name}-1.0.dist-info"
    dist.mkdir()
    lines = "".join(f"Requires-Dist: {line}\n" for line in requires)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n{lines}")


def test_unmet_requirements_are_the_ones_the_interpreter_lacks(monkeypatch, tmp_path):
    _distribution(
        tmp_path,
        "probe",
        "pytest>=1",
        "absent-package==1.0",
        'other-absent; extra == "dev"',
        "helper[fast]",
    )
    _distribution(
        tmp_path, "helper", 'fast-absent; extra == "fast"', 'slow-absent; extra == "slow"'
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert unmet_requirements(str(tmp_path)) == ["absent-package==1.0", "fast-absent"]


def test_missing_requirements_install_layer_by_layer_without_dependencies(monkeypatch, tmp_path):
    marked = 'absent>=1; python_version >= "3.10"'
    monkeypatch.setattr(env, "_unmet_requirements", Mock(side_effect=[[marked], ["leaf"], []]))
    execute = Mock()
    monkeypatch.setattr(env, "execute_install_command", execute)

    env._install_missing_requirements(tmp_path)

    assert [call.args[0][-1] for call in execute.call_args_list] == [marked, "leaf"]
    assert all("--no-deps" in call.args[0] for call in execute.call_args_list)


def test_a_requirement_still_unmet_after_installing_it_is_a_conflict(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "_unmet_requirements", Mock(return_value=["numpy<2"]))
    monkeypatch.setattr(env, "execute_install_command", Mock())

    with pytest.raises(RuntimeError, match="numpy<2"):
        env._install_missing_requirements(tmp_path)


def test_target_fails_on_an_invalid_id(monkeypatch, tmp_path):
    _hub_client(monkeypatch)

    result = CliRunner().invoke(env.app, ["install", "owner/", "--target", str(tmp_path)])

    assert result.exit_code == 1, result.output
