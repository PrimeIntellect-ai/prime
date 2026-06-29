import pytest
from click.testing import CliRunner
from prime_cli.main import app
from prime_cli.verifiers_bridge import exec_verifiers_process

runner = CliRunner()


class ExecCalled(Exception):
    pass


def test_eval_run_help_flags_are_forwarded(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "prime_cli.command_router.exec_verifiers_process",
        lambda name, args, plain: captured.append((name, list(args), plain)),
    )

    for flag in ("-h", "--help"):
        result = runner.invoke(
            app,
            ["eval", "run", flag],
            env={"PRIME_DISABLE_VERSION_CHECK": "1"},
        )
        assert result.exit_code == 0, result.output

    assert captured == [
        ("eval", ["-h"], False),
        ("eval", ["--help"], False),
    ]


def test_lab_setup_help_flags_use_prime_owned_help():
    for flag in ("-h", "--help"):
        result = runner.invoke(
            app,
            ["lab", "setup", flag],
            env={"PRIME_DISABLE_VERSION_CHECK": "1"},
        )
        assert result.exit_code == 0, result.output
        assert "Set up a Lab workspace." in result.output
        assert "--skip-install" in result.output
        assert "--prime-rl" not in result.output


def test_exec_verifiers_process_forwards_once(monkeypatch):
    captured = {}

    def fake_exec(executable, command, env):
        captured.update(executable=executable, command=command, env=env)
        raise ExecCalled

    monkeypatch.setattr(
        "prime_cli.verifiers_bridge.build_verifiers_command",
        lambda name, args: ["python", "-m", f"verifiers.{name}", *args],
    )
    monkeypatch.setattr("prime_cli.verifiers_bridge.os.execvpe", fake_exec)

    with pytest.raises(ExecCalled):
        exec_verifiers_process("eval", ["gsm8k-v1", "--dry-run"], plain=True)

    assert captured["executable"] == "python"
    assert captured["command"] == [
        "python",
        "-m",
        "verifiers.eval",
        "gsm8k-v1",
        "--dry-run",
    ]
    assert captured["env"]["NO_COLOR"] == "1"
    assert captured["env"]["PYDANTIC_CONFIG_PLAIN"] == "1"


@pytest.mark.parametrize(
    ("subcommand", "args"),
    [
        ("validate", ["gsm8k-v1", "--runtime.type", "subprocess"]),
        ("serve", ["--id", "legacy-env", "--dry-run"]),
    ],
)
def test_env_verifiers_commands_forward_argv(monkeypatch, subcommand, args):
    captured = {}
    monkeypatch.setattr(
        "prime_cli.command_router.exec_verifiers_process",
        lambda name, forwarded, plain: captured.update(
            name=name, args=list(forwarded), plain=plain
        ),
    )

    result = runner.invoke(
        app,
        ["env", subcommand, *args],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {"name": subcommand, "args": args, "plain": False}


def test_env_init_runs_hygiene_then_delegates_to_verifiers(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "prime_cli.command_router.exec_verifiers_process",
        lambda name, forwarded, plain: captured.update(
            name=name, args=list(forwarded), plain=plain
        ),
    )
    monkeypatch.setattr(
        "prime_cli.commands.env._run_env_init_lab_hygiene_preflight",
        lambda: captured.update(hygiene=True),
    )

    result = runner.invoke(
        app,
        ["env", "init", "demo-v1", "--add-tool", "--force"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "hygiene": True,
        "name": "init",
        "args": ["demo-v1", "--add-tool", "--force"],
        "plain": False,
    }


def test_eval_view_uses_prime_viewer(monkeypatch):
    calls = {}

    def fake_run_eval_view(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("prime_lab_app.run_eval_view", fake_run_eval_view)

    result = runner.invoke(
        app,
        ["eval", "view", "--limit", "25", "--env-dir", "envs", "--outputs-dir", "outs"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 0, result.output
    assert calls["limit"] == 25
    assert calls["env_dir"] == "envs"
    assert calls["outputs_dir"] == "outs"


def test_eval_view_rejects_non_positive_limit():
    result = runner.invoke(
        app,
        ["eval", "view", "--limit", "0"],
        env={"PRIME_DISABLE_VERSION_CHECK": "1"},
    )

    assert result.exit_code == 1
    assert "--limit must be at least 1" in result.output
