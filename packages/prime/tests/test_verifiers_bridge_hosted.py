from types import SimpleNamespace

import pytest
import typer
from prime_cli import verifiers_bridge
from prime_cli.verifiers_bridge import ResolvedEnvironment


class _PluginStub:
    eval_module = "verifiers.cli.commands.eval"

    def build_module_command(self, module_name: str, args: list[str] | None = None) -> list[str]:
        return ["python", "-m", module_name, *(args or [])]


def test_resolve_environment_reference_preserves_version_in_display_id():
    resolved = verifiers_bridge._resolve_environment_reference(
        "primeintellect/wordle@2.0.0",
        "./environments",
    )

    assert resolved.install_slug == "primeintellect/wordle@2.0.0"
    assert resolved.env_display_id == "primeintellect/wordle@2.0.0"


def test_add_default_inference_and_key_args_requires_base_when_local():
    config = SimpleNamespace(api_key="test-key", inference_url="")

    with pytest.raises(typer.Exit):
        verifiers_bridge._add_default_inference_and_key_args(
            passthrough_args=[],
            config=config,
            require_base_url=True,
        )


def test_add_default_inference_and_key_args_allows_missing_base_for_hosted():
    config = SimpleNamespace(api_key="test-key", inference_url="")

    args, env, model, base = verifiers_bridge._add_default_inference_and_key_args(
        passthrough_args=["--hosted"],
        config=config,
        require_base_url=False,
    )

    assert "--hosted" in args
    assert "-k" in args
    assert "PRIME_API_KEY" in args
    assert env["PRIME_API_KEY"] == "test-key"
    assert model == verifiers_bridge.DEFAULT_MODEL
    assert base == ""


def test_run_eval_passthrough_hosted_skips_local_upload(monkeypatch):
    captured: dict[str, list[str] | dict[str, str] | bool] = {
        "command": [],
        "env": {},
        "uploaded": False,
    }

    monkeypatch.setattr(
        verifiers_bridge,
        "load_verifiers_prime_plugin",
        lambda console=None: _PluginStub(),
    )
    monkeypatch.setattr(
        verifiers_bridge,
        "Config",
        lambda: SimpleNamespace(api_key="test-key", inference_url="", team_id=None),
    )
    monkeypatch.setattr(
        verifiers_bridge,
        "_prepare_single_environment",
        lambda plugin, env_ref, env_dir: ResolvedEnvironment(
            original=env_ref,
            env_name="wordle",
            install_mode="remote",
            install_slug="primeintellect/wordle@2.0.0",
            upstream_slug="primeintellect/wordle",
            env_display_id="primeintellect/wordle@2.0.0",
            platform_slug="primeintellect/wordle",
            platform_url="https://app.primeintellect.ai/dashboard/environments/primeintellect/wordle",
        ),
    )

    def fake_run_command(command, env=None):
        captured["command"] = command
        captured["env"] = env or {}

    monkeypatch.setattr(verifiers_bridge, "_run_command", fake_run_command)

    def fake_push(**kwargs):
        captured["uploaded"] = True

    monkeypatch.setattr(verifiers_bridge, "push_eval_results_to_hub", fake_push)

    verifiers_bridge.run_eval_passthrough(
        environment="primeintellect/wordle@2.0.0",
        passthrough_args=["--hosted"],
        skip_upload=False,
        env_path=None,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "primeintellect/wordle@2.0.0" in command
    assert "--hosted" in command
    assert "X-Prime-Eval-Env-Display: primeintellect/wordle@2.0.0" in command
    assert "-s" not in command
    assert captured["uploaded"] is False
