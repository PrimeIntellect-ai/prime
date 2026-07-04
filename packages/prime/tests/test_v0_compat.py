"""The frozen v0 eval surface: prime parses the argv, converts via verifiers'
compat helpers, and runs the result on the v1 CLI (field mapping is tested in
verifiers itself)."""

import json

import pytest
import toml
import typer
from prime_cli.utils.verifiers_v0_cli import v0_argv_to_fields
from prime_cli.verifiers_bridge import ResolvedEnvironment, run_eval_passthrough

PLATFORM_TRANSITIONAL_TOML = """\
model = "openai/gpt-4.1-mini"
num_examples = 5
rollouts_per_example = 2
save_results = true

[[eval]]
env_args = { difficulty = "easy" }
max_retries = 4
headers = ["X-From-Platform: 1"]
taskset = { id = "gsm8k-v1" }
harness = { id = "default" }
group_size = 2
"""


class TestArgvParsing:
    def test_only_explicit_flags_are_returned(self):
        fields = v0_argv_to_fields(
            "reverse-text",
            ["--num-examples", "10", "--env-args", '{"difficulty": "hard"}', "-s"],
        )

        assert fields == {
            "env_target": "reverse-text",
            "num_examples": 10,
            "env_args": {"difficulty": "hard"},
            "save_results": True,
        }

    def test_unknown_flags_raise_system_exit(self):
        with pytest.raises(SystemExit):
            v0_argv_to_fields("reverse-text", ["--num-exmaples", "5"])


class TestEvalDispatch:
    """--save-results and transitional configs run through the conversion, not a v0 module."""

    @pytest.fixture
    def bridge_env(self, monkeypatch):
        class DummyConfig:
            api_key = "test-api-key"
            inference_url = "https://api.pinference.ai/api/v1"
            team_id = None

        class DummyPlugin:
            eval_module = "verifiers.v1.cli.eval.main"

            def build_module_command(self, module, args):
                return [module, *args]

        class DummyInferenceClient:
            def __init__(self, timeout=None):
                pass

            def retrieve_model(self, model):
                return {"id": model}

            def chat_completion(self, payload, stream=False):
                return {"id": "cmpl-123", "choices": []}

        commands = []
        monkeypatch.setattr("prime_cli.verifiers_bridge.Config", lambda: DummyConfig())
        monkeypatch.setattr(
            "prime_cli.verifiers_bridge.load_verifiers_prime_plugin", lambda: DummyPlugin()
        )
        monkeypatch.setattr("prime_cli.verifiers_bridge.InferenceClient", DummyInferenceClient)
        monkeypatch.setattr(
            "prime_cli.verifiers_bridge._prepare_v1_environment",
            lambda ref, env_dir_path: ResolvedEnvironment(
                original=ref, env_name=ref.split("/")[-1].split("@")[0], install_mode="local"
            ),
        )
        monkeypatch.setattr(
            "prime_cli.verifiers_bridge._run_command",
            lambda command, env=None: commands.append(command),
        )
        return commands

    def _converted_config_from(self, command):
        assert command[0] == "verifiers.v1.cli.eval.main"
        assert command[1] == "@"
        return toml.load(command[2])

    def test_save_results_argv_converts_to_v1_run(self, bridge_env):
        run_eval_passthrough(
            environment="reverse-text",
            passthrough_args=["--num-examples", "5", "--save-results", "--disable-tui"],
            skip_upload=True,
            env_path=None,
        )

        config = self._converted_config_from(bridge_env[0])
        assert config["taskset"] == {"id": "reverse-text"}
        assert config["num_tasks"] == 5
        assert config["rich"] is False
        assert "save_results" not in config

    def test_transitional_config_without_save_results_converts(self, bridge_env, tmp_path):
        config_path = tmp_path / "hosted-eval-config.toml"
        config_path.write_text(PLATFORM_TRANSITIONAL_TOML)

        # the platform's raw v1 invocation: config path + --skip-upload + TUI flag
        run_eval_passthrough(
            environment=str(config_path),
            passthrough_args=["--disable-tui"],
            skip_upload=True,
            env_path=None,
        )

        config = self._converted_config_from(bridge_env[0])
        assert config["taskset"] == {"id": "gsm8k-v1", "difficulty": "easy"}
        assert config["num_rollouts"] == 2
        assert config["rich"] is False
        # prime's run plumbing still applies to the converted config
        command = bridge_env[0]
        assert "--client.headers" in command
        headers = json.loads(command[command.index("--client.headers") + 1])
        assert headers["X-From-Platform"] == "1"
        assert "X-PI-Job-Id" in headers

    def test_debug_flag_is_accepted_as_tui_disable(self, bridge_env):
        run_eval_passthrough(
            environment="reverse-text",
            passthrough_args=["--save-results", "--debug"],
            skip_upload=True,
            env_path=None,
        )

        config = self._converted_config_from(bridge_env[0])
        assert config["rich"] is False

    def test_hub_slug_taskset_is_pinned_to_installed_name(self, bridge_env, tmp_path):
        config_path = tmp_path / "hosted-eval-config.toml"
        config_path.write_text(
            'model = "m"\n\n[[eval]]\ntaskset = { id = "prime/some-env@0.1.2" }\n'
        )

        run_eval_passthrough(
            environment=str(config_path),
            passthrough_args=[],
            skip_upload=True,
            env_path=None,
        )

        config = self._converted_config_from(bridge_env[0])
        assert config["taskset"] == {"id": "some-env"}

    def test_typo_flags_exit_with_usage_error(self, bridge_env):
        with pytest.raises(typer.Exit) as exc_info:
            run_eval_passthrough(
                environment="reverse-text",
                passthrough_args=["--save-results", "--num-exmaples", "5"],
                skip_upload=True,
                env_path=None,
            )
        assert exc_info.value.exit_code == 2


def test_prime_eval_default_routing_forms(monkeypatch):
    """`prime eval <env>` / `@ config` / `@config` route to the eval run (no `run` token)."""
    from prime_cli.main import app
    from typer.testing import CliRunner

    calls = []
    monkeypatch.setattr(
        "prime_cli.commands.evals.run_eval_passthrough",
        lambda **kwargs: calls.append(kwargs),
    )

    for argv, expected_environment, expected_args in [
        (["eval", "my-env", "-n", "5"], "my-env", ["-n", "5"]),
        (["eval", "@", "cfg.toml"], "cfg.toml", []),
        (["eval", "@cfg.toml"], "@cfg.toml", []),  # the bridge strips the @ prefix
        (["eval", "run", "my-env"], "my-env", []),  # hidden hosted-sandbox alias
    ]:
        result = CliRunner().invoke(app, argv, env={"PRIME_DISABLE_VERSION_CHECK": "1"})
        assert result.exit_code == 0, result.output
        call = calls.pop()
        assert call["environment"] == expected_environment
        assert call["passthrough_args"] == expected_args
