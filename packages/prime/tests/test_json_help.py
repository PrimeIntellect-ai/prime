from prime_cli.main import app
from prime_cli.utils import strip_ansi
from typer.testing import CliRunner

runner = CliRunner()


def _clean(output: str) -> str:
    return strip_ansi(output)


def _squash(output: str) -> str:
    return " ".join(_clean(output).split())


def test_json_schema_is_shown_for_output_help() -> None:
    result = runner.invoke(app, ["sandbox", "list", "--help"])

    assert result.exit_code == 0, result.output
    output = _squash(result.output)
    assert "JSON output (--output json)" in output
    assert ".sandboxes[] = {id, name, image, status, resources, labels[], created_at}" in output
    assert ".has_next = boolean" in output


def test_json_schema_is_shown_for_plain_help() -> None:
    result = runner.invoke(app, ["env", "secret", "link", "--plain", "--help"])

    assert result.exit_code == 0, result.output
    output = _squash(result.output)
    assert "JSON output (--output json)" in output
    assert ". = {id, secretId, secretName, environmentId, createdAt}" in output


def test_json_schema_is_shown_for_json_only_help() -> None:
    result = runner.invoke(app, ["rl", "progress", "--help"])

    assert result.exit_code == 0, result.output
    output = _squash(result.output)
    assert "JSON output" in output
    assert ". = {latest_step, steps_with_samples[]" in output
    assert "steps_with_distributions[]" in output
    assert "last_updated_at}" in output


def test_json_schema_is_shown_for_eval_push_help() -> None:
    result = runner.invoke(app, ["eval", "push", "--help"])

    assert result.exit_code == 0, result.output
    output = _squash(result.output)
    assert "JSON output (--output json)" in output
    assert "Single push: .evaluation_id = string" in output
    assert "Auto-" in output
    assert "discovery batch push: .results[] = {path, status, eval_id?, error?}" in output
