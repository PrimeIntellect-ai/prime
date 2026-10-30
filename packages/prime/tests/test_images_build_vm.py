from prime_cli.main import app
from typer.testing import CliRunner


def test_build_vm_command_is_removed():
    result = CliRunner().invoke(
        app, ["images", "build-vm", "app:v1"], env={"PRIME_DISABLE_VERSION_CHECK": "1"}
    )
    assert result.exit_code != 0
    assert "No such command 'build-vm'" in result.output
