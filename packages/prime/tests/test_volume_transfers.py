"""`prime volumes import/export/transfers`: removed, but they say where to go."""

import pytest
from prime_cli.commands import volumes
from prime_cli.main import app
from typer.testing import CliRunner


@pytest.mark.parametrize(
    "args",
    [
        ["import", "data", "s3://bucket/prefix", "in/"],
        ["export", "data", "out/", "s3://bucket/prefix", "--region", "eu-west-1"],
        ["transfers", "list", "data", "--id", "t1", "--follow"],
        ["transfers", "cancel", "data", "t1"],
        ["import", "--help"],
    ],
)
def test_s3_transfer_commands_point_at_put_and_get(monkeypatch, args):
    monkeypatch.setattr(volumes, "_client", lambda: pytest.fail("no API call expected"))

    result = CliRunner().invoke(app, ["volumes", *args])

    text = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "S3 import/export was removed" in text
    assert "prime volumes put" in text and "prime volumes get" in text
