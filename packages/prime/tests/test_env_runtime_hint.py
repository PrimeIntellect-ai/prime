import pytest
from prime_cli.commands.env import app
from prime_cli.utils.environment_runtime import (
    VERIFIERS_V0,
    VERIFIERS_V1,
    classify_runtime_from_metadata,
    parse_runtime_option,
)
from typer.testing import CliRunner


@pytest.mark.parametrize(
    ("requirements", "expected"),
    [
        (["verifiers>=0.2.0"], VERIFIERS_V1),
        (["verifiers[harbor]>=0.3.1"], VERIFIERS_V1),
        (["verifiers>=0.1.2"], VERIFIERS_V0),
        (["verifiers"], VERIFIERS_V0),
        (["httpx>=0.25"], None),
        ([], None),
        (["verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git"], None),
        # Unconditional entries beat marker-guarded ones.
        (['verifiers>=0.2.0; python_version >= "3.11"', "verifiers>=0.1.0"], VERIFIERS_V0),
        # Repeated entries intersect, so the highest floor wins.
        (["verifiers>=0.1", "verifiers>=0.2"], VERIFIERS_V1),
        (["verifiers==0.2.*"], VERIFIERS_V1),
        (["not a requirement", "verifiers~=0.3.0"], VERIFIERS_V1),
    ],
)
def test_classify_from_metadata(requirements, expected):
    assert classify_runtime_from_metadata(requirements) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("v1", VERIFIERS_V1),
        ("V0", VERIFIERS_V0),
        (" v1 ", VERIFIERS_V1),
    ],
)
def test_parse_runtime_option(value, expected):
    assert parse_runtime_option(value) == expected


@pytest.mark.parametrize("value", ["v2", "verifiers_v1", "legacy", ""])
def test_parse_runtime_option_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="--runtime must be 'v0' or 'v1'"):
        parse_runtime_option(value)


def test_push_rejects_invalid_runtime_without_unexpected_error():
    result = CliRunner().invoke(app, ["push", "--runtime", "v2"])

    assert result.exit_code == 1
    assert "--runtime must be 'v0' or 'v1'" in result.output
    assert "Unexpected error" not in result.output
