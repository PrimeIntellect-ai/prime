import io

from prime_cli.utils.display import output_data_as_json
from prime_cli.utils.plain import get_console


def test_plain_json_output_preserves_literal_strings(monkeypatch) -> None:
    monkeypatch.setattr("prime_cli.utils.plain.is_plain_mode", lambda args=None: True)

    buffer = io.StringIO()
    console = get_console(file=buffer)

    output_data_as_json(
        {"prompt": "[INST]", "label": "[red]literal[/red]", "close": "[/]"},
        console,
    )

    assert (
        buffer.getvalue()
        == '{\n  "prompt": "[INST]",\n  "label": "[red]literal[/red]",\n  "close": "[/]"\n}\n'
    )
