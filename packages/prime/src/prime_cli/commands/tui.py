from typing import Optional

import typer

app = typer.Typer(help="Launch TUI for viewing eval results")


@app.callback(invoke_without_command=True)
def tui(
    env_dir: Optional[str] = typer.Option(
        None, "--env-dir", "-e", help="Path to environments directory"
    ),
    outputs_dir: Optional[str] = typer.Option(
        None, "--outputs-dir", "-o", help="Path to outputs directory"
    ),
) -> None:
    """Launch TUI for viewing eval results (passthrough to vf-tui)."""
    from verifiers.scripts.tui import VerifiersTUI

    env_path = env_dir or "./environments"
    outputs_path = outputs_dir or "./outputs"
    tui_app = VerifiersTUI(env_dir_path=env_path, outputs_dir_path=outputs_path)
    tui_app.run()
