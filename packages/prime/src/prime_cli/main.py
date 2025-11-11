from importlib.metadata import version

import typer

from .commands.availability import app as availability_app
from .commands.config import app as config_app
from .commands.env import app as env_app
from .commands.evals import app as evals_app
from .commands.inference import app as inference_app
from .commands.login import app as login_app
from .commands.pods import app as pods_app
from .commands.registry import app as registry_app
from .commands.sandbox import app as sandbox_app
from .commands.teams import app as teams_app
from .commands.whoami import app as whoami_app

__version__ = version("prime")

app = typer.Typer(
    name="prime",
    help=f"Prime Intellect CLI (v{__version__})",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(availability_app, name="availability")
app.add_typer(config_app, name="config")
app.add_typer(pods_app, name="pods")
app.add_typer(sandbox_app, name="sandbox")
app.add_typer(registry_app, name="registry")
app.add_typer(login_app, name="login")
app.add_typer(env_app, name="env")
app.add_typer(inference_app, name="inference")
app.add_typer(whoami_app, name="whoami")
app.add_typer(teams_app, name="teams")
app.add_typer(evals_app, name="eval")


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
) -> None:
    """Prime Intellect CLI"""
    if version_flag:
        typer.echo(f"Prime CLI version: {__version__}")
        raise typer.Exit()


def run() -> None:
    """Entry point for the CLI"""
    app()
