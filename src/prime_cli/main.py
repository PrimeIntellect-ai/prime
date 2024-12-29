import typer
from .commands.availability import app as availability_app
from .commands.config import app as config_app

# Create the main CLI app
app = typer.Typer(name="prime", help="Prime Intellect CLI")


app.add_typer(availability_app, name="availability")
app.add_typer(config_app, name="config")


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Prime Intellect CLI"""
    if ctx.invoked_subcommand is None:
        ctx.get_help()


def run():
    """Entry point for the CLI"""
    app()
