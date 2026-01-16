from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from prime_cli.core import APIClient, APIError

from ..api.ssh_keys import SSHKey, SSHKeysClient
from ..utils import confirm_or_skip, output_data_as_json, validate_output_format

app = typer.Typer(help="Manage SSH keys", no_args_is_help=True)
console = Console()


def _format_ssh_key_for_list(key: SSHKey) -> Dict[str, Any]:
    """Format SSH key data for list display"""
    return {
        "id": key.id,
        "name": key.name,
        "is_primary": key.is_primary,
        "is_user_key": key.is_user_key,
        "public_key_preview": (
            f"{key.public_key[:50]}..."
            if key.public_key and len(key.public_key) > 50
            else key.public_key
        ),
    }


@app.command()
def list(
    limit: int = typer.Option(100, help="Maximum number of SSH keys to list"),
    offset: int = typer.Option(0, help="Number of SSH keys to skip"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your SSH keys"""
    validate_output_format(output, console)

    try:
        base_client = APIClient()
        ssh_keys_client = SSHKeysClient(base_client)

        keys_list = ssh_keys_client.list(offset=offset, limit=limit)

        if output == "json":
            keys_data = []
            for key in keys_list.data:
                key_data = _format_ssh_key_for_list(key)
                keys_data.append(
                    {
                        "id": key_data["id"],
                        "name": key_data["name"],
                        "is_primary": key_data["is_primary"],
                        "is_user_key": key_data["is_user_key"],
                    }
                )

            output_data = {
                "ssh_keys": keys_data,
                "total_count": keys_list.total_count,
                "offset": offset,
                "limit": limit,
            }
            output_data_as_json(output_data, console)
        else:
            table = Table(
                title=f"SSH Keys (Total: {keys_list.total_count})",
                show_lines=True,
            )
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="blue")
            table.add_column("Primary", style="green")
            table.add_column("Type", style="yellow")

            for key in keys_list.data:
                key_data = _format_ssh_key_for_list(key)

                is_primary = key_data["is_primary"]
                primary_text = Text("Yes", style="green") if is_primary else Text("No", style="dim")
                key_type = "Custom" if key_data["is_user_key"] else "System"

                table.add_row(
                    key_data["id"],
                    key_data["name"] or "N/A",
                    primary_text,
                    key_type,
                )

            console.print(table)

            console.print(
                "\n[blue]Use the Key ID when creating pods via API "
                "with the sshKeyId parameter[/blue]"
            )

            if keys_list.total_count > offset + limit:
                remaining = keys_list.total_count - (offset + limit)
                console.print(
                    f"\n[yellow]Showing {limit} of {keys_list.total_count} SSH keys. "
                    f"Use --offset {offset + limit} to see the next "
                    f"{min(limit, remaining)} keys.[/yellow]"
                )

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def create(
    name: str = typer.Option(..., help="Name for the SSH key"),
    public_key: Optional[str] = typer.Option(None, help="The public key content"),
    public_key_file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Path to public key file (e.g., ~/.ssh/id_rsa.pub)"
    ),
) -> None:
    """Create a new SSH key"""
    try:
        # Get public key from file or argument
        key_content = public_key
        if public_key_file:
            import os

            expanded_path = os.path.expanduser(public_key_file)
            if not os.path.exists(expanded_path):
                console.print(f"[red]Error: File not found: {expanded_path}[/red]")
                raise typer.Exit(1)
            with open(expanded_path, "r") as f:
                key_content = f.read().strip()

        if not key_content:
            console.print("[red]Error: Must provide either --public-key or --file[/red]")
            raise typer.Exit(1)

        base_client = APIClient()
        ssh_keys_client = SSHKeysClient(base_client)

        with console.status("[bold blue]Creating SSH key...", spinner="dots"):
            key = ssh_keys_client.create(name=name, public_key=key_content)

        console.print("\n[green]Successfully created SSH key[/green]")
        console.print(f"ID: {key.id}")
        console.print(f"Name: {key.name}")
        console.print(f"\n[blue]Use this ID when creating pods via API: {key.id}[/blue]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def delete(
    key_id: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete an SSH key"""
    try:
        base_client = APIClient()
        ssh_keys_client = SSHKeysClient(base_client)

        if not confirm_or_skip(f"Are you sure you want to delete SSH key {key_id}?", yes):
            console.print("Deletion cancelled")
            raise typer.Exit(0)

        with console.status("[bold blue]Deleting SSH key...", spinner="dots"):
            ssh_keys_client.delete(key_id)

        console.print(f"[green]Successfully deleted SSH key {key_id}[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command(no_args_is_help=True)
def set_primary(
    key_id: str,
) -> None:
    """Set an SSH key as the primary key"""
    try:
        base_client = APIClient()
        ssh_keys_client = SSHKeysClient(base_client)

        with console.status("[bold blue]Setting primary SSH key...", spinner="dots"):
            ssh_keys_client.set_primary(key_id)

        console.print(f"[green]Successfully set SSH key {key_id} as primary[/green]")

    except APIError as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {str(e)}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
