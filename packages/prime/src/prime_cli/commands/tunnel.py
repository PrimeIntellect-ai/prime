import asyncio
import signal
from typing import List, Optional

import typer
from prime_tunnel import Tunnel
from prime_tunnel.core.client import TunnelClient
from prime_tunnel.core.config import Config
from rich.console import Console
from rich.table import Table

from prime_cli.utils.prompt import confirm_or_skip

app = typer.Typer(help="Manage tunnels for exposing local services", no_args_is_help=True)
console = Console()


@app.command("start")
def start_tunnel(
    port: int = typer.Option(8765, "--port", "-p", help="Local port to tunnel"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Friendly name for the tunnel"),
    team_id: Optional[str] = typer.Option(
        None, "--team-id", help="Team ID for team tunnels (uses config team_id if not specified)"
    ),
) -> None:
    """Start a tunnel to expose a local port."""

    async def run_tunnel():
        tunnel = Tunnel(local_port=port, name=name, team_id=team_id)

        shutdown_event = asyncio.Event()

        def signal_handler():
            console.print("\n[yellow]Shutting down tunnel...[/yellow]")
            shutdown_event.set()

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        try:
            url = await tunnel.start()
            console.print("\n[green]Tunnel started successfully![/green]")
            console.print(f"[bold]URL:[/bold] {url}")
            console.print(f"[bold]Tunnel ID:[/bold] {tunnel.tunnel_id}")
            console.print(f"\n[dim]Forwarding to localhost:{port}[/dim]")
            console.print("[dim]Press Ctrl+C to stop the tunnel[/dim]\n")

            await shutdown_event.wait()

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}", style="bold")
            raise typer.Exit(1)
        finally:
            await tunnel.stop()
            console.print("[green]Tunnel stopped[/green]")

    try:
        asyncio.run(run_tunnel())
    except KeyboardInterrupt:
        pass


@app.command("list")
def list_tunnels(
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID to list team tunnels (uses config team_id if not specified)",
    ),
) -> None:
    """List active tunnels."""

    async def fetch_tunnels():
        client = TunnelClient()
        try:
            tunnels = await client.list_tunnels(team_id=team_id)
            return tunnels
        finally:
            await client.close()

    try:
        tunnels = asyncio.run(fetch_tunnels())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)

    if not tunnels:
        console.print("[dim]No active tunnels[/dim]")
        return

    table = Table(title="Active Tunnels")
    table.add_column("Tunnel ID", style="cyan")
    table.add_column("URL", style="green")
    table.add_column("Expires At")

    for tunnel in tunnels:
        table.add_row(
            tunnel.tunnel_id,
            tunnel.url,
            str(tunnel.expires_at),
        )

    console.print(table)


@app.command("status")
def tunnel_status(
    tunnel_id: str = typer.Argument(..., help="Tunnel ID to check"),
) -> None:
    """Get status of a specific tunnel."""

    async def fetch_status():
        client = TunnelClient()
        try:
            return await client.get_tunnel(tunnel_id)
        finally:
            await client.close()

    try:
        tunnel = asyncio.run(fetch_status())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)

    if not tunnel:
        console.print(f"[red]Tunnel not found:[/red] {tunnel_id}")
        raise typer.Exit(1)

    console.print(f"[bold]Tunnel ID:[/bold] {tunnel.tunnel_id}")
    console.print(f"[bold]URL:[/bold] {tunnel.url}")
    console.print(f"[bold]Hostname:[/bold] {tunnel.hostname}")
    console.print(f"[bold]Expires At:[/bold] {tunnel.expires_at}")


@app.command("stop")
def stop_tunnel(
    tunnel_ids: Optional[List[str]] = typer.Argument(
        None, help="Tunnel ID(s) to stop (space or comma-separated)"
    ),
    all: bool = typer.Option(False, "--all", help="Stop all tunnels"),
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID to include team tunnels for --all (uses config team_id if not specified)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    only_mine: bool = typer.Option(
        True,
        "--only-mine/--all-users",
        help="Restrict '--all' deletes to only your tunnels (default: only yours)",
        show_default=True,
    ),
) -> None:
    """Stop and delete one or more tunnels.

    --only-mine controls whether '--all' will restrict to your tunnels or delete for all users.
    """

    if all and tunnel_ids:
        console.print("[red]Error:[/red] Cannot specify tunnel IDs with --all")
        raise typer.Exit(1)

    if not all and not tunnel_ids:
        console.print("[red]Error:[/red] Must specify at least one tunnel ID or --all")
        raise typer.Exit(1)

    parsed_ids: List[str] = []
    if all:

        async def fetch_tunnels() -> List[str]:
            client = TunnelClient()
            try:
                tunnels = await client.list_tunnels(team_id=team_id)
                if only_mine:
                    config = Config()
                    current_user_id = config.user_id
                    if not current_user_id:
                        console.print(
                            "[red]Error:[/red] Cannot filter by user - no user_id configured. "
                            "Use --all-users to delete all tunnels, or configure your user_id."
                        )
                        raise typer.Exit(1)
                    tunnels = [t for t in tunnels if t.user_id == current_user_id]
                return [t.tunnel_id for t in tunnels]
            finally:
                await client.close()

        try:
            parsed_ids = asyncio.run(fetch_tunnels())
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}", style="bold")
            raise typer.Exit(1)

        if not parsed_ids:
            console.print("[yellow]No active tunnels to stop[/yellow]")
            if only_mine:
                console.print(
                    "\n[dim]Note: --all only deletes your own tunnels by default. "
                    "Use --all-users to delete tunnels from all team members.[/dim]"
                )
            return
    else:
        raw_ids: List[str] = []
        for id_string in tunnel_ids or []:
            if "," in id_string:
                raw_ids.extend([id_.strip() for id_ in id_string.split(",") if id_.strip()])
            else:
                raw_ids.append(id_string.strip())

        seen = set()
        for tunnel_id in raw_ids:
            if tunnel_id and tunnel_id not in seen:
                parsed_ids.append(tunnel_id)
                seen.add(tunnel_id)

        if not parsed_ids:
            console.print("[red]Error:[/red] No valid tunnel IDs provided")
            raise typer.Exit(1)

    if all:
        confirmation_msg = (
            f"Are you sure you want to stop ALL {len(parsed_ids)} tunnel(s)? "
            "This action cannot be undone."
        )
        cancel_msg = "Stop all cancelled"
    elif len(parsed_ids) == 1:
        confirmation_msg = f"Are you sure you want to stop tunnel {parsed_ids[0]}?"
        cancel_msg = "Stop cancelled"
    else:
        confirmation_msg = f"Are you sure you want to stop {len(parsed_ids)} tunnel(s)?"
        cancel_msg = "Bulk stop cancelled"

    if not confirm_or_skip(confirmation_msg, yes):
        console.print(cancel_msg)
        return

    async def delete_tunnels() -> tuple[List[str], List[dict], List[dict]]:
        client = TunnelClient()
        succeeded: List[str] = []
        not_found: List[dict] = []
        failed: List[dict] = []
        try:
            result = await client.bulk_delete_tunnels(parsed_ids)
            succeeded = result.get("succeeded", [])
            for failure in result.get("failed", []):
                tid = failure.get("tunnel_id", "")
                error = failure.get("error", "Unknown error")
                if "not found" in error.lower():
                    not_found.append({"tunnel_id": tid, "error": error})
                else:
                    failed.append({"tunnel_id": tid, "error": error})
        except Exception as e:
            for tid in parsed_ids:
                failed.append({"tunnel_id": tid, "error": str(e)})
        finally:
            await client.close()
        return succeeded, not_found, failed

    try:
        with console.status("[bold blue]Stopping tunnel(s)...", spinner="dots"):
            succeeded, not_found, failed = asyncio.run(delete_tunnels())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)

    if len(parsed_ids) == 1 and succeeded and not all:
        console.print(f"[green]Tunnel deleted:[/green] {parsed_ids[0]}")
        return

    total_failed = len(not_found) + len(failed)
    total_processed = len(succeeded) + total_failed
    console.print(f"\n[green]Processed {total_processed} tunnel(s)[/green]")

    if succeeded:
        console.print(
            f"\n[bold green]Successfully deleted {len(succeeded)} tunnel(s):[/bold green]"
        )
        for tunnel_id in succeeded:
            console.print(f"  ✓ {tunnel_id}")

    if not_found:
        console.print(f"\n[bold yellow]Not found ({len(not_found)}):[/bold yellow]")
        for failure in not_found:
            console.print(f"  - {failure['tunnel_id']}")

    if failed:
        console.print(f"\n[bold red]Failed to delete {len(failed)} tunnel(s):[/bold red]")
        for failure in failed:
            console.print(f"  ✗ {failure['tunnel_id']}: {failure['error']}")

    if total_failed > 0:
        raise typer.Exit(1)

    if all:
        console.print("[green]All tunnels deleted successfully[/green]")
    else:
        console.print("[green]All specified tunnels deleted successfully[/green]")
