import asyncio
import signal
from typing import Optional

import typer
from prime_tunnel import Tunnel
from prime_tunnel.core.client import TunnelClient
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Manage tunnels for exposing local services")
console = Console()


@app.command("start")
def start_tunnel(
    port: int = typer.Option(8765, "--port", "-p", help="Local port to tunnel"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Friendly name for the tunnel"),
) -> None:
    """Start a tunnel to expose a local port."""

    async def run_tunnel():
        tunnel = Tunnel(local_port=port, name=name)

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
def list_tunnels() -> None:
    """List active tunnels."""

    async def fetch_tunnels():
        client = TunnelClient()
        try:
            tunnels = await client.list_tunnels()
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
    tunnel_id: str = typer.Argument(..., help="Tunnel ID to stop"),
) -> None:
    """Stop and delete a tunnel."""

    async def delete_tunnel():
        client = TunnelClient()
        try:
            return await client.delete_tunnel(tunnel_id)
        finally:
            await client.close()

    try:
        success = asyncio.run(delete_tunnel())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)

    if success:
        console.print(f"[green]Tunnel deleted:[/green] {tunnel_id}")
    else:
        console.print(f"[red]Tunnel not found:[/red] {tunnel_id}")
        raise typer.Exit(1)
