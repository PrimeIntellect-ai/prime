import asyncio
import signal
from typing import List, Optional

import typer
from rich.table import Table

from prime_cli.utils import (
    PlainTyper,
    get_console,
    human_age,
    iso_timestamp,
    output_data_as_json,
    validate_output_format,
)
from prime_cli.utils.prompt import confirm_or_skip

app = PlainTyper(help="Manage tunnels for exposing local services", no_args_is_help=True)
console = get_console()


def _format_tunnel_for_output(tunnel) -> dict:
    created_at = tunnel.created_at
    return {
        "tunnel_id": tunnel.tunnel_id,
        "name": tunnel.name,
        "url": tunnel.url,
        "hostname": tunnel.hostname,
        "status": tunnel.status or "UNKNOWN",
        "labels": tunnel.labels,
        "local_port": tunnel.local_port,
        "user_id": tunnel.user_id,
        "team_id": tunnel.team_id,
        "created_at": iso_timestamp(created_at) if created_at else None,
        "created": human_age(created_at) if created_at else None,
        "expires_at": iso_timestamp(tunnel.expires_at),
    }


@app.command("start")
def start_tunnel(
    port: int = typer.Option(8765, "--port", "-p", help="Local port to tunnel"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Friendly name for the tunnel"),
    labels: Optional[List[str]] = typer.Option(
        None,
        "--label",
        "-l",
        help="Labels/tags for the tunnel. Can be specified multiple times.",
    ),
    team_id: Optional[str] = typer.Option(
        None, "--team-id", help="Team ID for team tunnels (uses config team_id if not specified)"
    ),
) -> None:
    """Start a tunnel to expose a local port."""

    async def run_tunnel():
        from prime_tunnel import Tunnel
        from prime_tunnel.exceptions import (
            TunnelConnectionError,
            TunnelLimitReachedError,
            TunnelTimeoutError,
        )

        tunnel = Tunnel(local_port=port, name=name, team_id=team_id, labels=labels)

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

            # Monitor tunnel health while waiting for shutdown signal
            while not shutdown_event.is_set():
                if not tunnel.is_running:
                    output = "\n".join(tunnel.recent_output) or "(no output captured)"
                    raise TunnelConnectionError(
                        message=(
                            f"Tunnel process exited unexpectedly\n--- frpc output ---\n{output}"
                        ),
                        tunnel_id=tunnel.tunnel_id,
                    )
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass

        except TunnelConnectionError as e:
            console.print(f"\n[red]Tunnel error:[/red] {e}", style="bold")
            if e.tunnel_id:
                console.print(f"[dim]Tunnel ID: {e.tunnel_id}[/dim]")
            raise typer.Exit(1)
        except TunnelLimitReachedError as e:
            console.print(f"\n[red]Tunnel limit reached:[/red] {e}", style="bold")
            console.print("[dim]Delete an existing tunnel before creating a new one.[/dim]")
            raise typer.Exit(1)
        except TunnelTimeoutError as e:
            console.print(f"\n[red]Connection timed out:[/red] {e}", style="bold")
            raise typer.Exit(1)
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
    labels: Optional[List[str]] = typer.Option(
        None,
        "--label",
        "-l",
        help="Filter by labels. Can be specified multiple times.",
    ),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    sort_by: str = typer.Option(
        "createdAt",
        "--sort-by",
        help="Sort field: createdAt, status, name, expiresAt, connectedAt",
    ),
    sort_order: str = typer.Option(
        "desc",
        "--sort-order",
        help="Sort order: asc or desc",
    ),
    page: int = typer.Option(1, "--page", "-p", help="Page number"),
    num: int = typer.Option(50, "--num", "-n", help="Items per page"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List active tunnels."""
    validate_output_format(output, console)

    async def fetch_tunnels():
        from prime_tunnel.core.client import TunnelClient

        client = TunnelClient()
        try:
            return await client.list_tunnels_page(
                team_id=team_id,
                labels=labels,
                status=status,
                page=page,
                per_page=num,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        finally:
            await client.close()

    try:
        page_result = asyncio.run(fetch_tunnels())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise typer.Exit(1)

    tunnels = page_result.tunnels
    total = page_result.total
    has_next = page_result.has_next

    tunnels_data = [_format_tunnel_for_output(tunnel) for tunnel in tunnels]
    if output == "json":
        output_data_as_json(
            {
                "tunnels": tunnels_data,
                "total": total,
                "page": page,
                "per_page": num,
                "has_next": has_next,
            },
            console,
        )
        return

    if not tunnels:
        console.print("[dim]No active tunnels[/dim]")
        return

    table = Table(title=f"Active Tunnels (Total: {total})")
    table.add_column("Tunnel ID", style="cyan")
    table.add_column("Name", style="blue")
    table.add_column("User ID", style="magenta")
    table.add_column("URL", style="green")
    table.add_column("Status")
    table.add_column("Labels")
    table.add_column("Created")

    for tunnel_data in tunnels_data:
        status_display = tunnel_data["status"]
        normalized_status = str(status_display).upper()
        if normalized_status == "CONNECTED":
            status_display = "[green]connected[/green]"
        elif normalized_status == "PENDING":
            status_display = "[yellow]pending[/yellow]"
        elif normalized_status == "DISCONNECTED":
            status_display = "[red]disconnected[/red]"
        elif normalized_status == "EXPIRED":
            status_display = "[dim]expired[/dim]"

        table.add_row(
            tunnel_data["tunnel_id"],
            tunnel_data["name"] or "",
            tunnel_data["user_id"] or "",
            tunnel_data["url"],
            status_display,
            ", ".join(tunnel_data["labels"]) if tunnel_data["labels"] else "-",
            tunnel_data["created"] or "-",
        )

    console.print(table)

    start = (page - 1) * num + 1
    end = start + len(tunnels) - 1
    console.print(f"\n[dim]Page {page} • showing {start}-{end} of {total} tunnel(s)[/dim]")
    if has_next:
        console.print(f"[dim]Use --page {page + 1} to see more.[/dim]")


@app.command("status")
def tunnel_status(
    tunnel_id: str = typer.Argument(..., help="Tunnel ID to check"),
) -> None:
    """Get status of a specific tunnel."""

    async def fetch_status():
        from prime_tunnel.core.client import TunnelClient

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
    console.print(f"[bold]Status:[/bold] {tunnel.status or 'unknown'}")
    if tunnel.labels:
        console.print(f"[bold]Labels:[/bold] {', '.join(tunnel.labels)}")
    console.print(f"[bold]Expires At:[/bold] {tunnel.expires_at}")


@app.command("stop")
def stop_tunnel(
    tunnel_ids: Optional[List[str]] = typer.Argument(
        None, help="Tunnel ID(s) to stop (space or comma-separated)"
    ),
    all: bool = typer.Option(False, "--all", "-a", help="Stop all tunnels"),
    labels: Optional[List[str]] = typer.Option(
        None,
        "--label",
        "-l",
        help="Stop tunnels matching labels. Can be specified multiple times.",
    ),
    team_id: Optional[str] = typer.Option(
        None,
        "--team-id",
        help="Team ID to include team tunnels for --all (uses config team_id if not specified)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    only_mine: bool = typer.Option(
        True,
        "--only-mine/--all-users",
        "-m/-A",
        help="Restrict '--all' deletes to only your tunnels",
        show_default=True,
    ),
) -> None:
    """Stop and delete one or more tunnels.

    --only-mine controls whether '--all' will restrict to your tunnels or delete for all users.
    """

    if sum(bool(flag) for flag in (all, tunnel_ids, labels)) > 1:
        console.print("[red]Error:[/red] Use only one of tunnel IDs, --all, or --label")
        raise typer.Exit(1)

    if not all and not tunnel_ids and not labels:
        console.print("[red]Error:[/red] Must specify tunnel IDs, --all, or --label")
        raise typer.Exit(1)

    parsed_ids: List[str] = []
    if tunnel_ids:
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

        async def fetch_tunnel_ids() -> List[str]:
            from prime_tunnel.core.client import TunnelClient

            client = TunnelClient()
            try:
                scoped_team_id = team_id
                if scoped_team_id is None:
                    scoped_team_id = client.config.team_id
                scoped_user_id = client.config.user_id if only_mine else None
                if only_mine and not scoped_user_id:
                    raise ValueError(
                        "Cannot resolve current user ID for scoped bulk delete. "
                        "Run `prime login`, set PRIME_USER_ID, or delete explicit tunnel IDs."
                    )
                if not only_mine and not scoped_team_id:
                    raise ValueError("all_users requires a team ID")

                # TODO: remove this migration compatibility path after
                # old tunnel registrations have aged out past their
                # 7-day TTL.
                # When removing this block, restore the commented --all branch
                # in delete_tunnels().
                ids: List[str] = []
                seen_ids: set[str] = set()
                page = 1
                max_pages = 1000
                while page <= max_pages:
                    result = await client.list_tunnels_page(
                        team_id=scoped_team_id,
                        page=page,
                        per_page=1000,
                    )
                    for tunnel in result.tunnels:
                        if only_mine and tunnel.user_id != scoped_user_id:
                            continue
                        if tunnel.tunnel_id not in seen_ids:
                            ids.append(tunnel.tunnel_id)
                            seen_ids.add(tunnel.tunnel_id)

                    if not result.has_next or not result.tunnels:
                        break
                    page += 1

                return ids
            finally:
                await client.close()

        try:
            parsed_ids = asyncio.run(fetch_tunnel_ids())
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

    if labels:

        async def validate_label_scope() -> None:
            from prime_tunnel.core.client import TunnelClient

            client = TunnelClient()
            try:
                scoped_user_id = client.config.user_id if only_mine else None
                scoped_team_id = team_id if team_id is not None else client.config.team_id
                if only_mine and not scoped_user_id:
                    raise ValueError(
                        "Cannot resolve current user ID for scoped bulk delete. "
                        "Run `prime login`, set PRIME_USER_ID, or delete explicit tunnel IDs."
                    )
                if not only_mine and not scoped_team_id:
                    raise ValueError("all_users requires a team ID")
            finally:
                await client.close()

        try:
            asyncio.run(validate_label_scope())
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}", style="bold")
            raise typer.Exit(1)

        confirmation_msg = (
            f"Are you sure you want to stop tunnels matching label(s): {', '.join(labels)}? "
            "This action cannot be undone."
        )
        cancel_msg = "Stop by label cancelled"
    elif all:
        confirmation_msg = (
            "Are you sure you want to stop all matching tunnel(s)? This action cannot be undone."
        )
        if team_id and only_mine:
            confirmation_msg = (
                f"Are you sure you want to stop all of your tunnels in team {team_id}? "
                "This action cannot be undone."
            )
        elif team_id and not only_mine:
            confirmation_msg = (
                f"Are you sure you want to stop all tunnels in team {team_id}? "
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
        from prime_tunnel.core.client import TunnelClient

        client = TunnelClient()
        succeeded: List[str] = []
        not_found: List[dict] = []
        failed: List[dict] = []
        try:
            scoped_team_id = team_id
            if labels and scoped_team_id is None:
                scoped_team_id = client.config.team_id
            scoped_user_id = client.config.user_id if only_mine else None
            if labels and only_mine and not scoped_user_id:
                raise ValueError(
                    "Cannot resolve current user ID for scoped bulk delete. "
                    "Run `prime login`, set PRIME_USER_ID, or delete explicit tunnel IDs."
                )
            if labels:
                result = await client.bulk_delete_tunnels(
                    labels=labels,
                    team_id=scoped_team_id,
                    user_id=scoped_user_id,
                    all_users=not only_mine,
                )
            # TODO: uncomment this branch after Redis-backed tunnel
            # registrations have aged out and the --all pre-listing fallback
            # above is removed.
            # elif all:
            #     if scoped_team_id is None:
            #         scoped_team_id = client.config.team_id
            #     result = await client.bulk_delete_tunnels(
            #         team_id=scoped_team_id,
            #         user_id=scoped_user_id,
            #         all_users=not only_mine,
            #     )
            else:
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
            succeeded = []
            not_found = []
            failed_ids = parsed_ids if parsed_ids else ["*"]
            failed = [{"tunnel_id": tid, "error": str(e)} for tid in failed_ids]
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
