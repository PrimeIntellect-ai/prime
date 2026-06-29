import asyncio
import signal
from typing import Any, List, Optional

from rich.table import Table

from prime_cli.command_configs import (
    TunnelListConfig,
    TunnelStartConfig,
    TunnelStatusConfig,
    TunnelStopConfig,
)
from prime_cli.utils import (
    get_console,
    human_age,
    iso_timestamp,
    output_data_as_json,
    validate_output_format,
)
from prime_cli.utils.prompt import confirm_or_skip

console = get_console()


def _create_tunnel_client() -> Any:
    from prime_tunnel.core.client import TunnelClient

    return TunnelClient()


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
        "http_user": getattr(tunnel, "http_user", None),
        "user_id": tunnel.user_id,
        "team_id": tunnel.team_id,
        "created_at": iso_timestamp(created_at) if created_at else None,
        "created": human_age(created_at) if created_at else None,
        "expires_at": iso_timestamp(tunnel.expires_at),
    }


def start_tunnel(config: TunnelStartConfig) -> None:
    """Start a tunnel to expose a local port."""
    port = config.port
    name = config.name
    labels = config.labels
    team_id = config.team_id
    auth = config.auth

    http_user: Optional[str] = None
    if auth is not None:
        http_user = auth.strip()
        if not http_user or ":" in http_user or " " in http_user:
            console.print(
                "[red]Invalid --auth username:[/red] must be non-empty without spaces or ':'",
                style="bold",
            )
            raise SystemExit(1)

    async def run_tunnel():
        from prime_tunnel import Tunnel
        from prime_tunnel.exceptions import (
            TunnelConnectionError,
            TunnelLimitReachedError,
            TunnelTimeoutError,
        )

        tunnel = Tunnel(
            local_port=port,
            name=name,
            team_id=team_id,
            labels=labels,
            http_user=http_user,
        )

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
            if http_user:
                console.print(f"[bold]Basic auth user:[/bold] {http_user}")
                console.print(f"[bold]Basic auth password:[/bold] {tunnel.http_password}")
                console.print(
                    "[yellow]Save this password - it is shown only once and "
                    "cannot be retrieved later[/yellow]"
                )
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
            raise SystemExit(1)
        except TunnelLimitReachedError as e:
            console.print(f"\n[red]Tunnel limit reached:[/red] {e}", style="bold")
            console.print("[dim]Delete an existing tunnel before creating a new one.[/dim]")
            raise SystemExit(1)
        except TunnelTimeoutError as e:
            console.print(f"\n[red]Connection timed out:[/red] {e}", style="bold")
            raise SystemExit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}", style="bold")
            raise SystemExit(1)
        finally:
            await tunnel.stop()
            console.print("[green]Tunnel stopped[/green]")

    try:
        asyncio.run(run_tunnel())
    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise SystemExit(1)


def list_tunnels(config: TunnelListConfig) -> None:
    """List active tunnels."""
    team_id = config.team_id
    labels = config.labels
    status = config.status
    sort_by = config.sort_by
    sort_order = config.sort_order
    page = config.page
    num = config.num
    output = config.output

    validate_output_format(output, console)

    async def fetch_tunnels():
        client = _create_tunnel_client()
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
        raise SystemExit(1)

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


def tunnel_status(config: TunnelStatusConfig) -> None:
    """Get status of a specific tunnel."""
    tunnel_id = config.tunnel_id

    async def fetch_status():
        client = _create_tunnel_client()
        try:
            return await client.get_tunnel(tunnel_id)
        finally:
            await client.close()

    try:
        tunnel = asyncio.run(fetch_status())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        raise SystemExit(1)

    if not tunnel:
        console.print(f"[red]Tunnel not found:[/red] {tunnel_id}")
        raise SystemExit(1)

    console.print(f"[bold]Tunnel ID:[/bold] {tunnel.tunnel_id}")
    console.print(f"[bold]URL:[/bold] {tunnel.url}")
    console.print(f"[bold]Hostname:[/bold] {tunnel.hostname}")
    console.print(f"[bold]Status:[/bold] {tunnel.status or 'unknown'}")
    if tunnel.labels:
        console.print(f"[bold]Labels:[/bold] {', '.join(tunnel.labels)}")
    console.print(f"[bold]Expires At:[/bold] {tunnel.expires_at}")


def stop_tunnel(config: TunnelStopConfig) -> None:
    """Stop and delete one or more tunnels.

    Select tunnels in exactly one of two ways: pass explicit tunnel IDs, or use
    the filter flags --all/--label/--status (which can be combined with each
    other but not with explicit IDs). By default filters apply to your tunnels;
    --all-users applies them to every member of a team.
    """
    tunnel_ids = config.tunnel_ids
    all = config.all
    labels = config.labels
    status = config.status
    team_id = config.team_id
    yes = config.yes
    all_users = config.all_users

    if tunnel_ids and (all or labels or status):
        console.print(
            "[red]Error:[/red] Tunnel IDs cannot be combined with --all, --label, or --status"
        )
        raise SystemExit(1)

    if all and labels:
        console.print("[red]Error:[/red] Use only one of --all or --label")
        raise SystemExit(1)

    if not all and not tunnel_ids and not labels and not status:
        console.print("[red]Error:[/red] Must specify tunnel IDs, --all, --label, or --status")
        raise SystemExit(1)

    if status is not None:
        status = status.strip().lower()
        allowed_statuses = {"pending", "connected", "disconnected"}
        if status not in allowed_statuses:
            console.print(
                "[red]Error:[/red] --status must be one of: " + ", ".join(sorted(allowed_statuses))
            )
            raise SystemExit(1)

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
            raise SystemExit(1)

    is_filter_delete = bool(all or labels or status)

    if is_filter_delete:

        async def validate_filter_scope() -> None:
            client = _create_tunnel_client()
            try:
                scoped_user_id = None if all_users else client.config.user_id
                scoped_team_id = team_id if team_id is not None else client.config.team_id
                if not all_users and not scoped_user_id:
                    raise ValueError(
                        "Cannot resolve current user ID for scoped bulk delete. "
                        "Run `prime login`, set PRIME_USER_ID, or delete explicit tunnel IDs."
                    )
                if all_users and not scoped_team_id:
                    raise ValueError("all_users requires a team ID")
            finally:
                await client.close()

        try:
            asyncio.run(validate_filter_scope())
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}", style="bold")
            raise SystemExit(1)

        filter_parts: List[str] = []
        if status:
            filter_parts.append(f"status '{status}'")
        if labels:
            filter_parts.append(f"label(s) {', '.join(labels)}")
        subject = f"tunnels matching {' and '.join(filter_parts)}" if filter_parts else "tunnels"

        if team_id and all_users:
            who = f"all {subject} in team {team_id}"
        elif team_id:
            who = f"your {subject} in team {team_id}"
        elif all_users:
            who = f"all {subject} (all users)"
        else:
            who = f"your {subject}"

        confirmation_msg = f"Are you sure you want to stop {who}? This action cannot be undone."
        cancel_msg = "Stop cancelled"
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
        client = _create_tunnel_client()
        succeeded: List[str] = []
        not_found: List[dict] = []
        failed: List[dict] = []
        try:
            if parsed_ids:
                result = await client.bulk_delete_tunnels(parsed_ids)
            else:
                scoped_team_id = team_id if team_id is not None else client.config.team_id
                scoped_user_id = None if all_users else client.config.user_id
                if not all_users and not scoped_user_id:
                    raise ValueError(
                        "Cannot resolve current user ID for scoped bulk delete. "
                        "Run `prime login`, set PRIME_USER_ID, or delete explicit tunnel IDs."
                    )
                result = await client.bulk_delete_tunnels(
                    labels=labels or None,
                    status=status,
                    team_id=scoped_team_id,
                    user_id=scoped_user_id,
                    all_users=all_users,
                )
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
        raise SystemExit(1)

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
        raise SystemExit(1)

    if parsed_ids:
        console.print("[green]All specified tunnels deleted successfully[/green]")
    else:
        console.print("[green]All matching tunnels deleted successfully[/green]")
