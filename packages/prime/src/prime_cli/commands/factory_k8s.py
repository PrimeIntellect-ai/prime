"""`prime factory k8s` — kubeconfig setup for researcher cluster access.

`prime factory k8s login <name>` writes a kubeconfig whose credentials come from an
`exec` block calling back into this CLI. The kubeconfig holds no Kubernetes
token: every `kubectl` invocation shells out to `prime auth k8s-token`, which
exchanges the user's platform API token for a short-lived ServiceAccount token
(cached privately under ~/.prime/kube/cache until shortly before it expires).

That indirection is what makes revocation work. Deleting the grant deletes the
namespace and its ServiceAccount, so any token already issued stops working
and the next mint is refused.

Researchers never reach a cluster's apiserver directly. Each pool has its own
Tailscale API proxy on the customer's tailnet, so every grant carries its own
`server`, and the proxy passes the apiserver's TLS through: the client has to
verify the apiserver certificate under `tlsServerName`, not the proxy's name.
"""

import ipaddress
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import typer
import yaml
from rich.markup import escape
from rich.table import Table

from ..client import APIClient, APIError
from ..utils import (
    PlainTyper,
    get_console,
    output_data_as_json,
    validate_output_format,
)
from ..utils import kube_token_cache as token_cache

app = PlainTyper(
    help="Kubernetes reservations on dedicated clusters",
    no_args_is_help=True,
)


@app.callback()
def _factory_k8s() -> None:
    """Present so Typer keeps this a command group whatever subcommands it
    has, rather than promoting a lone one to the group."""


console = get_console()

# Standalone file rather than merging into ~/.kube/config: merging means
# rewriting a file the researcher may share with unrelated clusters, and a bad
# merge is far more annoying than an extra environment variable.
KUBECONFIG_DIR = Path.home() / ".prime" / "kube"


# What `prime factory k8s list` and a refused login say when there is nothing to
# log in to: the platform cannot grant access itself, so point at who can.
NO_RESERVATION_HINT = (
    "You're not a member of any Kubernetes reservation. Ask your team admin to "
    "add you, or reserve nodes from the dashboard."
)

# Whole budget for the post-login DNS check, across every context: a warning
# is never worth more than a moment of the researcher's time.
DNS_CHECK_TIMEOUT_SECONDS = 2.0


def _kubeconfig_path(cluster: str) -> Path:
    return KUBECONFIG_DIR / f"{cluster}.yaml"


def _context_name(cluster: str, pool: str) -> str:
    return f"{cluster}-{pool}"


def _build_kubeconfig(
    *,
    cluster: str,
    ca_data: str,
    grants: List[Dict[str, Any]],
    base_url: str,
    current_pool: Optional[str] = None,
    environment: Optional[str] = None,
    key_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a kubeconfig with one cluster entry and one context per reachable
    pool grant. The current context is `current_pool`'s, or else the first
    grant's.

    Every grant must already carry a `server`. Pools do not share an endpoint
    (each has its own API proxy), so each context points at its own cluster
    entry, while the CA is the cluster's and is shared.

    The `exec` block is the whole point: `interactiveMode: Never` because
    kubectl must not try to prompt, and `provideClusterInfo: false` because the
    plugin resolves the cluster from its own arguments.
    """
    # `prime --context X factory k8s login` runs with PRIME_CONTEXT set by the root
    # callback. The credential plugin runs much later, from kubectl, where that
    # variable does not exist — so the flag has to be baked into the exec args
    # or every refresh would silently hit the default context's platform.
    # A custom environment selected persistently (`prime config use staging`)
    # is pinned the same way: otherwise switching environments later would
    # send the new environment's API key to the old platform's URL. The
    # built-in production context carries no key of its own, so pinning it
    # would change nothing.
    prime_context = os.getenv("PRIME_CONTEXT")
    if not prime_context and environment and environment.lower() != "production":
        prime_context = environment
    context_args = ["--context", prime_context] if prime_context else []

    # The context flag alone is not enough: reloading a context re-resolves
    # `base_url` from the *environment's* file (or, for the built-in
    # production context, forces the public default), which discards a custom
    # URL set via `prime config set-base-url`. So the base URL the login
    # itself resolved is pinned into the exec environment — PRIME_API_BASE_URL
    # outranks every other source in `Config.base_url`, making each refresh
    # hit the same platform the login did, while `--context` still selects the
    # matching API key.
    # kubectl forwards plugin stderr to the terminal on every call, so an
    # update banner would be printed above the output of every kubectl command.
    exec_env = [
        {"name": "PRIME_API_BASE_URL", "value": base_url},
        {"name": "PRIME_DISABLE_VERSION_CHECK", "value": "1"},
    ]
    # The plugin reads whichever API key is current when kubectl runs it. The
    # login's key fingerprint keeps a key selected later, for another
    # platform, from being sent to this pinned URL.
    if key_fingerprint:
        exec_env.append({"name": token_cache.KEY_FINGERPRINT_ENV, "value": key_fingerprint})

    clusters = []
    contexts = []
    users = []
    for grant in grants:
        pool = grant["pool"]
        context_name = _context_name(cluster, pool)
        cluster_entry: Dict[str, Any] = {
            "server": grant["server"],
            "certificate-authority-data": ca_data,
        }
        if grant.get("tlsServerName"):
            cluster_entry["tls-server-name"] = grant["tlsServerName"]
        clusters.append({"name": context_name, "cluster": cluster_entry})
        contexts.append(
            {
                "name": context_name,
                "context": {
                    "cluster": context_name,
                    "user": context_name,
                    "namespace": grant["namespace"],
                },
            }
        )
        users.append(
            {
                "name": context_name,
                "user": {
                    "exec": {
                        "apiVersion": "client.authentication.k8s.io/v1",
                        "command": "prime",
                        "args": context_args
                        + [
                            "auth",
                            "k8s-token",
                            "--cluster",
                            cluster,
                            "--pool",
                            pool,
                        ],
                        "env": exec_env,
                        "interactiveMode": "Never",
                        "provideClusterInfo": False,
                    }
                },
            }
        )

    if current_pool is not None:
        current = _context_name(cluster, current_pool)
    else:
        current = contexts[0]["name"] if contexts else ""
    return {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": clusters,
        "contexts": contexts,
        "users": users,
        "current-context": current,
    }


@app.command("list")
def list_clusters(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List the Kubernetes reservations you can log in to."""
    validate_output_format(output, console)
    try:
        data: Dict[str, Any] = APIClient().get("/clusters")
    except APIError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    clusters = data.get("clusters") or []
    if output == "json":
        output_data_as_json({"clusters": clusters}, console)
        return
    if not clusters:
        console.print(NO_RESERVATION_HINT)
        return

    # The table holds only short values. Namespaces and login commands are long
    # and meant to be copied, so they go below it, one reservation per block,
    # printed whole: Rich would cut a cell to "…" to fit the terminal.
    table = Table(title="Kubernetes reservations", show_lines=True)
    table.add_column("Cluster", style="cyan", overflow="fold")
    table.add_column("Pool", style="blue", overflow="fold")
    table.add_column("Status", style="white", no_wrap=True)
    table.add_column("Proxy", style="white", no_wrap=True)
    details: List[List[str]] = []
    for entry in clusters:
        cluster = entry.get("cluster") or "?"
        reservations = entry.get("reservations") or []
        loginable = [r for r in reservations if r.get("server")]
        for reservation in reservations:
            pool = reservation.get("pool") or "?"
            table.add_row(
                _cluster_label(entry),
                escape(pool),
                _status_label(reservation.get("status")),
                _proxy_label(reservation.get("proxyReady")),
            )
            details.append(
                [
                    f"[bold]{escape(cluster)}[/bold] / [bold]{escape(pool)}[/bold]",
                    f"  Namespace: {escape(reservation.get('namespace') or '-')}",
                    "  " + _next_step(cluster, reservation, several=len(loginable) > 1),
                ]
            )
    console.print(table)
    for block in details:
        console.print()
        for line in block:
            console.print(line, soft_wrap=True, highlight=False)


def _cluster_label(entry: Dict[str, Any]) -> str:
    name = escape(entry.get("cluster") or "?")
    display = entry.get("displayName")
    return f"{name}\n[dim]{escape(display)}[/dim]" if display and display != name else name


def _proxy_label(ready: Any) -> str:
    return "[green]ready[/green]" if ready else "[yellow]not ready[/yellow]"


_STATUS_STYLES = {
    "Ready": "green",
    "Provisioning": "yellow",
    "Releasing": "dim",
    "Failed": "red",
}


def _status_label(status: Optional[str]) -> str:
    status = status or "Unknown"
    style = _STATUS_STYLES.get(status, "white")
    return f"[{style}]{status}[/{style}]"


def _next_step(cluster: str, reservation: Dict[str, Any], *, several: bool) -> str:
    if reservation.get("server"):
        command = f"prime factory k8s login {cluster}"
        if several:
            command += f" --pool {reservation.get('pool')}"
        return f"Log in:    {escape(command)}"
    reason = reservation.get("unavailableReason") or "Not ready yet."
    return f"Not ready: [dim]{escape(reason)}[/dim]"


@app.command("login")
def login(
    cluster: str = typer.Argument(..., help="Cluster name, as shown by `prime factory k8s list`"),
    pool: Optional[str] = typer.Option(
        None,
        "--pool",
        help="Make this pool's context the current one (default: the first, alphabetically)",
    ),
) -> None:
    """Write a kubeconfig for a cluster you have been granted access to."""
    # Re-login is the documented step after a grant changes, and a changed
    # grant can mean a new ServiceAccount, so cached tokens for this cluster go
    # first, whatever the rest of the login does.
    try:
        token_cache.clear_cluster(cluster)
    except OSError as e:
        console.print(f"[yellow]Could not clear cached credentials for '{cluster}': {e}[/yellow]")

    path = _kubeconfig_path(cluster)
    try:
        client = APIClient()
        info: Dict[str, Any] = client.get(f"/clusters/{cluster}/kubeconfig-info")
    except APIError as e:
        detail = str(e)
        if detail.startswith("HTTP 403:"):
            # The platform's "no live grant": neutral about whether the
            # cluster exists, and permanent until someone grants access.
            console.print(f"[red]{detail.removeprefix('HTTP 403:').strip()}[/red]")
            _remove_stale_kubeconfig(path)
            console.print("Run `prime factory k8s list` to see the reservations you can log in to.")
        else:
            console.print(f"[red]{detail}[/red]")
        raise typer.Exit(1) from e

    grants = info.get("grants") or []
    if not grants:
        console.print(f"[red]No active access grant for '{cluster}'.[/red]")
        _remove_stale_kubeconfig(path)
        console.print("Run `prime factory k8s list` to see the reservations you can log in to.")
        raise typer.Exit(1)

    ca_data = info.get("certificateAuthorityData")
    if not ca_data:
        console.print(
            f"[red]The platform returned incomplete connection info for '{cluster}'.[/red]"
        )
        raise typer.Exit(1)

    # Any top-level `server` (an older platform's single cluster endpoint) is
    # deliberately ignored: the only supported path to the apiserver is the
    # grant's own API proxy on the researcher's tailnet.
    reachable = []
    for grant in grants:
        name = grant.get("pool") or "?"
        reason = _unreachable_reason(grant)
        if reason:
            console.print(f"[yellow]Skipping pool '{name}': {reason}[/yellow]")
        else:
            reachable.append(grant)

    if not reachable:
        # Failing before the write leaves any earlier kubeconfig as it was,
        # rather than replacing it with one that has no contexts.
        console.print(
            f"[red]None of your pools on '{cluster}' is reachable yet; "
            f"{path} was not written.[/red]"
        )
        if path.exists():
            console.print(
                f"[yellow]The existing {path} was kept, but its contexts may be stale.[/yellow]"
            )
        raise typer.Exit(1)

    reachable_pools = [grant["pool"] for grant in reachable]
    if pool is not None and pool not in reachable_pools:
        # Checked before the write, so a typo never replaces a working file.
        known = next((g for g in grants if g.get("pool") == pool), None)
        why = (
            f"is not reachable: {_unreachable_reason(known)}"
            if known is not None
            else "is not one of your reservations there."
        )
        console.print(f"[red]Pool '{pool}' on '{cluster}' {why}[/red]")
        console.print(f"Available pools: {', '.join(reachable_pools)}")
        raise typer.Exit(1)

    kubeconfig = _build_kubeconfig(
        cluster=cluster,
        ca_data=ca_data,
        grants=reachable,
        # The exact URL this login request just used — not re-derived at
        # refresh time, where context resolution would lose a custom one.
        base_url=client.base_url,
        current_pool=pool,
        environment=client.config.current_environment,
        key_fingerprint=token_cache.key_fingerprint(
            base_url=client.base_url, api_key=client.api_key
        ),
    )

    # The file is regenerated from scratch, so contexts for grants that are
    # gone or currently unreachable drop out on every login.
    try:
        _write_private(path, yaml.safe_dump(kubeconfig, sort_keys=False))
    except OSError as e:
        console.print(f"[red]Could not write {path}: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]Wrote kubeconfig to {path}[/green]")
    current = kubeconfig["current-context"]
    console.print(f"Current context: {current}")
    others = [c["name"] for c in kubeconfig["contexts"] if c["name"] != current]
    if others:
        console.print(f"Other contexts: {', '.join(others)}")
        console.print(f"Switch with: kubectl config use-context {others[0]}")
    console.print("\nUse it with:")
    console.print(f"  export KUBECONFIG={path}")
    console.print("  kubectl get pods")

    _warn_unresolvable_proxies(reachable)


def _remove_stale_kubeconfig(path: Path) -> None:
    """A refused login means none of the file's contexts can work any more;
    leaving it would only produce confusing kubectl errors later. The
    cluster's cached tokens were already cleared at the start of the login."""
    if not path.exists():
        return
    try:
        path.unlink()
    except OSError as e:
        console.print(f"[yellow]Could not remove the stale {path}: {e}[/yellow]")
        return
    console.print(f"[yellow]Removed the stale kubeconfig {path}.[/yellow]")


def _resolve(host: str) -> None:
    socket.getaddrinfo(host, None)


def _unresolved_hosts(hosts: List[str], timeout: float) -> List[str]:
    """The hosts that did not resolve within `timeout` seconds in total.

    Daemon threads rather than an executor: a lookup stuck in the system
    resolver must not keep the process alive after the login has finished.
    """
    resolved: set = set()
    lock = threading.Lock()

    def _lookup(host: str) -> None:
        try:
            _resolve(host)
        except (OSError, UnicodeError):
            return
        with lock:
            resolved.add(host)

    threads = [threading.Thread(target=_lookup, args=(h,), daemon=True) for h in hosts]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    with lock:
        return [h for h in hosts if h not in resolved]


def _warn_unresolvable_proxies(grants: List[Dict[str, Any]]) -> None:
    """Warn, never fail, when a written context's proxy name does not resolve.

    The proxy is only reachable on the reservation's tailnet, so a name that
    does not resolve almost always means Tailscale is off or on another
    tailnet — which kubectl would otherwise report as a bare dial error.
    """
    hosts: List[str] = []
    for grant in grants:
        host = urlparse(grant["server"]).hostname
        if not host or host in hosts or _is_ip_literal(host):
            continue
        hosts.append(host)
    if not hosts:
        return
    for host in _unresolved_hosts(hosts, DNS_CHECK_TIMEOUT_SECONDS):
        console.print(
            f"[yellow]Can't resolve {host}. Make sure Tailscale is connected to your "
            "team's tailnet (the one the reservation's proxy joined), then retry.[/yellow]"
        )


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _unreachable_reason(grant: Dict[str, Any]) -> Optional[str]:
    """Why a grant cannot get a context, or None when it can."""
    if not grant.get("pool") or not grant.get("namespace"):
        return "the platform returned an incomplete grant."
    if not grant.get("server"):
        return (
            grant.get("unavailableReason")
            or "the platform returned no API endpoint for this pool yet."
        )
    return None


def _write_private(path: Path, content: str) -> None:
    """Replace `path` atomically with a 0600 file.

    It carries no credential, but it does name every namespace the user can
    reach, so it is never group-readable — not even briefly, which is why the
    temp file is created 0600 rather than chmod-ed after the fact. Writing
    beside the target and renaming means kubectl never sees a half-written
    file and a failed login leaves the previous one intact.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
