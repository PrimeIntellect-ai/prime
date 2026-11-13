"""`prime cluster` — kubeconfig setup for researcher cluster access.

`prime cluster login <name>` writes a kubeconfig whose credentials come from an
`exec` block calling back into this CLI. No Kubernetes token is ever written to
disk: every `kubectl` invocation shells out to `prime auth k8s-token`, which
exchanges the user's platform API token for a short-lived ServiceAccount token.

That indirection is what makes revocation work. Deleting the grant deletes the
namespace and its ServiceAccount, so the next refresh fails and any token
already issued expires within the hour.

Researchers never reach a cluster's apiserver directly. Each pool has its own
Tailscale API proxy on the customer's tailnet, so every grant carries its own
`server`, and the proxy passes the apiserver's TLS through: the client has to
verify the apiserver certificate under `tlsServerName`, not the proxy's name.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

from ..client import APIClient, APIError
from ..utils import PlainTyper, get_console

app = PlainTyper(
    help="Kubernetes access for clusters you have been granted",
    no_args_is_help=True,
)


@app.callback()
def _cluster() -> None:
    """Present so Typer keeps `login` as a subcommand rather than promoting it
    to the group when it is the only one."""


console = get_console()

# Standalone file rather than merging into ~/.kube/config: merging means
# rewriting a file the researcher may share with unrelated clusters, and a bad
# merge is far more annoying than an extra environment variable.
KUBECONFIG_DIR = Path.home() / ".prime" / "kube"


def _kubeconfig_path(cluster: str) -> Path:
    return KUBECONFIG_DIR / f"{cluster}.yaml"


def _build_kubeconfig(
    *,
    cluster: str,
    ca_data: str,
    grants: List[Dict[str, Any]],
    base_url: str,
) -> Dict[str, Any]:
    """Render a kubeconfig with one cluster entry and one context per reachable
    pool grant.

    Every grant must already carry a `server`. Pools do not share an endpoint
    (each has its own API proxy), so each context points at its own cluster
    entry, while the CA is the cluster's and is shared.

    The `exec` block is the whole point: `interactiveMode: Never` because
    kubectl must not try to prompt, and `provideClusterInfo: false` because the
    plugin resolves the cluster from its own arguments.
    """
    # `prime --context X cluster login` runs with PRIME_CONTEXT set by the root
    # callback. The credential plugin runs much later, from kubectl, where that
    # variable does not exist — so the flag has to be baked into the exec args
    # or every refresh would silently hit the default context's platform.
    prime_context = os.getenv("PRIME_CONTEXT")
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

    clusters = []
    contexts = []
    users = []
    for grant in grants:
        pool = grant["pool"]
        context_name = f"{cluster}-{pool}"
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

    return {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": clusters,
        "contexts": contexts,
        "users": users,
        "current-context": contexts[0]["name"] if contexts else "",
    }


@app.command("login")
def login(
    cluster: str = typer.Argument(..., help="Cluster name, as shown by your admin"),
) -> None:
    """Write a kubeconfig for a cluster you have been granted access to."""
    try:
        client = APIClient()
        info: Dict[str, Any] = client.get(f"/clusters/{cluster}/kubeconfig-info")
    except APIError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    grants = info.get("grants") or []
    if not grants:
        console.print(f"[red]No active access grant for '{cluster}'.[/red]")
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
        pool = grant.get("pool") or "?"
        reason = _unreachable_reason(grant)
        if reason:
            console.print(f"[yellow]Skipping pool '{pool}': {reason}[/yellow]")
        else:
            reachable.append(grant)

    path = _kubeconfig_path(cluster)
    if not reachable:
        # Failing before the write leaves any earlier kubeconfig as it was,
        # rather than replacing it with one that has no contexts.
        console.print(
            f"[red]None of your pools on '{cluster}' is reachable yet; "
            f"{path} was not written.[/red]"
        )
        raise typer.Exit(1)

    kubeconfig = _build_kubeconfig(
        cluster=cluster,
        ca_data=ca_data,
        grants=reachable,
        # The exact URL this login request just used — not re-derived at
        # refresh time, where context resolution would lose a custom one.
        base_url=client.base_url,
    )

    # The file is regenerated from scratch, so contexts for grants that are
    # gone or currently unreachable drop out on every login.
    try:
        _write_private(path, yaml.safe_dump(kubeconfig, sort_keys=False))
    except OSError as e:
        console.print(f"[red]Could not write {path}: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]Wrote kubeconfig to {path}[/green]")
    if len(reachable) > 1:
        pools = ", ".join(grant["pool"] for grant in reachable)
        console.print(f"Contexts for pools: {pools}")
    console.print("\nUse it with:")
    console.print(f"  export KUBECONFIG={path}")
    console.print("  kubectl get pods")


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
