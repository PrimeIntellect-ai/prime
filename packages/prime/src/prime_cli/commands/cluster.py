"""`prime cluster` — kubeconfig setup for researcher cluster access.

`prime cluster login <name>` writes a kubeconfig whose credentials come from an
`exec` block calling back into this CLI. No Kubernetes token is ever written to
disk: every `kubectl` invocation shells out to `prime auth k8s-token`, which
exchanges the user's platform API token for a short-lived ServiceAccount token.

That indirection is what makes revocation work. Deleting the grant deletes the
namespace and its ServiceAccount, so the next refresh fails and any token
already issued expires within the hour.
"""

from pathlib import Path
from typing import Any, Dict, List

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
    *, cluster: str, server: str, ca_data: str, grants: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Render a kubeconfig with one context per pool the user has access to.

    The `exec` block is the whole point: `interactiveMode: Never` because
    kubectl must not try to prompt, and `provideClusterInfo: false` because the
    plugin resolves the cluster from its own arguments.
    """
    contexts = []
    users = []
    for grant in grants:
        pool = grant["pool"]
        context_name = f"{cluster}-{pool}"
        contexts.append(
            {
                "name": context_name,
                "context": {
                    "cluster": cluster,
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
                        "args": [
                            "auth",
                            "k8s-token",
                            "--cluster",
                            cluster,
                            "--pool",
                            pool,
                        ],
                        "interactiveMode": "Never",
                        "provideClusterInfo": False,
                    }
                },
            }
        )

    return {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": cluster,
                "cluster": {
                    "server": server,
                    "certificate-authority-data": ca_data,
                },
            }
        ],
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

    kubeconfig = _build_kubeconfig(
        cluster=cluster,
        server=info["server"],
        ca_data=info["certificateAuthorityData"],
        grants=grants,
    )

    path = _kubeconfig_path(cluster)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(kubeconfig, sort_keys=False))
    # It carries no credential, but it does name every namespace the user can
    # reach, so don't leave it group-readable.
    path.chmod(0o600)

    console.print(f"[green]Wrote kubeconfig to {path}[/green]")
    if len(grants) > 1:
        pools = ", ".join(grant["pool"] for grant in grants)
        console.print(f"Contexts for pools: {pools}")
    console.print("\nUse it with:")
    console.print(f"  export KUBECONFIG={path}")
    console.print("  kubectl get pods")
