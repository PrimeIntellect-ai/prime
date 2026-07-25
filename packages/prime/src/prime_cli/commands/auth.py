"""`prime auth` — credential plugin entry points.

`prime auth k8s-token` is a Kubernetes client-go credential plugin. kubectl
runs it on every API call and reads an ExecCredential from stdout, so this
command has a stricter contract than the rest of the CLI:

* **stdout is protocol.** Only the ExecCredential JSON goes there. Every
  human-readable message goes to stderr, or kubectl fails to parse the reply
  and reports something unhelpful about the credential plugin.
* **Exit codes distinguish transient from permanent.** kubectl surfaces the
  message either way, but a wrapper script — or a human reading `$?` — needs to
  know whether retrying could help. Retrying a revoked grant never will.
* **No prompting, ever.** The kubeconfig sets `interactiveMode: Never`; there
  may be no terminal attached at all.

It deliberately does not use the shared `APIClient`: that collapses HTTP status
codes into a single `APIError`, and the whole failure contract here is a
function of the status code.
"""

import json
import sys
from typing import NoReturn

import httpx
import typer

from prime_cli.core import Config

from ..utils import PlainTyper

app = PlainTyper(
    help="Authentication helpers (used by other tools, not usually by hand)",
    no_args_is_help=True,
)


@app.callback()
def _auth() -> None:
    """Present so Typer keeps `k8s-token` as a subcommand.

    Without a callback, a Typer app holding exactly one command promotes it to
    the group itself — `prime auth` would run the plugin and `prime auth
    k8s-token` would fail, which is the exact string every kubeconfig we write
    contains.
    """


# Exit codes. 1 is "try again later", everything above it is "this will not
# work until something changes".
EXIT_UNREACHABLE = 1
EXIT_FORBIDDEN = 2
EXIT_RATE_LIMITED = 3
EXIT_AUTH_EXPIRED = 4
EXIT_AMBIGUOUS = 5

_TIMEOUT_SECONDS = 15.0


def _fail(message: str, code: int) -> NoReturn:
    print(message, file=sys.stderr)
    raise typer.Exit(code)


@app.command("k8s-token")
def k8s_token(
    cluster: str = typer.Option(..., "--cluster", help="Cluster name"),
    pool: str = typer.Option(
        None, "--pool", help="Pool name, when you have access to more than one"
    ),
) -> None:
    """Print a Kubernetes ExecCredential for a granted cluster.

    Invoked by kubectl via the `exec` block that `prime cluster login` writes.
    """
    config = Config()
    api_key = config.api_key
    if not api_key:
        _fail("Not logged in. Run `prime login`.", EXIT_AUTH_EXPIRED)

    url = f"{config.base_url}/clusters/{cluster}/kube-token"
    params = {"pool": pool} if pool else None

    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            timeout=_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException:
        _fail(
            f"Timed out reaching the Prime platform at {config.base_url}.",
            EXIT_UNREACHABLE,
        )
    except httpx.TransportError as e:
        _fail(
            f"Cannot reach the Prime platform at {config.base_url}: {e}",
            EXIT_UNREACHABLE,
        )

    if response.status_code == 401:
        _fail("Platform auth expired. Run `prime login`.", EXIT_AUTH_EXPIRED)

    if response.status_code == 403:
        _fail(
            f"Access to '{cluster}' has been revoked, or was never granted. Contact an admin.",
            EXIT_FORBIDDEN,
        )

    if response.status_code == 409:
        # Several pools and no --pool. The server lists them in the detail,
        # which is more useful than anything this side could reconstruct.
        _fail(_detail(response) or "Specify --pool.", EXIT_AMBIGUOUS)

    if response.status_code == 429:
        # Honour Retry-After silently when present: a well-behaved kubectl is
        # about to back off anyway, and a scary error for what is usually a
        # hot loop just confuses people.
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            _fail(
                f"Too many credential requests. Retry after {retry_after}s.",
                EXIT_RATE_LIMITED,
            )
        _fail("Too many credential refresh attempts.", EXIT_RATE_LIMITED)

    if response.status_code >= 500:
        _fail(
            f"Prime platform error ({response.status_code}). Try again shortly.",
            EXIT_UNREACHABLE,
        )

    if response.status_code != 200:
        _fail(
            _detail(response) or f"Unexpected response ({response.status_code}).",
            EXIT_UNREACHABLE,
        )

    try:
        credential = response.json()
    except ValueError:
        _fail("Platform returned a malformed credential.", EXIT_UNREACHABLE)

    if not isinstance(credential, dict) or "status" not in credential:
        _fail("Platform returned a malformed credential.", EXIT_UNREACHABLE)

    # stdout, and nothing else on stdout.
    sys.stdout.write(json.dumps(credential))
    sys.stdout.flush()


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) else ""
