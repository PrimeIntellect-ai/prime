"""`prime auth` — credential plugin entry points.

`prime auth k8s-token` is a Kubernetes client-go credential plugin. kubectl
runs it once per process (so once per `kubectl` command) and reads an
ExecCredential from stdout, so this command has a stricter contract than the
rest of the CLI:

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

Minted credentials are cached on disk (see `kube_token_cache`) and reused until
shortly before they expire, because minting on every kubectl command runs into
the platform's rate limit.

A cached token can still be rejected before it expires: revoking a grant
deletes its ServiceAccount, and the token dies with it. client-go reacts to a
401 by running the plugin again, from the same kubectl process, to refresh its
credential for the next request (the failed request itself is not retried).
Handing back the same cached token would only repeat the 401, so a second
run from the same parent process while the token it was given is still fresh
is taken as that rejection: the entry is skipped and a new token minted,
which for a revoked grant fails with the platform's 403 reason instead. That
happens at most once per process per entry, so a caller that simply re-runs
the plugin without caching (some client libraries do) costs one extra mint,
not one per call.
"""

import json
import os
import sys
from typing import NoReturn

import httpx
import typer

from prime_cli.core import Config

from ..utils import PlainTyper
from ..utils import kube_token_cache as token_cache

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


# Exit codes. 1 is "try again later" (unreachable, or a 503 from the platform),
# everything above it is "this will not work until something changes".
EXIT_UNREACHABLE = 1
EXIT_FORBIDDEN = 2
EXIT_RATE_LIMITED = 3
EXIT_AUTH_EXPIRED = 4
EXIT_AMBIGUOUS = 5

_TIMEOUT_SECONDS = 15.0


def _fail(message: str, code: int) -> NoReturn:
    print(message, file=sys.stderr)
    raise typer.Exit(code)


def _check_key_binding(base_url: str, api_key: str, cluster: str) -> None:
    """The kubeconfig pins the platform URL; the key is whichever is current.
    Send it only if it is the key, for this platform, that the login used."""
    expected = os.environ.get(token_cache.KEY_FINGERPRINT_ENV)
    if not expected:
        _fail(
            f"This kubeconfig predates key binding. Run `prime factory k8s login {cluster}`.",
            EXIT_AUTH_EXPIRED,
        )
    if token_cache.key_fingerprint(base_url=base_url, api_key=api_key) != expected:
        _fail(
            f"The current API key or platform differs from the one this kubeconfig was "
            f"written with ({base_url}). Run `prime factory k8s login {cluster}` "
            "with the key you want to use.",
            EXIT_AUTH_EXPIRED,
        )


_MAX_REDIRECTS = 5


class _CrossOriginRedirect(Exception):
    def __init__(self, target: str):
        super().__init__(target)
        self.target = target


def _origin(url: httpx.URL) -> tuple:
    return (url.scheme, url.host, url.port)


def _post_same_origin(url: str, *, api_key: str, params) -> httpx.Response:
    """POST with the bearer, following redirects only within the same origin."""
    origin = _origin(httpx.URL(url))
    for _ in range(_MAX_REDIRECTS + 1):
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            timeout=_TIMEOUT_SECONDS,
            follow_redirects=False,
        )
        if not response.is_redirect:
            return response
        target = response.request.url.join(response.headers["Location"])
        if _origin(target) != origin:
            raise _CrossOriginRedirect(str(target))
        url, params = str(target), None
    return response


@app.command("k8s-token")
def k8s_token(
    cluster: str = typer.Option(..., "--cluster", help="Cluster name"),
    pool: str = typer.Option(
        None, "--pool", help="Pool name, when you have access to more than one"
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help=(
            "Mint a new token every time, without reading or writing the on-disk "
            "cache. Also set by PRIME_K8S_TOKEN_NO_CACHE=1, which reaches the plugin "
            "through kubectl without editing the kubeconfig"
        ),
    ),
) -> None:
    """Print a Kubernetes ExecCredential for a granted cluster.

    Invoked by kubectl via the `exec` block that `prime factory k8s login` writes.
    """
    config = Config()
    api_key = config.api_key
    if not api_key:
        _fail("Not logged in. Run `prime login`.", EXIT_AUTH_EXPIRED)
    _check_key_binding(config.base_url, api_key, cluster)

    # Parsed here rather than through Click's envvar support, which (as of
    # Click 8.2) turns a flag on for any non-empty value, "0" included.
    no_cache = no_cache or _env_flag("PRIME_K8S_TOKEN_NO_CACHE")
    cache_path = token_cache.entry_path(
        base_url=config.base_url, cluster=cluster, pool=pool or None, api_key=api_key
    )
    caller = os.getppid()
    reissue = False
    if not no_cache:
        entry = token_cache.load(cache_path)
        if entry is not None and token_cache.is_fresh(entry["credential"]):
            served_to = entry.get("servedTo")
            if served_to != caller or entry.get("reissuedFor") == caller:
                if served_to != caller:
                    # Bookkeeping for the rejection check; losing it only
                    # means a rejected token is served once more.
                    try:
                        token_cache.save(cache_path, {**entry, "servedTo": caller})
                    except OSError:
                        pass
                _emit(entry["credential"])
                return
            reissue = True

    # Config.base_url deliberately strips any /api/v1 suffix, and the shared
    # APIClient re-adds it on every request — so it must be added here too, or
    # every call 404s against the bare host.
    url = f"{config.base_url}/api/v1/clusters/{cluster}/kube-token"
    params = {"pool": pool} if pool else None

    try:
        response = _post_same_origin(url, api_key=api_key, params=params)
    except _CrossOriginRedirect as e:
        _fail(
            f"The Prime platform at {config.base_url} redirected to {e.target}; "
            "the credential is not sent to another host.",
            EXIT_UNREACHABLE,
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

    if response.status_code in (401, 403, 409):
        # Whatever was cached for this key is no longer something to serve.
        token_cache.evict(cache_path)

    if response.status_code == 401:
        _fail("Platform auth expired. Run `prime login`.", EXIT_AUTH_EXPIRED)

    if response.status_code == 403:
        _fail(
            _detail(response)
            or f"Access to '{cluster}' has been revoked, or was never granted. Contact an admin.",
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

    if response.status_code == 503:
        # The platform's signal that this should work again on its own: the
        # cluster is not ready, its heartbeat is stale, or the pool's API
        # capability was withdrawn for now. Its reason says which.
        _fail(
            _detail(response)
            or f"Cluster '{cluster}' is temporarily unavailable. Try again shortly.",
            EXIT_UNREACHABLE,
        )

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

    if not no_cache:
        if token_cache.is_fresh(credential):
            entry = {
                "credential": credential,
                "servedTo": caller,
                "reissuedFor": caller if reissue else None,
            }
            try:
                token_cache.save(cache_path, entry)
            except OSError as e:
                # The credential is good; failing to cache it only costs a
                # mint next time, so kubectl still gets it.
                print(
                    f"Warning: could not cache the credential under "
                    f"{token_cache.CACHE_DIR}: {e.strerror or e}",
                    file=sys.stderr,
                )
        else:
            # Nothing to cache, and anything older is superseded.
            token_cache.evict(cache_path)

    _emit(credential)


def _emit(credential: dict) -> None:
    # stdout, and nothing else on stdout.
    sys.stdout.write(json.dumps(credential))
    sys.stdout.flush()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, str) else ""
