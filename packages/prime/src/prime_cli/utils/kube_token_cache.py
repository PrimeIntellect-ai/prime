"""On-disk cache for the credentials `prime auth k8s-token` hands to kubectl.

client-go caches an exec plugin's credential only inside one process, so every
`kubectl` command runs the plugin again. Without this cache each of those runs
minted a fresh one-hour token, and the platform's mint rate limit (60 per
grant, 300 per user, per hour) locked out anyone polling `kubectl get pods`.

Entries live at ``~/.prime/kube/cache/<cluster hash>/<identity hash>.json``,
directories 0700 and files 0600. The identity is the platform base URL, the
cluster, the pool and the API key, all hashed into the file name, so a token
is only ever reused for the same platform, grant and key, and the key itself
is never on disk here.
Grouping by cluster is what lets `prime cluster login` drop a cluster's
entries without reading any of them.
"""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_DIR = Path.home() / ".prime" / "kube" / "cache"

# A cached token is served only while it has more than this left. client-go
# keeps using a credential until its exact expirationTimestamp, so the margin
# is what guarantees kubectl a working token for a command that runs for a few
# minutes and absorbs clock skew against the apiserver. Against a one-hour
# token it costs about one extra mint an hour per pool.
FRESHNESS_MARGIN = timedelta(minutes=5)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cluster_dir(cluster: str) -> Path:
    # Hashed rather than used verbatim: the name comes from a kubeconfig and
    # must never be able to point outside the cache.
    return CACHE_DIR / _sha256(cluster)[:16]


# PBKDF2 rather than a bare SHA-256 because the API key is part of what is
# hashed. The key is long and random, and ~/.prime/config.json beside this
# cache holds it in the clear anyway, so this is not what keeps it secret; it
# just means a file name is never a cheap way to test a guessed key. 100k
# rounds cost a few milliseconds per kubectl command.
_KEY_DERIVATION_SALT = b"prime-cli/kube-token-cache/v1"
_KEY_DERIVATION_ROUNDS = 100_000


def entry_path(*, base_url: str, cluster: str, pool: Optional[str], api_key: str) -> Path:
    identity = json.dumps([base_url, cluster, pool, api_key]).encode()
    digest = hashlib.pbkdf2_hmac(
        "sha256", identity, _KEY_DERIVATION_SALT, _KEY_DERIVATION_ROUNDS
    ).hex()
    return _cluster_dir(cluster) / f"{digest[:32]}.json"


def load(path: Path) -> Optional[Dict[str, Any]]:
    """The entry at `path`, or None when it is missing or unreadable."""
    try:
        entry = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("credential"), dict):
        return None
    return entry


def is_fresh(credential: Dict[str, Any]) -> bool:
    """Whether `credential` expires more than FRESHNESS_MARGIN from now.

    A credential without a parseable, timezone-aware expirationTimestamp is
    never fresh, so it is never cached.
    """
    status = credential.get("status")
    raw = status.get("expirationTimestamp") if isinstance(status, dict) else None
    if not isinstance(raw, str):
        return False
    try:
        expiry = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if expiry.tzinfo is None:
        return False
    return expiry - FRESHNESS_MARGIN > datetime.now(timezone.utc)


def save(path: Path, entry: Dict[str, Any]) -> None:
    """Atomically replace `path` with `entry`. Raises OSError on failure."""
    CACHE_DIR.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for directory in (CACHE_DIR, path.parent):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
    # mkstemp creates the file 0600, so the token is never readable by anyone
    # else, not even before the rename.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def evict(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def clear_cluster(cluster: str) -> None:
    """Drop every cached credential for `cluster`, whatever platform or key it
    was minted for. Raises OSError on failure."""
    try:
        shutil.rmtree(_cluster_dir(cluster))
    except FileNotFoundError:
        pass
